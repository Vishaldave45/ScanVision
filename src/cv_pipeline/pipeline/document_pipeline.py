from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from cv_pipeline.capture.image_capture import ImageCapture
from cv_pipeline.config.settings import PipelineConfig
from cv_pipeline.detect.base import DocumentDetector
from cv_pipeline.detect.edge_detector import EdgeDocumentDetector
from cv_pipeline.detect.threshold_detector import ThresholdDocumentDetector
from cv_pipeline.ocr.document_preprocessor import DocumentPreprocessor
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor
from cv_pipeline.transform.perspective import PerspectiveTransformer


@dataclass
class DocumentPipelineResult:
    """Encapsulates all intermediate and final artifacts from the document scanning pipeline."""
    original_image: np.ndarray
    detected_contour: np.ndarray | None
    document_image: np.ndarray | None
    warped: np.ndarray | None
    otsu: np.ndarray | None
    adaptive: np.ndarray | None
    cleaned: np.ndarray | None

    @property
    def has_document(self) -> bool:
        return self.detected_contour is not None and self.warped is not None


class DocumentPipeline:
    """High-level orchestrator for classical document detection, rectification, and OCR preprocessing."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: DocumentDetector | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.capture = ImageCapture()
        self.preprocessor = ImagePreprocessor()
        self.transformer = PerspectiveTransformer()
        self.ocr_preprocessor = DocumentPreprocessor()

        if detector is not None:
            self.detector = detector
        else:
            # Default fallback detector composition: Threshold detector first, Edge detector fallback
            det_cfg = self.config.detection
            thresh_det = ThresholdDocumentDetector(
                blur_kernel=det_cfg.thresh_blur_kernel,
                min_area_ratio=det_cfg.min_area_ratio,
                morph_kernel_size=det_cfg.morph_kernel_size,
            )
            edge_det = EdgeDocumentDetector(
                low_threshold=det_cfg.canny_low,
                high_threshold=det_cfg.canny_high,
                blur_kernel=det_cfg.edge_blur_kernel,
                min_area_ratio=det_cfg.min_area_ratio,
            )
            from cv_pipeline.detect.fallback_detector import FallbackDetector
            self.detector = FallbackDetector([thresh_det, edge_det])

    def process(self, image: np.ndarray) -> DocumentPipelineResult:
        """Execute pipeline on an in-memory BGR or Grayscale image."""
        doc = self.detector.detect(image)

        if doc is None:
            return DocumentPipelineResult(
                original_image=image,
                detected_contour=None,
                document_image=None,
                warped=None,
                otsu=None,
                adaptive=None,
                cleaned=None,
            )

        # Draw contour on a copy of original
        document_image = image.copy()
        contour_draw = doc.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(document_image, [contour_draw], -1, (0, 255, 0), 3)

        # Perspective transform
        warped = self.transformer.warp(image, doc)

        # OCR Preprocessing
        warped_gray = self.preprocessor.to_grayscale(warped)
        otsu = self.ocr_preprocessor.otsu_threshold(warped_gray)

        ocr_cfg = self.config.ocr_preprocess
        adaptive = self.ocr_preprocessor.adaptive_threshold(
            warped_gray,
            block_size=ocr_cfg.adaptive_block_size,
            constant=ocr_cfg.adaptive_constant,
        )
        cleaned = self.ocr_preprocessor.morphological_open(
            adaptive,
            kernel_size=ocr_cfg.opening_kernel_size,
        )

        return DocumentPipelineResult(
            original_image=image,
            detected_contour=doc,
            document_image=document_image,
            warped=warped,
            otsu=otsu,
            adaptive=adaptive,
            cleaned=cleaned,
        )

    def process_file(self, file_path: str | Path) -> DocumentPipelineResult:
        """Load image from disk and process it."""
        image = self.capture.read(file_path)
        return self.process(image)

    def save_artifacts(
        self,
        result: DocumentPipelineResult,
        output_dir: Path | None = None,
    ) -> dict[str, Path]:
        """Save intermediate and final stages to output directory."""
        target_dir = output_dir or self.config.processed_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if not result.has_document:
            return {}

        saved: dict[str, Path] = {}
        images_to_save = {
            "01_detected_document.jpg": result.document_image,
            "02_scanned_document.jpg": result.warped,
            "03_otsu_threshold.jpg": result.otsu,
            "04_adaptive_threshold.jpg": result.adaptive,
            "05_ocr_ready_cleaned.jpg": result.cleaned,
        }

        for filename, img in images_to_save.items():
            if img is not None:
                dest = target_dir / filename
                cv2.imwrite(str(dest), img)
                saved[filename] = dest

        return saved
