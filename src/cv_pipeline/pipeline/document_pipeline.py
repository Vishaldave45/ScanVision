from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from cv_pipeline.capture.image_capture import ImageCapture
from cv_pipeline.config.settings import PipelineConfig
from cv_pipeline.detect.base import DocumentDetector
from cv_pipeline.detect.edge_detector import EdgeDocumentDetector
from cv_pipeline.detect.threshold_detector import ThresholdDocumentDetector
from cv_pipeline.ocr.base import OCREngine
from cv_pipeline.ocr.document_preprocessor import DocumentPreprocessor
from cv_pipeline.ocr.result import OCRResult
from cv_pipeline.ocr.tesseract_engine import TesseractEngine
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor
from cv_pipeline.transform.perspective import PerspectiveTransformer


@dataclass
class DocumentPipelineResult:
    """Encapsulates all intermediate and final artifacts from the document scanning pipeline."""
    original_image: np.ndarray
    detected_corners: np.ndarray | None
    document_image: np.ndarray | None
    warped: np.ndarray | None
    otsu: np.ndarray | None
    adaptive: np.ndarray | None
    cleaned: np.ndarray | None
    ocr: OCRResult | None = None

    @property
    def has_document(self) -> bool:
        return self.detected_corners is not None and self.warped is not None

    @property
    def detected_contour(self) -> np.ndarray | None:
        """Backward compatibility alias for detected_corners."""
        return self.detected_corners


class DocumentPipeline:
    """High-level orchestrator for classical document detection, rectification, and OCR preprocessing."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: DocumentDetector | None = None,
        ocr_engine: OCREngine | None = None,
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

        if ocr_engine is not None:
            self.ocr_engine: OCREngine | None = ocr_engine
        else:
            # Default to TesseractEngine with configured parameters
            self.ocr_engine = TesseractEngine(
                psm=self.config.ocr.psm,
                oem=self.config.ocr.oem,
                lang=self.config.ocr.lang,
            )

    def process(self, image: np.ndarray) -> DocumentPipelineResult:
        """Execute pipeline on an in-memory BGR or Grayscale image."""
        doc = self.detector.detect(image)

        if doc is None:
            return DocumentPipelineResult(
                original_image=image,
                detected_corners=None,
                document_image=None,
                warped=None,
                otsu=None,
                adaptive=None,
                cleaned=None,
                ocr=None,
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

        # OCR Recognition: use warped scan by default, or cleaned if explicitly configured
        ocr_result: OCRResult | None = None
        if self.ocr_engine is not None:
            target_ocr_input = cleaned if self.config.ocr.use_preprocessed else warped
            ocr_result = self.ocr_engine.recognize(target_ocr_input)

        return DocumentPipelineResult(
            original_image=image,
            detected_corners=doc,
            document_image=document_image,
            warped=warped,
            otsu=otsu,
            adaptive=adaptive,
            cleaned=cleaned,
            ocr=ocr_result,
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

        # If OCR was performed, render 06_ocr_boxes.jpg
        if result.warped is not None and result.ocr is not None:
            from cv_pipeline.ocr.visualize import draw_ocr_boxes
            ocr_boxes_img = draw_ocr_boxes(result.warped, result.ocr)
            images_to_save["06_ocr_boxes.jpg"] = ocr_boxes_img

        for filename, img in images_to_save.items():
            if img is not None:
                dest = target_dir / filename
                cv2.imwrite(str(dest), img)
                saved[filename] = dest

        return saved
