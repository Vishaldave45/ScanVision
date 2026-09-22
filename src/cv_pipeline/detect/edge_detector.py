import cv2
import numpy as np

from cv_pipeline.detect.base import DocumentDetector
from cv_pipeline.detect.contour_detector import ContourDetector
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor


class EdgeDocumentDetector(DocumentDetector):
    """Detect documents using Canny edge detection and contour approximation."""

    def __init__(
        self,
        low_threshold: int = 50,
        high_threshold: int = 150,
        blur_kernel: tuple[int, int] = (5, 5),
        min_area_ratio: float = 0.20,
    ) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.blur_kernel = blur_kernel
        self.min_area_ratio = min_area_ratio

        self.preprocessor = ImagePreprocessor()
        self.contour_detector = ContourDetector()

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect document quadrilateral using Canny edges."""
        if len(image.shape) == 3:
            gray = self.preprocessor.to_grayscale(image)
        else:
            gray = image

        blurred = self.preprocessor.gaussian_blur(gray, self.blur_kernel)
        edges = self.preprocessor.detect_edges(
            blurred,
            self.low_threshold,
            self.high_threshold,
        )

        contours = self.contour_detector.find_contours(edges)
        return self.contour_detector.find_document_contour(
            contours,
            image_shape=image.shape,
            min_area_ratio=self.min_area_ratio,
        )
