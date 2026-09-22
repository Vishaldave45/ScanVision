import cv2
import numpy as np

from cv_pipeline.detect.base import DocumentDetector
from cv_pipeline.detect.contour_detector import ContourDetector
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor


class ThresholdDocumentDetector(DocumentDetector):
    """Detect documents using brightness segmentation (thresholding) and morphology.

    Ideal when the document has a noticeably higher (or distinct) brightness
    than the background surface (e.g. white receipt on a wooden table).
    """

    def __init__(
        self,
        blur_kernel: tuple[int, int] = (9, 9),
        min_area_ratio: float = 0.20,
    ) -> None:
        self.blur_kernel = blur_kernel
        self.min_area_ratio = min_area_ratio

        self.preprocessor = ImagePreprocessor()
        self.contour_detector = ContourDetector()

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Segment the document mask using Otsu thresholding on blurred gray image."""
        if len(image.shape) == 3:
            gray = self.preprocessor.to_grayscale(image)
        else:
            gray = image

        # Heavy blur to merge text and paper texture into a single bright region
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        # Otsu thresholding creates a clean binary mask of bright paper vs dark table
        _, mask = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        # Morphological close: fill internal text holes and small gaps in the paper mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find external contours on the mask
        contours = self.contour_detector.find_contours(closed_mask)

        return self.contour_detector.find_document_contour(
            contours,
            image_shape=image.shape,
            min_area_ratio=self.min_area_ratio,
        )
