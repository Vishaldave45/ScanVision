import cv2
import numpy as np


class DocumentPreprocessor:
    """Prepare document images for OCR."""

    def otsu_threshold(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply Otsu's global thresholding."""
        _, binary = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return binary

    def adaptive_threshold(
        self,
        image: np.ndarray,
        block_size: int = 11,
        constant: int = 2,
    ) -> np.ndarray:
        """Apply adaptive Gaussian thresholding."""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            constant,
        )

    def morphological_open(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (3, 3),
    ) -> np.ndarray:
        """Remove small foreground noise."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size,
        )

        return cv2.morphologyEx(
            image,
            cv2.MORPH_OPEN,
            kernel,
        )

    def morphological_close(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (3, 3),
    ) -> np.ndarray:
        """Close small gaps in foreground regions."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            kernel_size,
        )

        return cv2.morphologyEx(
            image,
            cv2.MORPH_CLOSE,
            kernel,
        )
