import cv2
import numpy as np


class ImagePreprocessor:
    """Preprocess images before detection."""

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (5, 5),
    ) -> np.ndarray:
        """Apply Gaussian smoothing."""
        return cv2.GaussianBlur(image, kernel_size, 0)

    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> np.ndarray:
        """Detect edges using Canny."""
        return cv2.Canny(
            image,
            low_threshold,
            high_threshold,
        )
