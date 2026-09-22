from typing import Protocol
import numpy as np


class DocumentDetector(Protocol):
    """Protocol defining the interface for document corner detectors."""

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        """Detect the 4 corners of a document in the image.

        Args:
            image: Input image (BGR or Grayscale).

        Returns:
            np.ndarray of shape (4, 1, 2) or (4, 2) with corner coordinates,
            or None if no document candidate is found.
        """
        ...
