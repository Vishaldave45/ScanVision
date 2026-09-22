from typing import Protocol
import numpy as np

from cv_pipeline.ocr.result import OCRResult


class OCREngine(Protocol):
    """Protocol defining the interface for OCR text recognition engines."""

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Extract text and word-level bounding boxes/confidence from image.

        Args:
            image: Input image (Grayscale or BGR).

        Returns:
            OCRResult with full text string and structured OCRWord entries.
        """
        ...
