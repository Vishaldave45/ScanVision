from typing import Sequence
import numpy as np

from cv_pipeline.detect.base import DocumentDetector


class FallbackDetector(DocumentDetector):
    """Executes a chain of detectors in order, returning the first detected document."""

    def __init__(self, detectors: Sequence[DocumentDetector]) -> None:
        self.detectors = list(detectors)

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        for detector in self.detectors:
            doc = detector.detect(image)
            if doc is not None:
                return doc
        return None
