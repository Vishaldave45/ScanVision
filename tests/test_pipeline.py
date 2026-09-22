import numpy as np
import pytest

from cv_pipeline.pipeline.document_pipeline import DocumentPipeline
from cv_pipeline.detect.base import DocumentDetector


class MockDocumentDetector:
    """Mock detector that returns a known rectangle regardless of image."""

    def __init__(self, corners: np.ndarray | None) -> None:
        self.corners = corners

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        return self.corners


def test_pipeline_with_mock_detector():
    """Verify DocumentPipeline respects an injected detector strategy."""
    # Synthetic 400x400 image
    image = np.full((400, 400, 3), 120, dtype=np.uint8)

    mock_corners = np.array(
        [[50, 50], [250, 50], [250, 350], [50, 350]],
        dtype=np.float32,
    )
    mock_detector = MockDocumentDetector(corners=mock_corners)

    pipeline = DocumentPipeline(detector=mock_detector)
    result = pipeline.process(image)

    assert result.has_document is True
    assert result.detected_corners is not None
    assert result.detected_contour is not None
    assert result.warped is not None
    assert result.otsu is not None
    assert result.adaptive is not None
    assert result.cleaned is not None

    # Warped shape should match width ~200, height ~300
    assert abs(result.warped.shape[1] - 200) <= 2
    assert abs(result.warped.shape[0] - 300) <= 2


def test_pipeline_no_document_detected():
    """Verify DocumentPipeline gracefully handles images where no document is found."""
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mock_detector = MockDocumentDetector(corners=None)

    pipeline = DocumentPipeline(detector=mock_detector)
    result = pipeline.process(image)

    assert result.has_document is False
    assert result.detected_contour is None
    assert result.warped is None
    assert result.otsu is None
    assert result.adaptive is None
    assert result.cleaned is None
    assert result.ocr is None


class MockOCREngine:
    """Mock OCR engine returning fixed test words."""

    def recognize(self, image: np.ndarray):
        from cv_pipeline.ocr.result import OCRResult, OCRWord
        words = [
            OCRWord(text="TAX", confidence=95.0, x=10, y=10, width=40, height=20),
            OCRWord(text="INVOICE", confidence=90.0, x=55, y=10, width=60, height=20),
        ]
        return OCRResult(text="TAX INVOICE", words=words)


def test_pipeline_with_injected_ocr_engine():
    """Verify DocumentPipeline integrates an injected OCREngine seamlessly."""
    image = np.full((300, 300, 3), 100, dtype=np.uint8)
    mock_corners = np.array(
        [[20, 20], [200, 20], [200, 200], [20, 200]],
        dtype=np.float32,
    )
    mock_detector = MockDocumentDetector(corners=mock_corners)
    mock_ocr = MockOCREngine()

    pipeline = DocumentPipeline(detector=mock_detector, ocr_engine=mock_ocr)
    result = pipeline.process(image)

    assert result.has_document is True
    assert result.ocr is not None
    assert result.ocr.text == "TAX INVOICE"
    assert result.ocr.word_count == 2
    assert result.ocr.average_confidence == 92.5

