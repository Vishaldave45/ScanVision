import numpy as np
import pytest

from cv_pipeline.ocr.result import OCRResult, OCRWord
from cv_pipeline.ocr.tesseract_engine import TesseractEngine


def test_tesseract_engine_empty_input():
    engine = TesseractEngine()
    empty = np.array([], dtype=np.uint8)

    with pytest.raises(ValueError, match="cannot be empty"):
        engine.recognize(empty)


def test_ocr_word_attributes():
    word = OCRWord(text="TOTAL", confidence=98.5, x=100, y=200, width=50, height=20)
    assert word.text == "TOTAL"
    assert word.confidence == 98.5
    assert word.x == 100
    assert word.y == 200
    assert word.width == 50
    assert word.height == 20


def test_ocr_result_aggregations():
    words = [
        OCRWord("Invoice", 90.0, 10, 10, 40, 15),
        OCRWord("No", 80.0, 55, 10, 20, 15),
    ]
    res = OCRResult(text="Invoice No", words=words)
    assert res.word_count == 2
    assert res.average_confidence == 85.0


def test_draw_ocr_boxes():
    from cv_pipeline.ocr.visualize import draw_ocr_boxes

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    words = [OCRWord("TEST", 95.0, 10, 10, 30, 20)]
    res = OCRResult(text="TEST", words=words)

    annotated = draw_ocr_boxes(image, res)
    assert annotated.shape == image.shape
    assert annotated.dtype == image.dtype
    # The image should no longer be purely black (boxes/text drawn)
    assert np.any(annotated > 0)

