import numpy as np
import pytest

from cv_pipeline.ocr.result import OCRResult, OCRWord


def test_ocr_result_properties():
    words = [
        OCRWord(text="Invoice", confidence=95.0, x=10, y=20, width=50, height=15),
        OCRWord(text="Total", confidence=85.0, x=10, y=40, width=40, height=15),
    ]
    result = OCRResult(text="Invoice Total", words=words)

    assert result.word_count == 2
    assert result.average_confidence == 90.0


def test_ocr_result_empty():
    result = OCRResult(text="", words=[])
    assert result.word_count == 0
    assert result.average_confidence == 0.0
