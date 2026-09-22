"""OCR processing package."""

from .base import OCREngine
from .document_preprocessor import DocumentPreprocessor
from .result import OCRResult, OCRWord
from .tesseract_engine import TesseractEngine
from .visualize import draw_ocr_boxes

__all__ = [
    "DocumentPreprocessor",
    "OCREngine",
    "OCRResult",
    "OCRWord",
    "TesseractEngine",
    "draw_ocr_boxes",
]
