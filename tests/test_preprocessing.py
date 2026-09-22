import numpy as np
import pytest

from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor
from cv_pipeline.ocr.document_preprocessor import DocumentPreprocessor


def test_to_grayscale_from_bgr():
    prep = ImagePreprocessor()
    bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr[:, :, 0] = 255  # Blue image
    gray = prep.to_grayscale(bgr)

    assert gray.shape == (100, 100)
    assert gray.dtype == np.uint8


def test_to_grayscale_idempotent():
    prep = ImagePreprocessor()
    orig_gray = np.full((100, 100), 128, dtype=np.uint8)
    gray = prep.to_grayscale(orig_gray)

    assert gray.shape == (100, 100)
    assert np.array_equal(gray, orig_gray)


def test_ocr_thresholds():
    ocr_prep = DocumentPreprocessor()
    # Create image with two distinct luminance regions
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:80, 20:80] = 220

    otsu = ocr_prep.otsu_threshold(img)
    assert otsu.dtype == np.uint8
    unique_vals = set(np.unique(otsu))
    assert unique_vals.issubset({0, 255})

    adaptive = ocr_prep.adaptive_threshold(img, block_size=11, constant=2)
    assert adaptive.dtype == np.uint8
    unique_adaptive = set(np.unique(adaptive))
    assert unique_adaptive.issubset({0, 255})
