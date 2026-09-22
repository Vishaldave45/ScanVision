import numpy as np
import cv2
import pytest

from cv_pipeline.detect.contour_detector import ContourDetector
from cv_pipeline.detect.edge_detector import EdgeDocumentDetector
from cv_pipeline.detect.threshold_detector import ThresholdDocumentDetector


def test_contour_detector_area_filtering():
    detector = ContourDetector()

    # Image shape 1000x1000 => area 1,000,000
    # Create one large square contour (area ~400,000 => 40%)
    large_square = np.array([[[100, 100]], [[700, 100]], [[700, 700]], [[100, 700]]], dtype=np.int32)
    # Create small square (area ~10,000 => 1%)
    small_square = np.array([[[10, 10]], [[110, 10]], [[110, 110]], [[10, 110]]], dtype=np.int32)

    # Filtering with min_area_ratio=0.15 should return the large square
    doc = detector.find_document_contour([small_square, large_square], image_shape=(1000, 1000), min_area_ratio=0.15)
    assert doc is not None
    assert len(doc) == 4

    # Filtering with min_area_ratio=0.50 should reject both
    doc_none = detector.find_document_contour([small_square, large_square], image_shape=(1000, 1000), min_area_ratio=0.50)
    assert doc_none is None


def test_synthetic_threshold_document_detection():
    detector = ThresholdDocumentDetector(min_area_ratio=0.10)

    # Black background with bright document rectangle
    canvas = np.zeros((400, 400), dtype=np.uint8)
    canvas[50:350, 50:350] = 255  # 300x300 white document

    doc = detector.detect(canvas)
    assert doc is not None
    assert len(doc) == 4
