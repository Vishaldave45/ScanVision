"""Contour and document detection module."""

from .base import DocumentDetector
from .contour_detector import ContourDetector
from .edge_detector import EdgeDocumentDetector
from .threshold_detector import ThresholdDocumentDetector

__all__ = [
    "ContourDetector",
    "DocumentDetector",
    "EdgeDocumentDetector",
    "ThresholdDocumentDetector",
]
