"""Contour and document detection module."""

from .base import DocumentDetector
from .contour_detector import ContourDetector
from .edge_detector import EdgeDocumentDetector
from .fallback_detector import FallbackDetector
from .threshold_detector import ThresholdDocumentDetector

__all__ = [
    "ContourDetector",
    "DocumentDetector",
    "EdgeDocumentDetector",
    "FallbackDetector",
    "ThresholdDocumentDetector",
]
