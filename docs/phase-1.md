# Phase 1: Classical Document Scanner & OCR

## Objective
Build a robust, modular document scanning and text recognition pipeline using classical computer vision and OCR.

## Pipeline Architecture
```text
Image Input
  ↓
Document Detection (Threshold / Edge / Fallback)
  ↓
4-Corner Geometric Ordering
  ↓
Perspective Transformation (Rectification)
  ↓
OCR Processing (Tesseract)
  ↓
OCRResult (Text, Word Bounding Boxes, Confidence)
```

## Key Modules
1. **`detect/`**:
   - `ThresholdDocumentDetector`: High-contrast brightness segmentation using Otsu thresholding + `cv2.RETR_EXTERNAL`.
   - `EdgeDocumentDetector`: Canny edge detection + contour approximation.
   - `FallbackDetector`: Sequential composite detector trying Threshold first, falling back to Edge.
2. **`geometry/`**:
   - `order_points()`: Explicit, coordinate-arithmetic ordering mapping arbitrary 4-point quadrilaterals into `[TL, TR, BR, BL]`.
3. **`transform/`**:
   - `PerspectiveTransformer`: Applies homography matrix warping with strict validation for `NaN`/`Inf`, duplicate coordinates, and non-collinear polygons.
4. **`ocr/`**:
   - `OCREngine`: Protocol decoupling text recognition engines from the pipeline.
   - `TesseractEngine`: Wraps Tesseract using `image_to_data`, extracting word-level bounding boxes and confidence scores.
   - `visualize`: Renders word bounding box rectangles and confidence badges on rectified documents.

## Experimental Findings
- **OCR Preprocessing**: Running OCR directly on the perspective-rectified document (`02_scanned_document.jpg`) yielded higher accuracy than aggressive binary/morphological filtering (`05_ocr_ready_cleaned.jpg`), which tended to erode thin numerical strokes.
- **Page Segmentation Mode (PSM)**: PSM affects layout interpretation rather than character confusion. PSM 3 and 6 yielded the highest word confidences (~91-92%), with PSM 6 naturally capturing tabular flow.
- **Evaluation**: Established character error rate (CER) and word error rate (WER) against ground-truth text, demonstrating that internal OCR confidence is not equivalent to ground-truth accuracy.

## Status
Phase 1 complete.
