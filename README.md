# ScanVision - Computer Vision Project

A modular computer vision processing pipeline.

## Project Structure

```
ScanVision/
│
├── src/
│   └── cv_pipeline/
│       ├── __init__.py
│       ├── capture/
│       │   └── image_capture.py
│       ├── config/
│       │   └── settings.py
│       ├── detect/
│       │   ├── base.py
│       │   ├── contour_detector.py
│       │   ├── edge_detector.py
│       │   └── threshold_detector.py
│       ├── geometry/
│       │   └── points.py
│       ├── ocr/
│       │   ├── base.py
│       │   ├── document_preprocessor.py
│       │   ├── result.py
│       │   └── tesseract_engine.py
│       ├── pipeline/
│       │   └── document_pipeline.py
│       ├── preprocess/
│       │   └── image_preprocessor.py
│       ├── transform/
│       │   └── perspective.py
│       └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Architecture & Design Principles

```text
                 ScanVision
                     │
                     ▼
             DocumentPipeline
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     Capture      Detection    Transform
                     │
             ┌───────┴───────┐
             ▼               ▼
          Threshold        Edge
                     │
                     ▼
             OCR Preprocessing
                     │
                     ▼
                OCR Engine
```

> **Key Architectural Principle**: Detectors implement a common `DocumentDetector` protocol (`detect(image) -> corners`). The pipeline is decoupled from specific detection algorithms, allowing threshold-based, edge-based, compound `FallbackDetector`, or future deep learning models (YOLO) to be interchanged without modifying downstream perspective rectification or OCR stages.


## Setup & Installation

Using `uv`:
```bash
# Create virtual environment
uv venv --python 3.11

# Activate environment
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Or install dependencies
uv pip install -r requirements.txt pytest
```

## Usage

### Run Document Pipeline
Run against the default input (`data/raw/document.png` or `data/raw/document.jpg`):
```bash
python -m cv_pipeline.main
```

Or pass a specific image file:
```bash
python -m cv_pipeline.main --input path/to/image.jpg
```

To run in headless mode without GUI preview windows:
```bash
python -m cv_pipeline.main --no-display
```

### Run Tests
```bash
pytest -v
```

## Project Status

### Phase 1 — Classical Document Scanner + OCR
**Status: Complete (v0.1.0)**

Implemented:
- Image capture & validation
- Grayscale conversion & Gaussian smoothing
- Canny edge detection & contour approximation
- Threshold-based document detection (Otsu + `cv2.RETR_EXTERNAL`)
- Fallback composite detector
- 4-corner coordinate ordering (`[TL, TR, BR, BL]`)
- Homography perspective rectification with degenerate checks
- OCR preprocessing experiments (Otsu, adaptive, morphology)
- Tesseract OCR integration via decoupled `OCREngine` protocol
- Word-level bounding boxes and confidence score extraction
- Page segmentation mode (PSM) experimentation
- CER & WER quantitative evaluation with `jiwer`
- Bounding-box visualization (`06_ocr_boxes.jpg`)
- 21 automated unit tests

### Phase 2 — Deep Learning Object Detection
**Status: Next**

Planned:
- Pretrained YOLOv8 inference
- Bounding-box parsing & confidence thresholding
- IoU (Intersection over Union) & NMS (Non-Maximum Suppression)
- Custom evaluation metrics (mAP)
- Classical CV vs. Deep Learning comparative benchmark


