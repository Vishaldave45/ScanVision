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
│       │   └── document_preprocessor.py
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

