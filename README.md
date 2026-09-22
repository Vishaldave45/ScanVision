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
│       │   ├── __init__.py
│       │   └── image_capture.py
│       │
│       └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── tests/
│
├── notebooks/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Installation

Using `uv`:
```bash
# Create virtual environment
uv venv --python 3.11

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

Run the main pipeline:
```bash
python -m src.cv_pipeline.main
```
