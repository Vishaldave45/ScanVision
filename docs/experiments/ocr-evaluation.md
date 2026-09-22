# OCR Evaluation & Experimental Analysis

## 1. Preprocessing Comparison (PSM 3)

| Input Representation | Word Count | Avg. Confidence (All) | Avg. Confidence (Non-Zero) | Qualitative Observations |
|---|---|---|---|---|
| `02_scanned_document.jpg` | 74 | **91.76%** | **91.76%** | Sharpest characters, highest stroke fidelity |
| `03_otsu_threshold.jpg` | 74 | 89.82% | 89.82% | Clean binary background, minor stroke thinning |
| `04_adaptive_threshold.jpg` | 84 | 84.95% | 85.98% | Introduces border noise artifacts (`ll | | | | |`) |
| `05_ocr_ready_cleaned.jpg` | 76 | 64.87% | 71.45% | Numerical characters corrupted by morphology (`10,000` → `10,000.90`) |

## 2. Page Segmentation Mode (PSM) Comparison on Scanned Document

| PSM Mode | Layout Assumption | Words | Avg. Confidence | CER (%) | WER (%) |
|---|---|---|---|---|---|
| **PSM 3** | Fully automatic page segmentation | 74 | **91.76%** | 21.51% | 20.78% |
| **PSM 6** | Single uniform block of text | 69 | 91.20% | 13.02% | 14.29% |
| **PSM 11** | Sparse text (no layout assumption) | 77 | 88.48% | **6.79%** | **12.99%** |

## 3. Key Takeaways
1. **Never Assume Preprocessing Always Helps**: Aggressive thresholding and morphological opening can severely degrade OCR accuracy by eroding fine strokes.
2. **Confidence != Accuracy**: High internal word confidence does not guarantee ground-truth correctness (e.g. `0` vs `O`).
3. **PSM Decoupling**: PSM controls layout interpretation, not character recognition.
