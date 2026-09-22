from pathlib import Path
import re
import cv2
import jiwer

from cv_pipeline.ocr.tesseract_engine import TesseractEngine

GROUND_TRUTH_PATH = Path("experiments/ground_truth.txt")
IMAGE_PATH = Path("data/processed/02_scanned_document.jpg")
PSM_MODES = [3, 6, 11]


def normalize_text(text: str) -> str:
    """Standardize whitespace and strip outer punctuation/spacing for fair comparison."""
    # Collapse multiple spaces and newlines into single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    raw_ground_truth = GROUND_TRUTH_PATH.read_text(encoding="utf-8")
    ground_truth_norm = normalize_text(raw_ground_truth)

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise ValueError(f"Could not load image: {IMAGE_PATH}")

    print("=" * 80)
    print("OCR QUANTITATIVE EVALUATION (CER & WER)")
    print(f"Target Image: {IMAGE_PATH.name}")
    print("=" * 80)
    print(f"Ground Truth Length: {len(ground_truth_norm)} chars, {len(ground_truth_norm.split())} words\n")

    results = []

    for psm in PSM_MODES:
        engine = TesseractEngine(psm=psm)
        ocr_result = engine.recognize(image)
        ocr_norm = normalize_text(ocr_result.text)

        # Calculate Word Error Rate (WER) & Character Error Rate (CER)
        wer = jiwer.wer(ground_truth_norm, ocr_norm)
        cer = jiwer.cer(ground_truth_norm, ocr_norm)

        results.append({
            "psm": psm,
            "word_count": ocr_result.word_count,
            "avg_conf": ocr_result.average_confidence,
            "wer": wer,
            "cer": cer,
            "text": ocr_norm,
        })

    # Print Summary Table
    print(f"{'PSM':<6} {'WORDS':<8} {'AVG CONF':<12} {'CER (%)':<12} {'WER (%)':<12}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['psm']:<6} "
            f"{r['word_count']:<8} "
            f"{r['avg_conf']:<12.2f} "
            f"{r['cer'] * 100:<12.2f} "
            f"{r['wer'] * 100:<12.2f}"
        )
    print("-" * 55)

    print("\nDetailed Differences (PSM 3 vs PSM 6):")
    for r in results:
        print(f"\n--- PSM {r['psm']} Sample Extract ---")
        print(r['text'][:250] + "...")


if __name__ == "__main__":
    main()
