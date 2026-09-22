from pathlib import Path
import cv2

from cv_pipeline.ocr.tesseract_engine import TesseractEngine

IMAGES = [
    "data/processed/02_scanned_document.jpg",
    "data/processed/03_otsu_threshold.jpg",
    "data/processed/04_adaptive_threshold.jpg",
    "data/processed/05_ocr_ready_cleaned.jpg",
]


def main() -> None:
    engine = TesseractEngine(psm=3)

    for img_path_str in IMAGES:
        image_path = Path(img_path_str)
        print("\n" + "=" * 80)
        print(f"OCR: {image_path.name}")
        print("=" * 80)

        if not image_path.exists():
            print(f"Could not read: {image_path}")
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to decode: {image_path}")
            continue

        result = engine.recognize(image)

        print(f"Word count: {result.word_count}")
        if result.words:
            # All detected words average
            avg_conf_all = result.average_confidence
            # Exclude 0 confidence words
            nonzero_words = [w for w in result.words if w.confidence > 0]
            avg_conf_nonzero = (
                sum(w.confidence for w in nonzero_words) / len(nonzero_words)
                if nonzero_words
                else 0.0
            )

            print(f"Average word confidence (all):     {avg_conf_all:.2f}%")
            print(f"Average word confidence (nonzero): {avg_conf_nonzero:.2f}% (from {len(nonzero_words)} words)")

        print("\nRECOGNIZED TEXT:")
        print(result.text if result.text else "<No text recognized>")


if __name__ == "__main__":
    main()
