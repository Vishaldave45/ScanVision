from pathlib import Path
import cv2

from cv_pipeline.ocr.tesseract_engine import TesseractEngine


def run_experiment() -> None:
    image_path = Path("data/processed/05_ocr_ready_cleaned.jpg")
    if not image_path.exists():
        # Fallback to scanned document if 05 doesn't exist
        image_path = Path("data/processed/02_scanned_document.jpg")

    if not image_path.exists():
        print(f"Error: Could not find processed image at {image_path}. Please run pipeline first.")
        return

    print(f"Reading image: {image_path}")
    image = cv2.imread(str(image_path))

    engine = TesseractEngine(psm=3)
    result = engine.recognize(image)

    print("\n" + "=" * 30 + " OCR FULL TEXT " + "=" * 30)
    print(result.text)

    print("\n" + "=" * 30 + f" RECOGNIZED WORDS ({result.word_count}) " + "=" * 30)
    print(f"{'WORD':<30} {'CONFIDENCE':<12} {'BOUNDING BOX (x, y, w, h)'}")
    print("-" * 75)

    for word in result.words:
        print(
            f"{word.text:<30} "
            f"{word.confidence:<12.2f} "
            f"({word.x}, {word.y}, {word.width}, {word.height})"
        )

    print("-" * 75)
    print(f"Average Confidence: {result.average_confidence:.2f}%")


if __name__ == "__main__":
    run_experiment()
