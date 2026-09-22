from pathlib import Path
import cv2

from cv_pipeline.ocr.tesseract_engine import TesseractEngine

IMAGE_PATH = Path("data/processed/02_scanned_document.jpg")
PSM_MODES = [3, 6, 11]

PSM_DESCRIPTIONS = {
    3: "Fully automatic page segmentation, but no OSD (Default)",
    6: "Assume a single uniform block of text",
    11: "Sparse text. Find as much text as possible in no particular order",
}


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Target image not found at: {IMAGE_PATH}")

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise ValueError(f"Failed to decode image at {IMAGE_PATH}")

    print("=" * 80)
    print(f"PSM COMPARISON EXPERIMENT ON: {IMAGE_PATH.name}")
    print("=" * 80)

    for psm in PSM_MODES:
        print("\n" + "#" * 80)
        print(f"PSM {psm}: {PSM_DESCRIPTIONS.get(psm, '')}")
        print("#" * 80)

        engine = TesseractEngine(psm=psm)
        result = engine.recognize(image)

        print(f"Word count: {result.word_count}")

        if result.words:
            avg_all = result.average_confidence
            nonzero = [w for w in result.words if w.confidence > 0]
            avg_nonzero = (
                sum(w.confidence for w in nonzero) / len(nonzero)
                if nonzero
                else 0.0
            )

            print(f"Average confidence (all):     {avg_all:.2f}%")
            print(f"Average confidence (nonzero): {avg_nonzero:.2f}% (from {len(nonzero)} words)")

        print("\nRECOGNIZED TEXT:")
        print(result.text if result.text else "<No text recognized>")


if __name__ == "__main__":
    main()
