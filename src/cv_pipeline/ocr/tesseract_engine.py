import cv2
import numpy as np
import pytesseract

from cv_pipeline.ocr.result import OCRResult, OCRWord


class TesseractEngine:
    """OCR engine wrapper around Tesseract using pytesseract."""

    def __init__(self, psm: int = 3, oem: int = 3, lang: str = "eng") -> None:
        self.psm = psm
        self.oem = oem
        self.lang = lang

    def recognize(self, image: np.ndarray) -> OCRResult:
        """Extract text and word metadata from an image."""
        if image is None or image.size == 0:
            raise ValueError("OCR image cannot be empty.")

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        config = f"--psm {self.psm} --oem {self.oem}"

        data = pytesseract.image_to_data(
            gray,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )

        words: list[OCRWord] = []

        for i, text in enumerate(data["text"]):
            text = text.strip()

            if not text:
                continue

            try:
                confidence = float(data["conf"][i])
            except (ValueError, TypeError):
                confidence = -1.0

            # Filter out negative confidences which represent blocks/paragraphs without recognized text
            if confidence < 0:
                continue

            words.append(
                OCRWord(
                    text=text,
                    confidence=confidence,
                    x=int(data["left"][i]),
                    y=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                )
            )

        recognized_text = " ".join(word.text for word in words)

        return OCRResult(
            text=recognized_text,
            words=words,
        )
