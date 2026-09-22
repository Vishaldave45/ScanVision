import cv2
import numpy as np

from cv_pipeline.ocr.result import OCRResult


def draw_ocr_boxes(
    image: np.ndarray,
    ocr_result: OCRResult,
    min_confidence: float = 0.0,
    box_color: tuple[int, int, int] = (0, 140, 255),  # Orange-amber box
    text_color: tuple[int, int, int] = (0, 0, 255),    # Red text
) -> np.ndarray:
    """Draw bounding boxes and confidence annotations for each recognized word.

    Args:
        image: Scanned document image (BGR).
        ocr_result: OCRResult containing word bounding boxes and confidences.
        min_confidence: Ignore words with confidence strictly lower than this threshold.
        box_color: BGR tuple for word bounding box rectangles.
        text_color: BGR tuple for confidence labels.

    Returns:
        Annotated copy of the input image.
    """
    annotated = image.copy()

    for word in ocr_result.words:
        if word.confidence < min_confidence:
            continue

        x, y, w, h = word.x, word.y, word.width, word.height

        # Draw bounding rectangle around the word
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            box_color,
            2,
        )

        # Draw confidence label above the box
        label = f"{int(word.confidence)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1

        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y = max(y - 3, label_h + 3)

        # Small background badge for readability
        cv2.rectangle(
            annotated,
            (x, label_y - label_h - 2),
            (x + label_w + 2, label_y + baseline),
            (240, 240, 240),
            cv2.FILLED,
        )

        cv2.putText(
            annotated,
            label,
            (x + 1, label_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated
