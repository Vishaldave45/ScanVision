from pathlib import Path
import cv2

from cv_pipeline.capture.image_capture import ImageCapture
from cv_pipeline.detect import EdgeDocumentDetector, ThresholdDocumentDetector
from cv_pipeline.ocr.document_preprocessor import DocumentPreprocessor
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor
from cv_pipeline.transform.perspective import PerspectiveTransformer


def main() -> None:
    capture = ImageCapture()
    preprocessor = ImagePreprocessor()
    transformer = PerspectiveTransformer()
    ocr_preprocessor = DocumentPreprocessor()

    input_path = Path("data/raw/document.png") if Path("data/raw/document.png").exists() else Path("data/raw/document.jpg")
    image = capture.read(input_path)

    # Compare detectors
    edge_detector = EdgeDocumentDetector(min_area_ratio=0.15)
    threshold_detector = ThresholdDocumentDetector(min_area_ratio=0.15)

    doc_edge = edge_detector.detect(image)
    doc_thresh = threshold_detector.detect(image)

    print(f"Edge-based detection found document: {doc_edge is not None}")
    print(f"Threshold-based detection found document: {doc_thresh is not None}")

    # Use the threshold detector result (or fall back to edge detector)
    document = doc_thresh if doc_thresh is not None else doc_edge

    if document is None:
        print("No document found by either detector.")
        return

    from cv_pipeline.geometry.points import order_points

    ordered = order_points(document)
    print("\n--- Geometry Debugging ---")
    print(f"Input shape:    {image.shape}")
    print(f"Detected points:\n{document.reshape(-1, 2)}")
    print("Ordered corners:")
    print("  Top-left (TL):     ", ordered[0])
    print("  Top-right (TR):    ", ordered[1])
    print("  Bottom-right (BR): ", ordered[2])
    print("  Bottom-left (BL):  ", ordered[3])

    warped = transformer.warp(
        image,
        document,
    )
    print(f"Warped shape:   {warped.shape}\n--------------------------")

    # OCR Preprocessing
    warped_gray = preprocessor.to_grayscale(warped)

    otsu = ocr_preprocessor.otsu_threshold(warped_gray)

    adaptive = ocr_preprocessor.adaptive_threshold(
        warped_gray,
        block_size=11,
        constant=2,
    )

    cleaned = ocr_preprocessor.morphological_open(
        adaptive,
        kernel_size=(3, 3),
    )

    # Save output images
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    document_image = image.copy()
    cv2.drawContours(
        document_image,
        [document],
        -1,
        (0, 255, 0),
        3,
    )

    saved_images = {
        "01_detected_document.jpg": document_image,
        "02_scanned_document.jpg": warped,
        "03_otsu_threshold.jpg": otsu,
        "04_adaptive_threshold.jpg": adaptive,
        "05_ocr_ready_cleaned.jpg": cleaned,
    }

    for filename, img in saved_images.items():
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), img)
        print(f"Saved: {save_path}")

    # Visualization
    windows = {
        "Detected Document": document_image,
        "Scanned Document": warped,
        "Otsu Threshold": otsu,
        "Adaptive Threshold": adaptive,
        "OCR Ready (Cleaned)": cleaned,
    }

    for win_name, win_img in windows.items():
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 1000)
        cv2.imshow(win_name, win_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()