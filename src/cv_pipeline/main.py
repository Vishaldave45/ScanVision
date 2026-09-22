import argparse
from pathlib import Path
import cv2

from cv_pipeline.config.settings import PipelineConfig
from cv_pipeline.geometry.points import order_points
from cv_pipeline.pipeline.document_pipeline import DocumentPipeline, DocumentPipelineResult


def display_results(result: DocumentPipelineResult) -> None:
    """Display intermediate stages in OpenCV windows."""
    if not result.has_document:
        print("No document detected to display.")
        return

    windows = {
        "Detected Document": result.document_image,
        "Scanned Document": result.warped,
        "Otsu Threshold": result.otsu,
        "Adaptive Threshold": result.adaptive,
        "OCR Ready (Cleaned)": result.cleaned,
    }

    if result.warped is not None and result.ocr is not None:
        from cv_pipeline.ocr.visualize import draw_ocr_boxes
        windows["OCR Bounding Boxes"] = draw_ocr_boxes(result.warped, result.ocr)

    for win_name, win_img in windows.items():
        if win_img is not None:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 800, 1000)
            cv2.imshow(win_name, win_img)

    print("Press any key in a preview window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resolve_input_path(raw_dir: Path, requested_path: str | None = None) -> Path:
    """Resolve input file from arguments or default candidates in raw_dir."""
    if requested_path:
        path = Path(requested_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified input file not found: {requested_path}")

    candidates = [
        raw_dir / "document.png",
        raw_dir / "document.jpg",
        raw_dir / "document.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No sample image found in {raw_dir}. Please place 'document.png' or 'document.jpg' there."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="ScanVision Document Processing Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input document image (default: data/raw/document.[png|jpg])",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Skip GUI window display (useful in headless/CI environments)",
    )
    args = parser.parse_args()

    config = PipelineConfig()
    pipeline = DocumentPipeline(config=config)

    input_file = resolve_input_path(config.raw_dir, args.input)
    print(f"Processing document: {input_file}")

    result = pipeline.process_file(input_file)

    if not result.has_document:
        print("No document detected.")
        return

    print("Document successfully detected and rectified.")
    if result.detected_corners is not None and result.warped is not None:
        ordered = order_points(result.detected_corners)
        print(f"  Input shape:  {result.original_image.shape}")
        print(f"  Warped shape: {result.warped.shape}")
        print(f"  Corners:\n    TL: {ordered[0]}\n    TR: {ordered[1]}\n    BR: {ordered[2]}\n    BL: {ordered[3]}")

    saved = pipeline.save_artifacts(result)
    for name, path in saved.items():
        print(f"Saved artifact: {path}")

    if result.ocr is not None:
        print("\n--- OCR Results ---")
        print(f"Recognized words: {result.ocr.word_count}")
        print(f"Average confidence: {result.ocr.average_confidence:.2f}%")
        print("Recognized Text Preview:")
        preview = (result.ocr.text[:200] + "...") if len(result.ocr.text) > 200 else result.ocr.text
        print(f"  {preview}")
        print("-------------------")

    if not args.no_display:
        display_results(result)


if __name__ == "__main__":
    main()