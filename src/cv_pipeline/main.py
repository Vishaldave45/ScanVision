import cv2

from cv_pipeline.capture.image_capture import ImageCapture
from cv_pipeline.detect.contour_detector import ContourDetector
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor
from cv_pipeline.transform.perspective import PerspectiveTransformer


def main() -> None:
    capture = ImageCapture()
    preprocessor = ImagePreprocessor()
    detector = ContourDetector()
    transformer = PerspectiveTransformer()

    image = capture.read("data/raw/document.jpg")

    gray = preprocessor.to_grayscale(image)

    blurred = preprocessor.gaussian_blur(
        gray,
        (5, 5),
    )

    edges = preprocessor.detect_edges(
        blurred,
        50,
        150,
    )

    contours = detector.find_contours(edges)

    document = detector.find_document_contour(contours)

    if document is None:
        print("No document found.")
        return

    warped = transformer.warp(
        image,
        document,
    )

    document_image = image.copy()

    cv2.drawContours(
        document_image,
        [document],
        -1,
        (0, 255, 0),
        3,
    )

    cv2.imshow(
        "Detected Document",
        document_image,
    )

    cv2.imshow(
        "Scanned Document",
        warped,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()