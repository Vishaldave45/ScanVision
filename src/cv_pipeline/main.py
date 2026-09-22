import cv2

from cv_pipeline.capture.image_capture import ImageCapture
from cv_pipeline.preprocess.image_preprocessor import ImagePreprocessor


def main() -> None:
    capture = ImageCapture()
    preprocessor = ImagePreprocessor()

    image = capture.read("data/raw/document.jpg")

    gray = preprocessor.to_grayscale(image)
    blurred = preprocessor.gaussian_blur(gray)

    print("Original:", image.shape)
    print("Grayscale:", gray.shape)
    print("Blurred:", blurred.shape)

    cv2.imshow("Original", image)
    cv2.imshow("Grayscale", gray)
    cv2.imshow("Blurred", blurred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()