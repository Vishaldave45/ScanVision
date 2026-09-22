import cv2

from cv_pipeline.capture.image_capture import ImageCapture


def main() -> None:
    capture = ImageCapture()

    image = capture.read("data/raw/document.jpg")

    print("Image loaded successfully")
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")

    cv2.imshow("Original Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()