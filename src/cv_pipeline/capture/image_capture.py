from pathlib import Path

import cv2
import numpy as np


class ImageCapture:
    """Responsible only for acquiring images from files."""

    def read(self, image_path: str | Path) -> np.ndarray:
        """
        Read an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            OpenCV image represented as a NumPy array.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """

        image_path = Path(image_path)

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        return image