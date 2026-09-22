import cv2
import numpy as np


class ContourDetector:
    """Detect geometric contours from an edge image."""

    def find_contours(
        self,
        edge_image: np.ndarray,
    ) -> list[np.ndarray]:
        """Find contours in a binary edge image."""
        contours, _ = cv2.findContours(
            edge_image,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        return list(contours)

    def sort_by_area(
        self,
        contours: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Return contours ordered largest to smallest."""
        return sorted(
            contours,
            key=cv2.contourArea,
            reverse=True,
        )

    def find_document_contour(
        self,
        contours: list[np.ndarray],
        image_shape: tuple[int, ...] | None = None,
        min_area_ratio: float = 0.0,
    ) -> np.ndarray | None:
        """Find the largest contour approximating a quadrilateral."""
        contours = self.sort_by_area(contours)

        min_area = 0.0
        if image_shape is not None and min_area_ratio > 0.0:
            h, w = image_shape[:2]
            min_area = (h * w) * min_area_ratio

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            perimeter = cv2.arcLength(
                contour,
                True,
            )

            epsilon = 0.02 * perimeter

            approximation = cv2.approxPolyDP(
                contour,
                epsilon,
                True,
            )

            if len(approximation) == 4:
                return approximation

        return None
