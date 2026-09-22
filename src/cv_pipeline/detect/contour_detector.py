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

        return contours

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
    ) -> np.ndarray | None:
        """Find the largest contour approximating a quadrilateral."""
        contours = self.sort_by_area(contours)

        for contour in contours:
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
