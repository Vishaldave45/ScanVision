import cv2
import numpy as np

from cv_pipeline.geometry.points import order_points


class PerspectiveTransformer:
    """Transform a quadrilateral into a rectangular view."""

    def warp(
        self,
        image: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Apply perspective correction to a four-point region."""
        rect = order_points(points)

        top_left, top_right, bottom_right, bottom_left = rect

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)

        output_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        output_height = int(max(height_left, height_right))

        destination = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            rect,
            destination,
        )

        warped = cv2.warpPerspective(
            image,
            matrix,
            (output_width, output_height),
        )

        return warped
