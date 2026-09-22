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
        if image is None or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array.")

        try:
            points_reshaped = points.reshape(4, 2)
        except ValueError as err:
            raise ValueError(
                f"Expected 4 points of shape (4, 2) or (4, 1, 2), got shape {points.shape}"
            ) from err

        if not np.all(np.isfinite(points_reshaped)):
            raise ValueError("Input points contain NaN or Inf values.")

        # Check for duplicate corners
        if len(np.unique(points_reshaped, axis=0)) < 4:
            raise ValueError("Input points must contain 4 distinct non-duplicate coordinates.")

        rect = order_points(points_reshaped)

        # Collinear / zero area check: quadrilateral area must be strictly positive
        quad_area = cv2.contourArea(rect)
        if quad_area <= 0:
            raise ValueError("Input points are collinear or form a degenerate polygon.")

        top_left, top_right, bottom_right, bottom_left = rect

        width_top = np.linalg.norm(top_right - top_left)
        width_bottom = np.linalg.norm(bottom_right - bottom_left)

        output_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(bottom_left - top_left)
        height_right = np.linalg.norm(bottom_right - top_right)

        output_height = int(max(height_left, height_right))

        if output_width <= 0 or output_height <= 0:
            raise ValueError(
                f"Computed invalid output dimensions: width={output_width}, height={output_height}"
            )

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
