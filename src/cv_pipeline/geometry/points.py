import numpy as np


def order_points(
    points: np.ndarray,
) -> np.ndarray:
    """Order four points as:

    top-left
    top-right
    bottom-right
    bottom-left
    """
    points = points.reshape(4, 2)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    sums = points[:, 0] + points[:, 1]
    differences = points[:, 0] - points[:, 1]

    ordered[0] = points[np.argmin(sums)]          # Top-Left: smallest (x + y)
    ordered[2] = points[np.argmax(sums)]          # Bottom-Right: largest (x + y)
    ordered[1] = points[np.argmax(differences)]   # Top-Right: largest (x - y)
    ordered[3] = points[np.argmin(differences)]   # Bottom-Left: smallest (x - y)

    return ordered
