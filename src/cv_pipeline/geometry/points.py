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

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).ravel()

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    ordered[1] = points[np.argmax(differences)]
    ordered[3] = points[np.argmin(differences)]

    return ordered
