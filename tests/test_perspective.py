import numpy as np
import pytest

from cv_pipeline.transform.perspective import PerspectiveTransformer


def test_warp_synthetic_rectangle():
    transformer = PerspectiveTransformer()
    # 200x300 image
    image = np.ones((200, 300, 3), dtype=np.uint8) * 100

    # Corners of a 100x150 subregion
    corners = np.array(
        [[20, 30], [120, 30], [120, 180], [20, 180]],
        dtype=np.float32,
    )

    warped = transformer.warp(image, corners)
    assert warped.ndim == 3
    assert warped.shape[2] == 3
    # Width approx 100, Height approx 150
    assert abs(warped.shape[1] - 100) <= 2
    assert abs(warped.shape[0] - 150) <= 2


def test_warp_invalid_points():
    transformer = PerspectiveTransformer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    # 3 points instead of 4
    with pytest.raises(ValueError, match="Expected 4 points"):
        transformer.warp(image, np.array([[0, 0], [10, 0], [10, 10]]))


def test_warp_nan_or_inf():
    transformer = PerspectiveTransformer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    nan_points = np.array([[0, 0], [np.nan, 0], [10, 10], [0, 10]], dtype=np.float32)

    with pytest.raises(ValueError, match="NaN or Inf"):
        transformer.warp(image, nan_points)


def test_warp_duplicate_points():
    transformer = PerspectiveTransformer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    dup_points = np.array([[10, 10], [10, 10], [50, 10], [10, 50]], dtype=np.float32)

    with pytest.raises(ValueError, match="distinct non-duplicate"):
        transformer.warp(image, dup_points)


def test_warp_collinear_points():
    transformer = PerspectiveTransformer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # 4 collinear points along a line
    collinear = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)

    with pytest.raises(ValueError, match="collinear or form a degenerate"):
        transformer.warp(image, collinear)

