import numpy as np
import pytest

from cv_pipeline.geometry.points import order_points


def test_order_points_basic():
    # Regular rectangle: TL=(0,0), TR=(100,0), BR=(100,200), BL=(0,200)
    expected_tl = [0.0, 0.0]
    expected_tr = [100.0, 0.0]
    expected_br = [100.0, 200.0]
    expected_bl = [0.0, 200.0]

    # Provide points scrambled
    points = np.array([expected_br, expected_tl, expected_bl, expected_tr], dtype=np.float32)
    ordered = order_points(points)

    assert np.allclose(ordered[0], expected_tl)
    assert np.allclose(ordered[1], expected_tr)
    assert np.allclose(ordered[2], expected_br)
    assert np.allclose(ordered[3], expected_bl)


def test_order_points_permuted():
    tl = [50.0, 60.0]
    tr = [300.0, 50.0]
    br = [310.0, 450.0]
    bl = [40.0, 460.0]

    scrambled = np.array([tr, bl, tl, br], dtype=np.float32)
    ordered = order_points(scrambled)

    assert np.allclose(ordered[0], tl)
    assert np.allclose(ordered[1], tr)
    assert np.allclose(ordered[2], br)
    assert np.allclose(ordered[3], bl)
