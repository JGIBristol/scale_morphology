import pyefd
import numpy as np

from ..scales import efa


def test_point_around_segmentation():
    """
    Check if we can get the expected points around a diamond shape

    """
    square = 255 * np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    x, y = efa.points_around_edge(square, 10)

    expected_x = [5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    expected_y = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0]

    assert np.allclose(x, expected_x)
    assert np.allclose(y, expected_y)


def test_rotation():
    """
    Check the rotation of coefficients works

    """
    coeffs = np.array([[1, -0.5, 1, 0.5], [0.5, 0.6, 0.7, 0.8]])

    assert np.allclose(pyefd.normalize_efd(coeffs.copy(), size_invariant=False), coeffs)
