"""
Tests for the efa module

"""

import pyefd
import numpy as np
from skimage.morphology import disk

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
    coeffs = np.array(
        [
            [1, -0.5, 2, 0.5],
            [0.5, 0.6, 0.7, 0.8],
            [0.1, 0.1, 0.2, 0.3],
        ]
    )

    assert np.allclose(
        pyefd.normalize_efd(coeffs.copy(), size_invariant=False),
        efa._rotate(coeffs),
    )


def test_img_rotation():
    """
    Check that we get the same EFA coefficients for two images that are the same but rotated
    """
    # Generate both images
    empty = np.zeros((32, 32))

    r1, r2 = 4, 6
    disc1 = disk(r1)
    disc2 = disk(r2)

    img1 = empty.copy()
    x1, y1 = 10, 10
    x2, y2 = 15, 18
    img1[y1 - r1 : y1 + r1 + 1, x1 - r1 : x1 + r1 + 1] = disc1
    img1[y2 - r2 : y2 + r2 + 1, x2 - r2 : x2 + r2 + 1] = np.maximum(
        img1[y2 - r2 : y2 + r2 + 1, x2 - r2 : x2 + r2 + 1], disc2
    )

    img2 = np.rot90(img1, k=1)

    img1 = (255 * img1).astype(np.uint8)
    img2 = (255 * img2).astype(np.uint8)

    # Get EFA coeffs for both
    coeffs1 = efa.coefficients(img1, 50, 5)
    coeffs2 = efa.coefficients(img2, 50, 5)

    assert (
        (coeffs1 == coeffs2).all()
    ), "Currently broken because my rotation doesn't do it right?"
