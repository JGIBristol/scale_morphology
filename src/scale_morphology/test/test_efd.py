"""
Tests for the efa module

"""

import pyefd
import pytest
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
    # Specially constructed coeffs matrix
    # Needs to be rotated, but handedness doesn't
    # need to be fixed (the pyefd one doesn't do this)
    coeffs = np.array(
        [
            [2, 0.5, 1, -0.5],
            [0.7, 0.8, 0.5, 0.6],
            [0.2, 0.3, 0.1, 0.1],
        ]
    )

    assert np.allclose(
        pyefd.normalize_efd(coeffs.copy(), size_invariant=False),
        efa._rotate(coeffs),
    )


@pytest.fixture
def rotated_imgs() -> tuple[np.ndarray, np.ndarray]:
    """
    Two images which differ only by a rigid rotation
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

    return img1, img2


def test_coeffs_rotated_imgs(rotated_imgs: tuple[np.ndarray, np.ndarray]):
    """
    Check that we get the same EFA coefficients for two images that are the same but rotated
    """
    img1, img2 = rotated_imgs

    # Get EFA coeffs for both
    coeffs1 = efa.coefficients(img1, 50, 5)
    coeffs2 = efa.coefficients(img2, 50, 5)

    np.testing.assert_allclose(coeffs1, coeffs2, atol=1e-2)


def test_contour_handedness():
    """
    Check we get the same coefficients for two contours of different handedness
    """
    # Generate some points on a square
    x1 = [-1, 0, 1, 1, 1, 0, -1, -1]
    y1 = [1, 1, 1, 0, -1, -1, -1, 0]

    # Generate some new points that are the same but go the other way
    x2 = [-1, -1, -1, 0, 1, 1, 1, 0]
    y2 = [1, 0, -1, -1, -1, 0, 1, 1]

    order = 3
    coeffs1 = pyefd.elliptic_fourier_descriptors(
        np.array([x1, y1]).T, order=order, normalize=False
    )
    coeffs2 = pyefd.elliptic_fourier_descriptors(
        np.array([x2, y2]).T, order=order, normalize=False
    )

    # They will be different, since they have different handedness
    assert not np.allclose(coeffs1, coeffs2)

    # This should make them align
    np.testing.assert_allclose(efa._rotate(coeffs1), efa._rotate(coeffs2))


def test_shapes_facing_different_ways():
    """
    Check we get the same coefficients for two contours representing the same
    shape with the same handedness, but which point in different ways
    """
    # Triangle pointing up
    x1 = [1, 2, 0]
    y1 = [2, 1, 1]

    # Triangle pointing down
    x2 = [0, -1, 1]
    y2 = [-1, 0, 0]

    order = 3
    coeffs1 = pyefd.elliptic_fourier_descriptors(
        np.array([x1, y1]).T, order=order, normalize=False
    )
    coeffs2 = pyefd.elliptic_fourier_descriptors(
        np.array([x2, y2]).T, order=order, normalize=False
    )

    # These should just differ by a sign
    assert not np.allclose(coeffs1, coeffs2)
    np.testing.assert_allclose(coeffs1, -coeffs2)

    # This should make them align
    np.testing.assert_allclose(efa._rotate(coeffs1), efa._rotate(coeffs2), atol=1e-8)


def test_reorder_distances():
    """
    Check we correctly do this
    """
    points = np.array(
        [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
    )

    reference = (-0.5, 0)

    expected = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    )

    np.testing.assert_allclose(efa.reorder_by_distance(points, reference), expected)


def test_efa_circle():
    """
    Check we get the right coefficients if we perform EFA on a simple shape
    (a circle)
    """
    img = np.zeros((32, 32))
    r = 4
    x, y = 10, 10
    img[y - r : y + r + 1, x - r : x + r + 1] = disk(r)

    img = img.astype(np.uint8) * 255

    coeffs = efa.coefficients(img, 20, 4)

    expected_coeffs = [
        [r, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]

    np.testing.assert_allclose(coeffs, expected_coeffs, atol=0.01)
