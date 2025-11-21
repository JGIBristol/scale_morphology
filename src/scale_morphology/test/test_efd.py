"""
Tests for the efa module

"""

import pyefd
import pytest
import numpy as np
from skimage.morphology import disk
from scipy.ndimage import rotate

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
    n_pts = 10
    x, y = efa.points_around_edge(square, n_pts)

    assert len(x) == n_pts
    assert len(y) == n_pts

    expected_x = [5.5, 4.5, 3.5, 2.5, 1.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    expected_y = [3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0]

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

    atol, rtol = 0.02, 0.01
    np.testing.assert_allclose(coeffs1, coeffs2, atol=0.02, rtol=0.01)

    # Break the image and check the coefficients are now different
    s = 6
    x1, y1 = 11, 12
    img1[y1 - s : y1 + s + 1, x1 - s : x1 + s + 1] = 0

    coeffs1 = efa.coefficients(img1, 50, 5)

    np.testing.assert_raises(
        AssertionError,
        lambda x, y: np.testing.assert_allclose(x, y, atol=atol, rtol=rtol),
        coeffs1,
        coeffs2,
    )


def test_reorder_distances():
    """
    Check that we correctly reorder input points starting from the closest to the centroid.
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
    # Generate some fake points
    thetas = np.linspace(0, 2 * np.pi, 10000)
    x = [np.cos(t) for t in thetas]
    y = [np.sin(t) for t in thetas]

    order = 4
    size = 150  # Fake size of our object

    coeffs = efa._coeffs([x, y], (0, 0), order)
    coeffs = efa._normalise(coeffs, size)

    # 4 harmonics gives us 16 numbers
    # -3 for redundancy gives us 13
    # Then +1 for size gives us 14 total
    # The first harmonic is circular, so after scaling is
    # [1, 0, 0, 1] - but we removed the first three of these
    # So we are just left with [size, 1, 0, 0, 0, ...]
    # Actually its -1 though not sure why
    expected_coeffs = np.array([*[size, -1], *[0] * 12])

    np.testing.assert_allclose(coeffs, expected_coeffs, atol=0.01)


def test_normalise_coeffs():
    """
    Scale coefficients, prepend size and remove redundant information
    """
    coeffs = np.arange(24, 0, -1).reshape((6, 4))
    coeffs[0] = [24, 0, 0, coeffs[0, 3]]

    size = 100

    # Expected array is [100, (21, 20, 19...2, 1) / 24]
    expected = np.array([100, *np.arange(21, 0, -1) / 24])

    np.testing.assert_allclose(efa._normalise(coeffs, size), expected)


# ==== Tests for the EFA inductive biases - things like size, rotation,
# ==== handedness, location invariance


def test_shapes_facing_different_ways():
    """
    Check we get the same coefficients for two contours representing the same
    shape with the same handedness, but which point in different ways
    """
    # Triangle pointing up
    x1 = [1, 2, 0]
    y1 = [2, 1, 1]
    com1 = [1, 4 / 3]

    # Triangle pointing down
    x2 = [0, -1, 1]
    y2 = [-1, 0, 0]
    com2 = [0, -1 / 3]

    order = 3
    size = 9

    coeffs1 = efa._coeffs([x1, y1], com1, order=order)
    coeffs1 = efa._normalise(coeffs1, size)

    coeffs2 = efa._coeffs([x2, y2], com2, order=order)
    coeffs2 = efa._normalise(coeffs2, size)

    np.testing.assert_allclose(coeffs1, coeffs2, atol=0.01, rtol=0.01)


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


def _rectangle(w: int, h: int, angle: float = 45, centre: tuple[int, int] = (50, 50)):
    """
    A rectangle of the provided width/height
    """
    x, y = centre
    rect = np.zeros((100, 100), dtype=np.uint8)
    rect[x - w // 2 : x + w // 2, y - h // 2 : y + h // 2] = 255

    return rotate(rect, angle, order=0)


def test_size_invariant():
    """
    Check that the coefficients are size-invariant by finding the coeffs
    for two identical shapes with different scales

    Of course the 0th element will be different, because this just encodes
    the size...
    """
    big_rect = _rectangle(40, 20)
    small_rect = _rectangle(20, 10)

    n_points, n_coeffs = 30, 5
    big_coeffs = efa.coefficients(big_rect, n_points, n_coeffs)
    small_coeffs = efa.coefficients(small_rect, n_points, n_coeffs)

    atol, rtol = 0.02, 0.01
    np.testing.assert_allclose(big_coeffs[1:], small_coeffs[1:], atol=atol, rtol=rtol)

    # This shape should have different coefficients, however, because it
    # has a different aspecct ratio
    long_rect = _rectangle(10, 90)
    long_coeffs = efa.coefficients(long_rect, n_points, n_coeffs)

    np.testing.assert_raises(
        AssertionError,
        lambda x, y: np.testing.assert_allclose(x, y, atol=atol, rtol=rtol),
        long_coeffs,
        big_coeffs,
    )
    np.testing.assert_raises(
        AssertionError,
        lambda x, y: np.testing.assert_allclose(x, y, atol=atol, rtol=rtol),
        long_coeffs,
        small_coeffs,
    )


def test_rotation_invariant():
    """
    Check we get the same coefficients for the same shape but rotated
    """
    h, w = 40, 40
    orig = np.zeros((100, 100), dtype=np.uint8)
    orig[*np.triu_indices(100)] = 255

    rotated = np.rot90(orig, k=1)

    n_points, n_coeffs = 30, 5
    orig_coeffs = efa.coefficients(orig, n_points, n_coeffs)
    rot_coeffs = efa.coefficients(rotated, n_points, n_coeffs)

    atol, rtol = 0.02, 0.01
    # aliasing during the rotation might have changed the size a bit
    np.testing.assert_allclose(orig_coeffs[1:], rot_coeffs[1:], atol=atol, rtol=rtol)

    np.testing.assert_allclose(orig_coeffs[0], rot_coeffs[0], rtol=0.1)


def test_translation_invariant():
    """
    Check we get the same coefficients for the same shape but translated
    """
    h, w = 40, 50
    orig = _rectangle(h, w, angle=0)
    translated = _rectangle(h, w, centre=(30, 30), angle=0)

    n_points, n_coeffs = 30, 5
    orig_coeffs = efa.coefficients(orig, n_points, n_coeffs)
    translated_coeffs = efa.coefficients(translated, n_points, n_coeffs)

    np.testing.assert_allclose(orig_coeffs, translated_coeffs)


def test_reflection_invariant():
    """
    Check we get the same coefficients for the same shape but reflected
    """
    orig = _rectangle(40, 50)
    reflected = np.fliplr(orig)

    n_points, n_coeffs = 31, 5
    orig_coeffs = efa.coefficients(orig, n_points, n_coeffs)
    ref_coeffs = efa.coefficients(reflected, n_points, n_coeffs)

    atol, rtol = 0.02, 0.01
    np.testing.assert_allclose(orig_coeffs, ref_coeffs, atol=atol, rtol=rtol)
