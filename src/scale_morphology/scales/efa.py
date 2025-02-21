"""
Helpers for the EFA algorithm

"""

import shapely
import numpy as np
from skimage import measure

import pyefd


def points_around_edge(
    binary_img: np.ndarray, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get equally spaced points along the boundary of a binary image.

    This performs a slight smoothing, as the contour points are on the edges of each pixel
    rather than on their corners.

    :param binary_img: Binary image.
    :param n_points: Number of points to generate.
    :return: x, y coordinates of the points.
    """
    assert np.isdtype(
        binary_img.dtype, np.uint8
    ), f"Input must be uint8: {binary_img.dtype=}"

    assert set(np.unique(binary_img)) <= {
        0,
        255,
    }, f"Input must be a binary image: {np.unique(binary_img)=}"

    contour_points = measure.find_contours(binary_img, fully_connected="high")[0]

    shape = shapely.LinearRing(contour_points)
    distances = np.linspace(0, 1, n_points + 1)
    pts = shapely.line_interpolate_point(shape, distances, normalized=True)

    x = [p.x for p in pts]
    y = [p.y for p in pts]

    return np.array(x), np.array(y)


def _rotate(coeffs: np.ndarray) -> np.ndarray:
    """
    Rotate EFD coefficients for one contour such that the principal axis
    is horizontal

    """
    assert coeffs.shape[1] == 4

    # Get first harmonic coefficients
    a, b, c, d = coeffs[0]

    # Calculate rotation angle
    numerator = 2 * (a * b + c * d)
    denominator = a * a - b * b + c * c - d * d
    theta = 0.5 * np.arctan2(numerator, denominator)

    # Create rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotated coefficients
    rotated = np.zeros_like(coeffs)
    for n in range(len(coeffs)):
        a, b, c, d = coeffs[n]
        rotated[n, 0] = a * cos_t + b * sin_t
        rotated[n, 1] = -a * sin_t + b * cos_t
        rotated[n, 2] = c * cos_t + d * sin_t
        rotated[n, 3] = -c * sin_t + d * cos_t

    return rotated


def coefficients(binary_img: np.ndarray, n_points: int, order: int) -> None:
    """
    Find the Elliptic Fourier expansion coefficients of an object in the given binary image.

    This image must be a uint8 numpy array containing a single object with no holes, where
    background pixels are marked with a 0 and foreground with 255.

    The returned EFA coefficients are un-normalised, to preserve size information, and are
    rotated so that the principal axis is horizontal.

    """
    assert measure.euler_number(binary_img) == 1, "Image must contain a single object"

    x, y = points_around_edge(binary_img, n_points)

    coeffs = pyefd.elliptic_fourier_descriptors(x, y, order=order, normalize=False)

    return _rotate(coeffs)


def coeffs2points(coeffs, locus, *, n_pts=300):
    """
    Get the x, y coordinates of the EFD points given the coefficients
    in the expansion

    """
    t = np.linspace(0, 1.0, n_pts)
    harmonics = np.arange(1, coeffs.shape[0] + 1)

    # Calculate all trig terms at once (num_harmonics x num_points)
    angles = 2 * np.pi * harmonics[:, None] * t[None, :]
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)

    # Calculate x and y coordinates using matrix multiplication
    xt = locus[0] + np.sum(
        coeffs[:, 0:1] * cos_terms + coeffs[:, 1:2] * sin_terms, axis=0
    )
    yt = locus[1] + np.sum(
        coeffs[:, 2:3] * cos_terms + coeffs[:, 3:4] * sin_terms, axis=0
    )

    return xt, yt
