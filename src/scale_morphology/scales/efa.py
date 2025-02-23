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
    a, b, c, d = coeffs[0]

    theta = 0.5 * np.arctan2(
        2 * (a * b + c * d),
        (a**2 - b**2 + c**2 - d**2),
    )

    # Step 1: align all harmonics with the major axis
    # Do this by rotating each harmonic by -i*theta,
    # where i is the harmonic number
    # We can do this efficiently by building Nx2x2 arrays
    # of coefficients and rotation matrices, then multiplying them
    coeff_matrices = coeffs.reshape(-1, 2, 2)
    indices = np.arange(1, coeffs.shape[0] + 1)
    cos = np.cos(indices * theta)
    sin = np.sin(indices * theta)
    rotation_matrices = np.stack(
        [
            np.stack([cos, -sin], axis=1),
            np.stack([sin, cos], axis=1),
        ],
        axis=1,
    )
    rotated = np.matmul(coeff_matrices, rotation_matrices).reshape(-1, 4)

    # Align the major axis with the x-axis
    psi = np.arctan2(rotated[0, 2], rotated[0, 0])
    matrix = np.array(
        [
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)],
        ]
    )
    for i in range(coeffs.shape[0]):
        rotated[i] = matrix.dot(
            np.array(
                [
                    [rotated[i, 0], rotated[i, 1]],
                    [rotated[i, 2], rotated[i, 3]],
                ]
            )
        ).flatten()

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
