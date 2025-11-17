"""
Helpers for the EFA algorithm

"""

import pyefd
import shapely
import numpy as np
from skimage import measure
from scipy.ndimage import center_of_mass


from scale_morphology.scales import errors


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
    errors.check_binary_img(binary_img)

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
    is horizontal.

    Also normalises handedness such that coefficients traced in opposite directions
    give the same coefficients.

    """

    # Step 1: check handedness of the contour
    # We might have traced the contour anti-/clockwise, which
    # will result in us flipping the sign of our contour and also
    # Swapping a->c and b->d
    a, b, c, d = coeffs[0]
    if b * c < a * d:
        coeffs = np.column_stack(
            [-coeffs[:, 2], -coeffs[:, 3], -coeffs[:, 0], -coeffs[:, 1]]
        )

    a, b, c, d = coeffs[0]

    theta = 0.5 * np.arctan2(
        2 * (a * b + c * d),
        (a**2 - b**2 + c**2 - d**2),
    )

    # Step 2: align all harmonics with the major axis
    # Do this by rotating each harmonic by -i*theta,
    # where i is the harmonic number
    # We can do this efficiently by building Nx2x2 arrays
    # of coefficients and rotation matrices, then multiplying them
    coeff_matrices = coeffs.reshape(-1, 2, 2)
    indices = np.arange(1, coeffs.shape[0] + 1)
    cos = np.cos(indices * theta)
    sin = np.sin(indices * theta)
    theta_rotations = np.stack(
        [
            np.stack([cos, -sin], axis=1),
            np.stack([sin, cos], axis=1),
        ],
        axis=1,
    )
    # This is now an Nx2x2 array of rotated coefficients
    rotated = np.matmul(coeff_matrices, theta_rotations)

    # Step 3: align the major axis with the x-axis
    # We just repeat the above, but now it's easier because the
    # rotation matrix is the same for all harmonics
    # psi is the angle defined by the c and a coeffs
    psi = np.arctan2(rotated[0, 1, 0], rotated[0, 0, 0])
    psi_rotation = np.array(
        [
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)],
        ]
    )
    rotated = np.matmul(psi_rotation, rotated.reshape(-1, 2, 2))

    # Put the coefficients back into the original shape
    return rotated.reshape(-1, 4)


def reorder_by_distance(
    points: np.ndarray, reference_point: tuple[float, float]
) -> np.ndarray:
    """
    Reorder a 2d array of points, preserving their order but shuffling such that the point
    closest to `reference_point` is first in the array

    :param points: np array of points, shape (N, 2)
    :param reference_point: point to use as reference
    """
    assert points.shape[1] == 2
    assert len(reference_point) == 2

    x, y = points.T
    distances_sq = (x - reference_point[0]) ** 2 + (y - reference_point[1]) ** 2
    start_idx = np.argmin(distances_sq)

    return np.roll(points, -start_idx, axis=0)


def coefficients(binary_img: np.ndarray, n_points: int, order: int) -> None:
    """
    Find the Elliptic Fourier expansion coefficients of an object in the given binary image.

    This image must be a uint8 numpy array containing a single object with no holes, where
    background pixels are marked with a 0 and foreground with 255.

    The returned EFA coefficients are un-normalised, to preserve size information, and are
    rotated so that the principal axis is horizontal.

    The contour begins at the point closest to the centroid of the object, which makes
    the coefficients consistent for shapes which differ by a rigid rotation.

    """
    if measure.euler_number(binary_img) != 1:
        raise errors.HolesError(
            "Image must contain a single object; got"
            f"{measure.euler_number(binary_img)}"
        )

    x, y = points_around_edge(binary_img, n_points)

    # Reorder the points to start in a consistent location
    # Starting at the closest point to the centroid
    centroid = center_of_mass(binary_img)
    x, y = reorder_by_distance(
        np.array([x, y]).T,
        centroid[::-1],
    ).T

    coeffs = pyefd.elliptic_fourier_descriptors([x, y], order=order, normalize=False)
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
        coeffs[:, 2:3] * cos_terms + coeffs[:, 3:4] * sin_terms, axis=0
    )
    yt = locus[1] - np.sum(
        coeffs[:, 0:1] * cos_terms + coeffs[:, 1:2] * sin_terms, axis=0
    )

    return xt, yt
