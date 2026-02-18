"""
Helpers for the EFA algorithm

"""

import pathlib
from concurrent.futures import ThreadPoolExecutor

import pyefd
import shapely
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import measure
from skimage.measure import euler_number
from scipy.ndimage import center_of_mass, binary_fill_holes


from scale_morphology.scales import errors
from scale_morphology.scales.segmentation import largest_connected_component


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

    # The final point is by default a repeat of the first, which breaks things
    # So cut it off here
    x = [p.x for p in pts][:-1]
    y = [p.y for p in pts][:-1]

    return np.array(x), np.array(y)


def _rotate(coeffs: np.ndarray) -> np.ndarray:
    """
    Rotate EFD coefficients for one contour such that the principal axis
    is horizontal.

    Also normalises handedness such that coefficients traced in opposite directions
    give the same coefficients.

    This is MUCH faster than the pyefd implementation as its fully vectorised.

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


def _coeffs(
    points: tuple[np.ndarray, np.ndarray], com: tuple[float, float], order: int
) -> np.ndarray:
    """
    Get the EFA coefficients from a set of points describing an outline.

    Re-orders them to have a consistent start point, find the coefficients
    and then rotates away and phase ambiguities.

    :param points: [x, y] points
    :param com: centre of mass of the input image
    """
    # Reorder the points to start in a consistent location
    # Starting at the closest point to the centroid
    points = reorder_by_distance(
        np.array(points).T,
        com[::-1],  # com is backwards (y, x)
    )

    coeffs = pyefd.elliptic_fourier_descriptors(points, order=order, normalize=False)
    return _rotate(coeffs)


def _normalise(coeffs: np.ndarray, size: float) -> np.ndarray:
    """
    Normalise and flatten coefficients to remove size information and remove
    redundant coefficients.

    Then prepend the size of the scale as the first element.
    """
    coeffs = coeffs / coeffs[0, 0]
    assert np.isclose(coeffs[0, 1], 0)
    assert np.isclose(coeffs[0, 1], 0)

    return np.concatenate(([size], coeffs.flatten()[3:]))


def coefficients(
    binary_img: np.ndarray, n_points: int, order: int, *, magnification: float = None
) -> np.ndarray:
    """
    Find the Elliptic Fourier expansion coefficients of an object in the given binary image.

    This image must be a uint8 numpy array containing a single object with no holes, where
    background pixels are marked with a 0 and foreground with 255.

    The returned EFA coefficients are un-normalised, to preserve size information, and are
    rotated so that the principal axis is horizontal.

    The contour begins at the point closest to the centroid of the object, which makes
    the coefficients consistent for shapes which differ by a rigid rotation.

    :param binary_img: 2D uint8 numpy array containing a single object and no holes;
                       background is 0; the object is 255
    :param n_points: number of points to linearly interpolate around the edge of the object
                     for our EFA calculation
    :param order: order of harmonics to use for EFA. Each harmonic has 4 degrees of freedom
                  (except the first, which has two; roughly)
    :param magnification: if specified, an additional scale factor to multiply the scale's size
                          by. Useful if some images were taken at a different resolution; e.g.
                          if most images were taken with a 4.0x magnification, but some with 3.2x,
                          then pass magnification=(4.0/3.2)

    :returns: Flattened EFA coefficients, normalised and with rotational ambiguity removed.
              The first element here is the size of the object (subject to any magnification).
              The next element is the d_1 coefficient (a_1, b_1 and c_1) are always (1, 0, 0)
              after our normalisation, so they are not included. The remaining elements are
              a_2, b_2, c_2, d_2, a_3, ...

    """
    if measure.euler_number(binary_img) != 1:
        raise errors.HolesError(
            "Image must contain a single object; got"
            f"{measure.euler_number(binary_img)}"
        )

    if magnification is None:
        magnification = 1

    x, y = points_around_edge(binary_img, n_points)

    coeffs = _coeffs([x, y], center_of_mass(binary_img), order)

    return _normalise(coeffs, (magnification**2) * (np.sum(binary_img) / 255))


def coeffs2points(coeffs, locus, *, n_pts=300):
    """
    Get the x, y coordinates of the EFD points given the coefficients
    in the expansion

    :param coeffs: 1d array of (size, d0, a1, b2, c2, d2, ...)

    """
    # Put the 2nd dimension back and remove the first element (it's the size)
    coeffs = coeffs[1:]
    coeffs = np.concatenate([[1, 0, 0], coeffs])
    coeffs = coeffs.reshape((-1, 4))

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


def _fix_segmentation(segmentation_path: pathlib.Path | str):
    """
    Read + fix segmentation
    """
    scale = tifffile.imread(segmentation_path)
    if euler_number(scale) != 1:
        # Fill holes
        scale = binary_fill_holes(scale)
        # Remove small objects
        scale = (largest_connected_component(scale) * 255).astype(np.uint8)

        # It's possible we might have removed everything, so just make sure we haven't here
        if euler_number(scale) != 1:
            raise errors.HolesError(f"Got {euler_number(scale)=}")

    return scale


# It would be nicer if this operated on a list of images, instead of image paths, but
# I think this is fine for now
def run_analysis(
    scale_paths: list[str | pathlib.Path],
    magnifications: np.ndarray,
    *,
    n_points: int,
    order: int,
    n_threads=8,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Find the EFA expansion from the images stored in the given paths.

    This will read each image, fill any holes and remove small objects, then find the
    Elliptic Fourier Descriptors from the above.
    The magnification also needs to be provided - some scales were taken at different
    magnifications, which needs to be accounted for by the EFA (one of the coefficients
    encodes the scale's size).

    :param scale_paths: an n-length list of paths containing the scales
    :param magnification: the magnification at which each scale image was taken.
    :param n_points: number of equally-spaced points to find on the outside of our object
                     to define the shape
    :param order: order of EFA analysis - intuitively, the number of ellipses. Using `order=k`
                  will result in a dimensionality of `4k-2`, since we have `4k` raw coefficients
                  but after rotation/normalisation we lose one from the ellipses' relative orientations
                  and one from their absolute orientation.
    :param n_threads: multithread the IO using this number of threads.
                      If reading from the RDSF (a network drive), you might have best results
                      by passing a large number e.g. 16 or 32
    :param show_progress: show a progress bar (intended for a Jupyter notebook)

    :return: an (N, k) shaped numpy array holding the coefficients for each scale.
    :raises ValueError: if the magnitudes
    :raises HolesError: if, even after correcting the image, we don't have a single object with no holes

    """
    # We probably actually passed a pandas series in
    magnifications = np.array(magnifications)
    if magnifications.ndim != 1 or len(magnifications) != len(scale_paths):
        raise ValueError(f"Got {magnifications.shape=} but {len(scale_paths)=}")

    pbar = tqdm if show_progress else lambda x, **kw: x

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        scales = list(
            pbar(
                executor.map(_fix_segmentation, scale_paths),
                total=len(scale_paths),
            )
        )

    # Rescale lengths in the images by the inverse of the magnification
    # I've forgotten why there's a 4 here over the magnification.
    # Probably because the default magnification was 4? Not sure that
    # it makes a difference here, it (should?) only affect the size parameter
    # which we should probably rescale to a senaible range anyway...
    coeffs = [
        coefficients(scale, n_points, order, magnification=4 / magnification)
        for scale, magnification in zip(pbar(scales), magnifications)
    ]

    return np.stack(coeffs)


def unscaled_efa_coeffs(
    img_path: pathlib.Path, *, n_points: int, order: int
) -> np.ndarray:
    """
    Get some un-normalised EFA coefficients.

    Normally, for feature extraction, we want to make sure that
    two images which are identical up to a rotation/scaling/etc.
    have the same EFA coefficients. We also want to remove things
    like the arbitrary starting point of the parameterisation and
    the relative rotation of each of the harmonics.

    This, however, means that we can't plot the EFA coefficients
    in a sensible way. This function returns the EFA coefficients
    without any normalisation/scaling that would otherwise break
    the plotting - it does mean that these coefficients are less
    good for things like feature selection, since they will contain
    information like arbitrary phase effects that come from how the
    contour was parameterised.

    Note that this gives us EFA coefficients in "pixel space" - to convert
    this to real, meaningful space we might have to do some manipulations with
    the magnification so that the size comes out correctly.

    This image must be a uint8 numpy array containing a single object with no holes, where
    background pixels are marked with a 0 and foreground with 255.

    The contour begins at the point closest to the centroid of the object, which makes
    the coefficients consistent for shapes which differ by a rigid rotation.

    :param img_path: path to a 2D image where background is 0; the object is 255
    :param n_points: number of points to linearly interpolate around the edge of the object
                     for our EFA calculation
    :param order: order of harmonics to use for EFA. Each harmonic has 4 degrees of freedom
                  (except the first, which has two; roughly)

    :returns: the EFA coefficients, as an Nx4 array.
    """
    # horrible messy "2 week before i get a new job" tier code
    img = _fix_segmentation(img_path)

    x, y = points_around_edge(img, n_points)
    points = [x, y]
    com = center_of_mass(img)
    points = reorder_by_distance(
        np.array(points).T,
        com[::-1],
    )
    coeffs = pyefd.elliptic_fourier_descriptors(points, order=order, normalize=False)

    return coeffs


def _approx_size(coeffs: np.ndarray, magnification: float) -> float:
    """
    Get the approximate size of an object described by its EFA coefficients.

    This returns just the area contained within the first ellipse - this is
    the area of the scale ignoring any higher order/smaller features, like bumps
    and protrusions. This might give us a better measure of overall bulk without
    looking at smaller features.
    """
    a1, b1, c1, d1 = coeffs[0, 0], coeffs[0, 1], coeffs[0, 2], coeffs[0, 3]
    size = np.pi * np.sqrt((a1**2 + b1**2) * (c1**2 + d1**2))

    return np.abs(size) / magnification**2


def _aspect_ratio(coeffs: np.ndarray) -> float:
    """
    Get the aspect ratio of the first harmonic.
    """
    a1, b1, c1, d1 = coeffs[0, 0], coeffs[0, 1], coeffs[0, 2], coeffs[0, 3]

    _, s, _ = np.linalg.svd([[a1, b1], [c1, d1]])
    major, minor = s[0], s[1]

    return major / minor


def _bumpiness(coeffs: np.ndarray, bump_threshold: int) -> float:
    """
    Get the "bumpiness metric" for a shape described by EFA coefficients:
    this is the fraction of Fourier power held in the coefficients
    above the bump_threshhold.
    """
    total_power = (coeffs**2).sum()  # Total Fourier Power
    high_power = (coeffs[bump_threshold:] ** 2).sum()

    return high_power / total_power


def shape_features(
    coeffs: np.ndarray, magnifications: np.ndarray, bump_threshold: int
) -> pd.DataFrame:
    """
    Get our shape features from the EFA coefficients.

    These are:
     - size: the size of the first harmonic ellipse
     - aspect ratio: the aspect ratio of the first harmonic ellipse
     - bumpiness: the amount of Fourier power held in the high-frequency components

    :param coeffs: Nx4 array of EFA coefficients, as returned by `unscaled_efa_coeffs`.
    :param magnification: magnification at which the image was taken
    :param bump_threshold: harmonic above which we consider their contribution "bumps".
                           e.g. bump_threshold = 5 will consider the first 5 harmonics to be
                           part of the overall shape, and anything higher a "bump". To set
                           an upper threshold, slice `coeffs`.

    :returns: a dataframe with "size", "aspect ratio" and "bumpiness" columns and N rows
    """
    sizes = [_approx_size(c, m) for c, m in zip(coeffs, magnifications)]
    aspect_ratios = [_aspect_ratio(c) for c in coeffs]
    bumpinesses = [np.log(_bumpiness(c, bump_threshold)) for c in coeffs]

    return pd.DataFrame(
        {"size": sizes, "aspect_ratio": aspect_ratios, "bumpiness": bumpinesses}
    )
