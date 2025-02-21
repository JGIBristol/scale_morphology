"""
Image processing

"""

import numpy as np

import shapely
from scipy import ndimage
from skimage.measure import find_contours


class BadImgError(Exception): ...


def has_holes(binary_img: np.ndarray) -> bool:
    """
    Check if a binary image has holes.

    :param binary_img: Binary image, with values 0 or 255 and dtype uint8
    :return: bool

    """
    assert np.isdtype(
        binary_img.dtype, np.uint8
    ), f"Input must be uint8: {binary_img.dtype=}"

    assert set(np.unique(binary_img)) <= {
        0,
        255,
    }, f"Input must be a binary image: {np.unique(binary_img)=}"

    # Find the number of disconnected background regions
    # If there's more than one, the image has a hole in it
    _, num_features = ndimage.label(~binary_img)

    return num_features > 1


def fill_background(binary_img: np.typing.NDArray) -> np.typing.NDArray:
    """
    Fill in the background of a binary image containing multiple disconnected background
    regions. All but the largest region will be filled in.

    :param binary_img: uint8 binary image, with values 0 or 255
    :return: the image with all regions but the largest filled in

    """
    assert np.isdtype(
        binary_img.dtype, np.uint8
    ), f"Input must be uint8: {binary_img.dtype=}"

    assert set(np.unique(binary_img)) <= {
        0,
        255,
    }, f"Binary image must be 0 or 255: {np.unique(binary_img)}"

    # Label distinct regions of background
    background = ~binary_img
    labels, num_features = ndimage.label(background)

    # More than 1 region of bkg means we have holes
    if num_features > 1:
        copy = binary_img.copy()

        # Create a list of bkg labels in descending order of size
        counts = {
            k: v for k, v in zip(*np.unique(labels, return_counts=True)) if k != 0
        }

        # Remove the pair from this dict with the highest value
        counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        counts = {k: v for k, v in counts.items() if k != list(counts.keys())[0]}

        # Remove the holes
        for k in counts.keys():
            copy[labels == k] = 255

    return copy


def even_edge_points(
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
    contour_points = find_contours(binary_img, fully_connected="high")[0]

    shape = shapely.LinearRing(contour_points)
    distances = np.linspace(0, 1, n_points + 1)
    pts = shapely.line_interpolate_point(shape, distances, normalized=True)

    x = [p.x for p in pts]
    y = [p.y for p in pts]

    return np.array(x), np.array(y)
