"""
Segment the scale from an unprocessed image
"""

import numpy as np

from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import (
    binary_opening,
    disk,
    erosion,
    reconstruction,
    binary_closing,
)


def _open_by_reconstruction(mask, radius):
    selem = disk(radius)
    seed = erosion(mask, selem)
    opened = reconstruction(
        seed.astype(np.uint8), mask.astype(np.uint8), method="dilation"
    )
    return opened.astype(bool)


def _largest_connected_component(binary_array):
    """
    Return the largest connected component of a binary array, as a binary array

    :param binary_array: Binary array.
    :returns: Largest connected component.

    """
    labelled, _ = ndimage.label(binary_array, np.ones((3, 3)))

    # Find the size of each component
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0

    retval = labelled == np.argmax(sizes)
    return retval


def classical_segmentation(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempt to segment the scale out using a classical computer vision pipeline-
    Otsu thresholding, removing detached objects and binary
    opening.

    :param img: 3-channel image to threshold
    :returns:
    :returns: segmentation mask
    """
    img = rgb2gray(img)
    threshold = threshold_otsu(img)

    mask = img < threshold

    mask = _open_by_reconstruction(mask, 20)
    mask = _largest_connected_component(mask)

    # Closing then opening to get rid of weird bits
    elem = np.ones((25, 25))
    mask = binary_closing(binary_opening(mask, footprint=elem), footprint=elem)

    return ndimage.binary_fill_holes(mask, structure=np.ones((3, 3)))
