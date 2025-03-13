"""
Image processing and manipulation

"""

import numpy as np
from scipy import ndimage


from scale_morphology.scales import errors


def has_holes(binary_img: np.ndarray) -> bool:
    """
    Check if a binary image has holes.

    :param binary_img: Binary image, with values 0 or 255 and dtype uint8
    :return: bool

    """
    errors.check_binary_img(binary_img)

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
    errors.check_binary_img(binary_img)

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
