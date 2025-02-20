"""
Image processing

"""

import numpy as np

from scipy import ndimage


def has_holes(binary_img: np.ndarray) -> bool:
    """
    Check if a binary image has holes.

    """
    assert set(np.unique(binary_img)) <= {
        0,
        1,
    }, f"Input must be a binary image: {np.unique(binary_img)=}"

    # Find the number of disconnected background regions
    # If there's more than one, the image has a hole in it
    _, num_features = ndimage.label(~binary_img.astype(bool))

    return num_features > 1
