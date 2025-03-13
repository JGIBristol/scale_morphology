"""
Error types and checking
"""

import numpy as np


class BadImgError(Exception):
    """
    Base class for custom exceptions to do with image validity
    """


class ImgTypeError(BadImgError):
    """
    Image has the wrong type
    """


class BinaryValError(BadImgError):
    """
    Image has the wrong values
    """


class HolesError(BadImgError):
    """
    Image contains holes
    """


def check_binary_img(binary_img: np.ndarray) -> None:
    """
    Check that we have a greyscale image with the right properties

    :param binary_img: uint8 binary image, with values 0 or 255

    :raises ImgTypeError: if the image is not a unint8
    :raises BinaryValError: if the image is not binary with the correct values

    """
    if not np.isdtype(binary_img.dtype, np.uint8):
        raise ImgTypeError(f"Input must be uint8: {binary_img.dtype=}")

    if not set(np.unique(binary_img)) <= {
        0,
        255,
    }:
        raise BinaryValError(
            f"Input must be a binary image: {np.unique(binary_img)=} with vals {{0, 255}}"
        )
