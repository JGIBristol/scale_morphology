"""
random other utilities
"""

import numpy as np


def drop_nan_coeffs(coeffs: np.typing.NDArray) -> np.typing.NDArray:
    """
    Drop any rows containing NaNs from the coefficients

    This function works for the (N, d, 4) shaped EFA coeffs
    and also the (N, d) shaped autoencoder and VAE coeffs

    :param coeffs: The coefficients to clean
    :return: The cleaned coefficients, with

    """
    if coeffs.ndim == 2:
        return coeffs[~np.isnan(coeffs).any(axis=1)]
    if coeffs.ndim == 3 and coeffs.shape[-1] == 4:
        return coeffs[~np.isnan(coeffs).any(axis=(1, 2))]
    raise ValueError("Coeffs should be 2D or 3D with last dimension 4")
