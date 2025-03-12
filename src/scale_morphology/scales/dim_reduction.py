"""
Stuff for dimensionality reduction

"""

import numpy as np


def flatten(coeffs: np.typing.NDArray) -> np.typing.NDArray:
    """
    Flatten the input data

    :param coeffs: An array of coefficients representing the data
    """


def pca(coeffs: np.typing.NDArray, *, flatten: bool = False) -> np.typing.NDArray:
    """
    Perform PCA on the input data

    :param scales: An array of coefficients representing the data.
                   May be shaped (N, d) if each image is represented by a single vector
                   or (N, d, 4), if each image is represented by a set of length-4 vectors,
                   as is the case for the EFA.
                   In the latter case the input should be flattened.
    :param flatten: Whether to flatten the output

    :return: The transformed data, as a numpy array of shape (N, 2)
    """
    # If flatten is True, flatten the data to the right shape
    # Check the shape of data - should be 2d
    # Perform PCA
