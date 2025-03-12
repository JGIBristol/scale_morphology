"""
Stuff for dimensionality reduction

"""

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP


def flatten(coeffs: np.typing.NDArray) -> np.typing.NDArray:
    """
    Flatten the input data to remove the harmonic dimension
    """
    return coeffs.reshape((coeffs.shape[0], -1))


def pca(coeffs: np.typing.NDArray, *, flatten: bool = False) -> np.typing.NDArray:
    """
    Perform PCA on the input data

    :param coeffs: An array of coefficients representing the data.
                   May be shaped (N, d) if each image is represented by a single vector
                   or (N, d, 4), if each image is represented by a set of length-4 vectors,
                   as is the case for the EFA.
                   In the latter case the input should be flattened.
    :param flatten: Whether to flatten the output

    :return: The transformed data, as a numpy array of shape (N, 2)
    """
    if flatten:
        coeffs = flatten(coeffs)

    if coeffs.ndim != 2:
        raise ValueError("Coeffs should be 2D; 3D input should be flattened")

    return np.ascontiguousarray(PCA(n_components=2).fit_transform(coeffs))


def umap(coeffs: np.typing.NDArray, *, flatten: bool = False) -> np.typing.NDArray:
    """
    Use UMAP for dimensionality reduction

    :param coeffs: An array of coefficients representing the data.
                   May be shaped (N, d) if each image is represented by a single vector
                   or (N, d, 4), if each image is represented by a set of length-4 vectors,
                   as is the case for the EFA.
                   In the latter case the input should be flattened.
    :param flatten: Whether to flatten the output

    :return: The transformed data, as a numpy array of shape (N, 2)

    """
    if flatten:
        coeffs = flatten(coeffs)

    if coeffs.ndim != 2:
        raise ValueError("Coeffs should be 2D; 3D input should be flattened")

    return UMAP(n_components=2).fit_transform(coeffs)
