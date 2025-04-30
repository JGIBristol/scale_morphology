"""
Stuff for dimensionality reduction

"""

import numpy as np
from sklearn.decomposition import PCA


# from umap import UMAP


def nan_scale_mask(coeffs: np.typing.NDArray) -> np.typing.NDArray:
    """
    Find where there are NaNs in the array of coefficients.

    Any scale with a NaN will be indicated here; returns an
    N-length 1d array of indices that can be used to remove any
    NaNs from the data.

    This function works for the (N, d, 4) shaped EFA coeffs
    and also the (N, d) shaped autoencoder and VAE coeffs

    :param coeffs: The array of coefficients, shaped (N, d) or (N, d, 4)
    :return: an N-length boolean mask indicating which scales have NaNs

    """
    if coeffs.ndim == 2:
        return np.isnan(coeffs).any(axis=1)
    if coeffs.ndim == 3 and coeffs.shape[-1] == 4:
        return np.isnan(coeffs).any(axis=(1, 2))
    raise ValueError("Coeffs should be 2D or 3D with last dimension 4")


def _flatten(coeffs: np.typing.NDArray) -> np.typing.NDArray:
    """
    Flatten the input data to remove the harmonic dimension
    """
    return coeffs.reshape((coeffs.shape[0], -1))


def pca(
    coeffs: np.typing.NDArray,
    *,
    flatten: bool = False,
    drop: np.ndarray | None = None,
    n_components: int = 2,
) -> tuple[np.typing.NDArray, PCA]:
    """
    Perform PCA on the input data

    :param coeffs: An array of coefficients representing the data.
                   May be shaped (N, d) if each image is represented by a single vector
                   or (N, d, 4), if each image is represented by a set of length-4 vectors,
                   as is the case for the EFA.
                   In the latter case the input should be flattened.
    :param flatten: Whether to flatten the output
    :param drop: 1d N-length boolean mask of scales to drop
    :param n_components: The number of components to keep

    :return: The transformed data, as a numpy array of shape (N, 2)
    :return: the PCA object
    """
    if drop is None:
        drop = np.zeros(coeffs.shape[0], dtype=bool)
    coeffs = coeffs[~drop]

    if flatten:
        coeffs = _flatten(coeffs)

    if coeffs.ndim != 2:
        raise ValueError("Coeffs should be 2D; 3D input should be flattened")

    pca = PCA(n_components=n_components)
    return np.ascontiguousarray(pca.fit_transform(coeffs)), pca


def umap(
    coeffs: np.typing.NDArray, *, flatten: bool = False
) -> tuple[np.typing.NDArray, ...]:  # TODO replace with the real type hint
    """
    Use UMAP for dimensionality reduction

    :param coeffs: An array of coefficients representing the data.
                   May be shaped (N, d) if each image is represented by a single vector
                   or (N, d, 4), if each image is represented by a set of length-4 vectors,
                   as is the case for the EFA.
                   In the latter case the input should be flattened.
    :param flatten: Whether to flatten the output
    :param drop: 1d N-length boolean mask of scales to drop

    :return: The transformed data, as a numpy array of shape (N, 2)
    :return: the UMAP object

    """
    raise NotImplementedError("TODO - add umap to the env")

    if drop is None:
        drop = np.zeros(coeffs.shape[0], dtype=bool)
    coeffs = coeffs[~drop]

    if flatten:
        coeffs = _flatten(coeffs)

    if coeffs.ndim != 2:
        raise ValueError("Coeffs should be 2D; 3D input should be flattened")

    if len(drop) != coeffs.shape[0]:
        raise ValueError("Drop should be 1D and N-length")

    umap_ = UMAP(n_components=2)
    return umap_.fit_transform(coeffs), umap_


def get_dim_reduction(dim_reduction_method: str) -> callable:
    """
    Get the dimensionality reduction method

    """
    match dim_reduction_method:
        case "pca":
            return pca
        case "umap":
            return umap
        case _:
            raise ValueError(
                "Unknown dimensionality reduction method:"
                f"{dim_reduction_method}, must be one of 'pca', 'umap'"
            )
