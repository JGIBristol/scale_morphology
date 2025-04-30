"""
Interpret the meaning of the reduced-dimensionality dataset

"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.measure import label, regionprops
from matplotlib.colors import SymLogNorm

from scale_morphology.scales import read, dim_reduction

OUT_DIR = read.output_dir() / "interpretation"
if not OUT_DIR.exists():
    OUT_DIR.mkdir(parents=True)


def _size_plot(reduced_coeffs: np.ndarray, scale_images: list[np.ndarray]) -> None:
    """
    Plot the scale size against each dimension
    """
    # Find the size of each scale
    sizes = np.sum(scale_images, axis=(1, 2)) / 255

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for axis, dim in zip(axes, reduced_coeffs.T):
        axis.scatter(sizes, dim)
        axis.set_xlabel("Scale size /px")
        axis.set_ylabel("PC value")

    fig.supxlabel("Scale size /px")
    axes[0].set_title("PC1")
    axes[1].set_title("PC2")

    fig.tight_layout()

    fig.savefig(OUT_DIR / "size_plot.png")
    plt.close(fig)


def _coeff_plot(coeffs: np.ndarray) -> None:
    """
    Plot the size of the original coefficients
    """
    if coeffs.ndim != 3:
        raise NotImplementedError("this only works with EFA coeffs so far whoops")

    n, n_harmonics, n_coeffs = coeffs.shape

    fig, axis = plt.subplots()
    vmax = np.max(np.abs(coeffs))
    im = axis.imshow(
        coeffs.reshape(n, n_harmonics * n_coeffs), cmap="seismic", vmin=-vmax, vmax=vmax
    )

    fig.colorbar(im, label="Correlation coefficient")
    fig.tight_layout()

    fig.savefig(OUT_DIR / "coeffs.png")
    plt.close(fig)


def _correlation_plot(reduced_coeffs: np.ndarray, coeffs: np.ndarray) -> None:
    """
    Plot the correlation of each dimension with the original coefficients
    """
    if coeffs.ndim != 3:
        raise NotImplementedError("this only works with EFA coeffs so far whoops")

    n, n_harmonics, n_coeffs = coeffs.shape
    assert reduced_coeffs.shape[0] == n

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    cutoff = 0.0
    linthresh = 0.99

    for i, ax in enumerate(axes):
        corr = np.zeros((n_harmonics, n_coeffs))
        c = reduced_coeffs[:, i]

        for j in range(n_harmonics):
            for k in range(n_coeffs):
                corr[j, k] = np.corrcoef(c, coeffs[:, j, k])[0, 1]

        # Mask the values that are below the cutoff
        corr[np.abs(corr) < cutoff] = 0

        im = ax.imshow(
            corr,
            cmap="seismic",
            aspect="auto",
            norm=SymLogNorm(vmin=-1, vmax=1, linthresh=0.95),
        )
        ax.set_xticks(range(n_coeffs), "abcd")

        ax.set_title(f"PC{i + 1}")

    axes[0].set_ylabel("Harmonic Order")

    # fig.suptitle(f"Only showing correlations above {cutoff}")
    fig.suptitle(f"Colorbar linear below {linthresh}")
    # Tight layout before messing things around with the colorbar
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Correlation coefficient")

    fig.savefig(OUT_DIR / "correlation.png")
    plt.close(fig)


def get_aspect_ratio(binary_mask):
    """
    Calculate aspect ratio of a binary blob using regionprops

    """
    # Label the mask if it's not already labeled
    labeled_mask = label(binary_mask)

    # Get region properties
    props = regionprops(labeled_mask)

    # Get the largest region if there are multiple
    largest_region = max(props, key=lambda p: p.area)

    # Calculate aspect ratio
    aspect_ratio = largest_region.major_axis_length / largest_region.minor_axis_length

    return aspect_ratio


def _aspect_ratio_plot(
    reduced_coeffs: np.ndarray, scale_images: list[np.ndarray]
) -> None:
    """
    Plot the scale aspect ratio against each dimension
    """
    aspect_ratios = []
    for image in scale_images:
        # Convert grayscale to binary if not already
        binary_mask = image > 0
        aspect_ratio = get_aspect_ratio(binary_mask)
        aspect_ratios.append(aspect_ratio)

    aspect_ratios = np.array(aspect_ratios)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for axis, dim in zip(axes, reduced_coeffs.T):
        axis.scatter(aspect_ratios, dim)
        axis.set_xlabel("Scale aspect ratio")
        axis.set_ylabel("PC value")

    fig.supxlabel("Scale aspect ratio")
    axes[0].set_title("PC1")
    axes[1].set_title("PC2")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "aspect_ratio_plot.png")
    plt.close(fig)


def _plot_pca_importance(coeffs: np.ndarray, nan_rows: np.ndarray) -> None:
    """
    Plot the importance of each dimension
    """
    _, pca = dim_reduction.pca(coeffs, flatten=True, drop=nan_rows, n_components=15)

    variance = pca.explained_variance_ratio_ * 100
    cumulative = np.cumsum(variance)

    fig, axis = plt.subplots()
    ymax = 100
    axis.bar(range(15), variance, label="PC Variance Explained")

    # Plot top 2 in red
    axis.bar(range(2), variance[:2], color="red", label="Top 2 PCs")
    axis.set_ylim(0, ymax)

    axis.set_xlabel("Principal Component")
    axis.set_ylabel("Variance Explained /%")

    # Plot cumulative variance
    axis.plot(range(15), cumulative, color="black", linestyle="--", label="Cumulative")
    axis.set_ylabel("Cumulative Variance Explained /%")
    axis.set_ylim(0, ymax)
    axis.legend()

    fig.tight_layout()

    fig.savefig(OUT_DIR / "importance.png")
    plt.close(fig)


def main(*, compression_method: str, dim_reduction_method: str, progress: bool) -> None:
    """
    Read the coefficients and segmentations from disk
    and make some plots
    """
    # Get the coefficients that we have already created
    coeffs = read.read_coeffs(compression_method)

    # If any are nan we want to drop them from the plots
    nan_rows = dim_reduction.nan_scale_mask(coeffs)

    # Plot feature importance with lots of rows
    _plot_pca_importance(coeffs, nan_rows)

    # Perform the dimensionality reduction
    red_method = dim_reduction.get_dim_reduction(dim_reduction_method)
    reduced, _ = red_method(
        coeffs, flatten=(compression_method == "efa"), drop=nan_rows
    )

    # Get the images
    scale_images = [
        image for image in np.array(read.greyscale_images(progress=progress))[~nan_rows]
    ]

    _coeff_plot(coeffs[~nan_rows])
    _correlation_plot(reduced, coeffs[~nan_rows])
    _size_plot(reduced, scale_images)
    _aspect_ratio_plot(reduced, scale_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret the reduced-dimensionality dataset."
        "Coeffs need to be created before running this script"
    )
    parser.add_argument(
        "compression_method",
        choices={"efa", "autoencoder", "vae"},
        help="The method used to compress the images to vectors",
    )
    parser.add_argument(
        "dim_reduction_method",
        type=str,
        choices={"pca"},
        help="The method used to reduce the dimensionality of the vectors",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars",
    )

    main(**vars(parser.parse_args()))
