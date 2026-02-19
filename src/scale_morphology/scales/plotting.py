"""
Utilities for data viz
"""

import pathlib

import tifffile
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from scipy.stats import gaussian_kde
from sklearn.base import BaseEstimator

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from . import efa


def plot_efa(
    centroid: tuple[float, float], coeffs: np.ndarray, *, axis: plt.Axes, **plot_kw
) -> None:
    """
    Plot the EFA of a scale

    :param centroid: centre of the scale; can be calculataed from a mask using
                     `[np.average(x) for x in np.where(mask > 0)]`
    :param coeffs: EFD coefficients [[a, b, c, d]_1, [a, b, c, d]_2, ...]
    :param axis: axis to plot on
    :param plot_kw: further keywords to be passed to the plot fcn (e.g. color, linestyle)

    """
    x, y = efa.coeffs2points(coeffs, centroid)
    axis.plot(x, y, **plot_kw)


def plot_unscaled_efa(
    centroid: tuple[float, float], coeffs: np.ndarray, *, axis: plt.Axes, **plot_kw
) -> None:
    """
    plot EFA coeffs that we got with efa.unscaled_efa_coeffs
    """
    # Parameterise
    n_pts = 300
    t = np.linspace(0, 1.0, n_pts)
    harmonics = np.arange(1, coeffs.shape[0] + 1)

    # Calculate all trig terms at once (num_harmonics x num_points)
    angles = 2 * np.pi * harmonics[:, None] * t[None, :]
    cos_terms = np.cos(angles)
    sin_terms = np.sin(angles)

    # Calculate x and y coordinates using matrix multiplication
    xt = centroid[0] + np.sum(
        coeffs[:, 2:3] * cos_terms + coeffs[:, 3:4] * sin_terms, axis=0
    )
    yt = centroid[1] - np.sum(
        coeffs[:, 0:1] * cos_terms + coeffs[:, 1:2] * sin_terms, axis=0
    )

    axis.plot(xt, yt, **plot_kw)


def clear2colour_cmap(colour) -> colors.Colormap:
    """
    Colormap that varies from clear to a colour - useful for plotting the KDEs
    """
    c_white = colors.colorConverter.to_rgba("white", alpha=0)
    c_black = colors.colorConverter.to_rgba(colour, alpha=0.5)
    return colors.ListedColormap([c_white, c_black], f"clear2{colour}")


def kdeplot(
    axis: plt.Axes,
    data: np.ndarray,
    labels: pd.Series,
    colour_lookup: dict[np.float64, str],
    normalise: bool,
):
    """
    Plot a smoothed 1d histogram of the data on an axis, splitting according to the labels
    """
    df = pd.DataFrame({"value": data, "label": labels})
    df = df.replace({"label": colour_lookup})
    palette = {c: c for c in df["label"].unique()}
    sns.kdeplot(
        data=df,
        x="value",
        hue="label",
        legend=False,
        fill=False,
        palette=palette,
        ax=axis,
        common_norm=not normalise,
    )
    axis.set_xlabel("")
    axis.set_ylabel("")


def _plot_kde_scatter(
    axis: plt.Axes,
    x_coeffs: np.ndarray,
    y_coeffs: np.ndarray,
    labels: pd.Series,
    colour_lookup: dict[np.float64, str],
):
    """
    Plot a scatter plot and colour-coded 2D KDE

    """
    unique_labels = np.unique(labels)
    assert len(unique_labels) == len(
        colour_lookup
    ), f"{len(unique_labels)=}, {len(colour_lookup)=}"

    # Grid for this pair
    xmin, xmax = x_coeffs.min(), x_coeffs.max()
    ymin, ymax = y_coeffs.min(), y_coeffs.max()
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            raise ValueError(
                f"KDE needs at least 2 points, got {mask.sum()} for {label}"
            )

        data = np.vstack([x_coeffs[mask], y_coeffs[mask]])
        kde = gaussian_kde(data)
        f = np.reshape(kde(positions).T, xx.shape)

        colour = colour_lookup[label]
        axis.contourf(xx, yy, f, levels=15, cmap=clear2colour_cmap(colour))
        n = int(np.sum(mask))

        axis.scatter(*data, color=colour, s=5, marker="s", linewidth=0.5, edgecolor="k")


def pair_plot(
    reduced_coeffs: np.ndarray,
    grouping_df: pd.DataFrame,
    colours: list[str],
    *,
    axis_label: str,
    normalise: bool = False,
) -> plt.Figure:
    """
    Plot a pair plot: a scatter plot of each PC (or LDA axis) against all the others,
    with 1d histograms on the diagonal.

    :param reduced_coeffs: the coefficients after PCA or LDA dimensionality reduction
    :param grouping_df: dataframe holding columns that tells us which variables to group by.
                        If grouping by a single variable, this will be a dataframe holding a single
                        column.
    :param colours: the colour to use for each group.
    :param axis_label: what to label the axes; probably either "PC" or "LDA vector"
    :param normalise: whether to normalise the 1d histograms

    """
    n_row, n_col = grouping_df.shape
    _, n_dim = reduced_coeffs.shape
    assert _ == n_row, f"Got {_} coeffs but {n_col} entries in dataframe"

    # Get the unique labels for each group
    labels, uniques = pd.factorize(
        grouping_df.apply(lambda row: tuple(row.values), axis=1)
    )
    assert len(uniques) == len(
        colours
    ), f"Got {len(uniques)} groups but {len(colours)} colours"
    colour_lookup = dict(zip(np.unique(labels), colours))

    # Init a figure
    # Make it bigger if it would be small
    figsize = (3 * n_dim, 3 * n_dim) if n_dim > 3 else (12, 12)
    fig, axes = plt.subplots(n_dim, n_dim, figsize=figsize, constrained_layout=True)
    if n_dim == 1:
        axes = np.array([[axes]])

    with tqdm(total=n_dim**2) as pbar:
        pbar.set_description("Plotting pair plot")
        for i in range(n_dim):
            for j in range(n_dim):
                axis = axes[j, i]
                x = reduced_coeffs[:, i]
                y = reduced_coeffs[:, j]

                # Plot histograms on the diagonal
                if i == j:
                    kdeplot(axis, x, labels, colour_lookup, normalise)
                else:
                    _plot_kde_scatter(axis, x, y, labels, colour_lookup)

                axis.set_xticks([])
                axis.set_yticks([])

                pbar.update(1)

    # Set labels along bottom/left edges
    for i, axis in enumerate(axes[-1]):
        axis.set_xlabel(f"{axis_label} {i+1}")
    for i, axis in enumerate(axes[:, 0]):
        axis.set_ylabel(f"{axis_label} {i+1}")

    # Add a legend
    fig.legend(
        handles=sorted(
            [
                Patch(
                    facecolor=colour_lookup[idx],
                    edgecolor="k",
                    linewidth=0.5,
                    label=" | ".join([str(u) for u in unique]),
                )
                for idx, unique in zip(np.unique(labels), uniques, strict=True)
            ],
            key=lambda p: p.get_label(),
        ),
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        title="Group",
    )
    return fig


def _1d_shape_plot(
    axis: plt.Axes, coeffs: np.ndarray, scale_paths: list[str | pathlib.Path]
):
    """
    Plot the shapes of scales as a coefficient varies along an axis
    """
    assert coeffs.ndim == 1
    assert len(coeffs) == len(scale_paths)

    axis.hist(coeffs, bins=100)

    # Get the scale paths corresponding to these quantiles in our coeffs
    quantiles = [0.01, 0.20, 0.5, 0.80, 0.99]

    quantile_values = np.quantile(coeffs, quantiles)
    quantile_idx = [np.abs(coeffs - q).argmin() for q in quantile_values]

    quantile_locs = coeffs[quantile_idx]
    quantile_imgs = [tifffile.imread(f)[::20, ::20] for f in scale_paths[quantile_idx]]

    # Plot them along the axis
    for loc, img in zip(quantile_locs, quantile_imgs, strict=True):
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.3, cmap=clear2colour_cmap("k")),
            (loc, axis.get_ylim()[1] / 2),
            pad=0.05,
            frameon=True,
            bboxprops={"edgecolor": "k", "linewidth": 1},
        )
        axis.add_artist(ab)


def _2d_shape_plot(
    axis: plt.Axes, x: np.ndarray, y: np.ndarray, scale_paths: list[str | pathlib.Path]
):
    """
    Plot the shapes of scales as vary in 2d
    """
    assert x.ndim == 1
    assert y.ndim == 1
    assert len(x) == len(y)
    assert len(x) == len(scale_paths)

    # directions - we'll dot with these to get the extrema in 2d
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    dirns = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    co_ords = np.stack([x, y])
    axis.scatter(*co_ords, s=2, color="k", alpha=0.4)

    # Find which co-ords are closest to the directions we chose
    scores = np.linalg.matmul(co_ords.T, dirns.T)
    extrema_idx = scores.argmax(axis=0)

    extrema_locs = co_ords.T[extrema_idx]
    extrema_imgs = [tifffile.imread(f)[::20, ::20] for f in scale_paths[extrema_idx]]
    for loc, img in zip(extrema_locs, extrema_imgs, strict=True):
        ab = AnnotationBbox(
            OffsetImage(img, zoom=0.3, cmap=clear2colour_cmap("k")),
            loc,
            pad=0.05,
            frameon=True,
            bboxprops={"edgecolor": "k", "linewidth": 1},
        )
        axis.add_artist(ab)


def shape_plot(
    reduced_coeffs: np.ndarray,
    *,
    scale_paths: list[str | pathlib.Path],
    axis_label: str,
) -> None:
    """
    Plot how the shape of the scales changes along each axes; in 1d along the diagonal,
    and along two axes otherwise.

    :param reduced_coeffs: the coefficients after PCA or LDA dimensionality reduction
    :param scale_paths: paths to the scale images - these will be read in and shown on the plot
    :param axis_label: what to label the axes; probably either "PC" or "LDA vector"

    """
    n_scales, n_dim = reduced_coeffs.shape
    assert n_scales == len(
        scale_paths
    ), f"Got {n_scales} coeffs but {len(scale_paths)} filepaths"

    # Init a figure
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(4 * n_dim, 4 * n_dim))

    with tqdm(total=n_dim**2) as pbar:
        pbar.set_description("Plotting pair plot")
        for i in range(n_dim):
            for j in range(n_dim):
                axis = axes[j, i]
                x = reduced_coeffs[:, i]
                y = reduced_coeffs[:, j]

                # Plot a 1d line of scale images if we're on the diagonal
                if i == j:
                    _1d_shape_plot(axis, x, scale_paths)

                # Plot the extrema along our two dimensions otherwise
                else:
                    _2d_shape_plot(axis, x, y, scale_paths)

                axis.set_xticks([])
                axis.set_yticks([])

                pbar.update(1)

    # Set labels along bottom/left edges
    for i, axis in enumerate(axes[-1]):
        axis.set_xlabel(f"{axis_label} {i+1}")
    for i, axis in enumerate(axes[:, 0]):
        axis.set_ylabel(f"{axis_label} {i+1}")


def heatmap(scalings):
    """
    Plot a heatmap showing component loadings in terms of features
    """
    fig, axis = plt.subplots(figsize=(10, 5))

    im = axis.matshow(
        scalings.T,
        aspect="auto",
        cmap="seismic",
        norm=colors.CenteredNorm(vcenter=0.0),
    )

    n_components = scalings.shape[0]
    axis.set_xticks(range(n_components), range(1, n_components + 1))
    axis.set_title("Feature contribution to each component")
    axis.set_ylabel("Component")
    axis.set_xlabel("Feature")
    fig.colorbar(im, ax=axis)

    return fig


def feature_importance(estimator: BaseEstimator):
    """
    Plot feature importance of PCA or LDA
    """
    fig, axis = plt.subplots(figsize=(8, 4))

    percent_importances = estimator.explained_variance_ratio_
    axis.bar(np.arange(len(percent_importances)), percent_importances)
    axis.set_yscale("log")

    return fig


def plot_metadata_bars(mdata: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """
    Plot bar charts showing counts of unique values in metadata.
    """
    columns = ["age", "sex", "magnification", "growth", "mutation", "stain"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        counts = mdata[col].value_counts().sort_index()
        axes[i].bar(counts.index.astype(str), counts.values)
        axes[i].set_title(col.capitalize())
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "metadata_bar_charts.png")


def pca_barplot(scalings):
    """
    Make a bar plot showing the contribution of the features to each PC
    """
    # Want to plot the first two features (size, d_1) and then a whole number
    # of ellipses, each of which are made up of 4 components
    n_ellipses = min(3, 1 + (scalings.shape[0] - 2) // 4)
    n_bars = 2 + n_ellipses * 4

    n_pcs = scalings.shape[0]

    # 4 columns, with as many rows as we need
    n_col = 4
    n_row = int(np.ceil(n_pcs / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 4, n_row * 4))

    # First two ticks for size, d_1; then calculate the rest + build their labels
    xticks = [-3, -1]
    xticklabels = ["size", r"$d_1$"]
    vlines = [-2, 0]
    for i in range(n_ellipses):
        xticklabels += [rf"$a_{i+2}$", rf"$b_{i+2}$", rf"$c_{i+2}$", rf"$d_{i+2}$"]

        start, end = 1 + 5 * i, 5 + 5 * i
        xticks += list(range(start, end))

        vlines.append(start - 1)

    for i, (axis, scaling) in enumerate(zip(axes.flat, scalings)):
        axis.bar(xticks, scaling[:n_bars])

        axis.set_xticks(
            xticks,
            xticklabels,
            ha="right",
        )

        # Every 4 features is one ellipse - separate them visually
        for v in vlines:
            axis.axvline(v, color="k", linestyle="--")

        axis.set_ylim(min(-1.1, axis.get_ylim()[0]), max(1.1, axis.get_ylim()[1]))
        axis.set_yticks([-1, 0, 1])

        axis.set_title(f"PC{i+1}")

    return fig
