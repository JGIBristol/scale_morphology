"""
Utilities for data viz
"""

import pandas as pd
import seaborn as sns
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import gaussian_kde

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
            raise ValueError("KDE needs at least 2 points")

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
    normalise: bool = False,
):
    """
    Plot a pair plot: a scatter plot of each PC (or LDA axis) against all the others,
    with 1d histograms on the diagonal.

    :param reduced_coeffs: the coefficients after PCA or LDA dimensionality reduction
    :param grouping_df: dataframe holding columns that tells us which variables to group by.
                        If grouping by a single variable, this will be a dataframe holding a single
                        column.
    :param colours: the colour to use for each group.
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
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(20, 20))

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
