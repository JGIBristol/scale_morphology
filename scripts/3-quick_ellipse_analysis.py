"""
Measure the size and aspect ratio of scales directly.

To get a basic idea of how these features split our dataset,
we can simply fit ellipses to each of the segmentation masks
and then make some basic plots and output some simple statistics.

"""

import pathlib
import warnings
import argparse
from contextlib import redirect_stdout

import cv2
import tifffile
import numpy as np
from numpy import (
    inf,
)  # Need this to parse queries including np.inf; query CLI won't accept np.inf with the dot
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse
from matplotlib.colors import TABLEAU_COLORS, CSS4_COLORS
from sklearn.feature_selection import f_classif
from statsmodels.multivariate.manova import MANOVA
from pingouin import pairwise_gameshowell

from scale_morphology.scales import metadata, ellipse_fit, plotting, efa


def _efa_coeffs(
    segmentation_paths: pd.Series,
    efa_cache_path: pathlib.Path | None,
    n_edge_points: int,
    n_harmonics: int,
) -> np.ndarray:
    """
    Get the EFA coefficients - either from a cache, or by reading the images from
    paths and running the analysis on them
    """
    if efa_cache_path is not None and efa_cache_path.is_file():
        return np.load(efa_cache_path)

    coeffs = []
    for path in tqdm(segmentation_paths):
        coeffs.append(
            efa.unscaled_efa_coeffs(path, n_points=n_edge_points, order=n_harmonics)
        )
    coeffs = np.stack(coeffs)

    if efa_cache_path is not None:
        np.save(efa_cache_path, coeffs)

    return coeffs


def _get_ellipse(
    img: np.ndarray, magnification: float, axis: plt.Axes | None
) -> tuple[float, float]:
    """
    Fit an ellipse to an image.

    Fits an ellipse to a binary image where 255 are object
    and 0 the background, returns the size of the ellipse and its
    aspect ratio.

    If a path is provided, saves the image with the ellipse drawn on.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit ellipse: returns ((center_x, center_y), (width, height), angle)
    ellipse = cv2.fitEllipse(largest_contour)
    (center_x, center_y), (minor_axis, major_axis), angle = ellipse

    # Calculate size (area of ellipse) and aspect ratio
    size = np.pi * (major_axis / 2) * (minor_axis / 2)
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    # Scale the size by the magnification squared
    size /= magnification**2

    # Save plot if path provided
    if axis is not None:
        (cx, cy), (MA, ma), angle = ellipse
        axis.imshow(img, cmap="binary")

        axis.add_patch(
            Ellipse(
                xy=(cx, cy),
                width=MA,
                height=ma,
                angle=angle,
                edgecolor="lime",
                facecolor="none",
                linewidth=2,
            )
        )

    return size, aspect_ratio


def _statstests(mdata: pd.DataFrame, labels: pd.Series, output_dir: pathlib.Path):
    """
    Run stats tests to see how separable our columns are
    """
    with open(output_dir / "stats.txt", "w") as f:
        with redirect_stdout(f):
            cols = ["size", "aspect_ratio", "bumpiness"]

            # F-test
            f_stat, p_vals = f_classif(mdata[cols], labels)
            print(
                "F-test: can we separate out at least one of our classes from the others?"
            )
            print("=" * 8)
            for col, f, p in zip(cols, f_stat, p_vals):
                print(f"{col}: f={float(f):.6f}, p={float(p):.6f}", end="", flush=False)
                if p < 0.05:
                    print("*", end="")
                if p < 0.01:
                    print("*", end="")
                if p < 0.001:
                    print("*", end="")
                print()

            # Pairwise comparison
            print()
            print(
                "Games-Howell pairwise test: can we separate our classes out from each other individually?"
            )
            print("=" * 8)
            df = pd.DataFrame(
                {
                    "size": mdata["size"],
                    "aspect_ratio": mdata["aspect_ratio"],
                    "bumpiness": mdata["bumpiness"],
                    "label": labels,
                }
            )
            for col in cols:
                print(f"{col}:")
                print("-" * 4)
                result = pairwise_gameshowell(data=df, dv=col, between="label")
                print(result)

            # MANOVA - joint comparison
            print()
            print(
                "MANOVA test: do class means differ in the joint (size, aspect ratio) space?"
            )
            print("Useful to see if our features interact meaningfully")
            print("Ignore the 'intercept' block below")
            print("=" * 8)
            manova = MANOVA.from_formula("size + aspect_ratio ~ label", data=df)
            print(manova.mv_test())


def _debug_plots(
    mdata: pd.DataFrame,
    coeffs: np.ndarray,
    metrics: pd.DataFrame,
    harmonic_cutoff: int,
    debug_plot_dir: pathlib.Path,
    labels: np.ndarray,
) -> None:
    """
    Make some debugging plots and save them.

    :param mdata: metadata df (contains paths, sex, age, etc.)
    :param coeffs: EFA coefficients
    :param metrics: the size/aspect ratio/bumpiness metrics
    :param debug_plot_dir: where to save this nonsense
    """
    plotting.plot_metadata_bars(mdata, debug_plot_dir)

    ellipse_dir = debug_plot_dir / "efa_plots"
    ellipse_dir.mkdir()

    ellipse_sizes, ellipse_ars = [], []
    img_sums = []
    for path, mag, c in zip(
        mdata["path"], mdata["magnification"], tqdm(coeffs), strict=True
    ):
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))

        # Plot the image and a best-fit ellipse
        img = efa._fix_segmentation(path)
        s, a = _get_ellipse(img, mag, axes[0])
        ellipse_sizes.append(s)
        ellipse_ars.append(a)

        # Track the size of the image
        img_sums.append(img.sum() / (255 * mag**2))

        # Plot the EFA fits, including the truncated/debumped one
        plotting.plot_unscaled_efa((0, 0), c, axis=axes[1])
        plotting.plot_unscaled_efa((0, 0), c[:harmonic_cutoff], axis=axes[2])

        axes[0].set_title("Naive Ellipse Fit")
        axes[1].set_title("EFA Expansion")
        axes[2].set_title(f"EFA - first {harmonic_cutoff} harmonics")

        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])

        fig.tight_layout()
        fig.savefig((ellipse_dir / pathlib.Path(path).name).with_suffix(".png"))
        plt.close(fig)

    # - plots comparing the segmentation mask size/elliptical fit size/area from EFA and aspect ratio from ellipse + EFA
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].scatter(ellipse_sizes, metrics["size"], label="Ellipse Fit", s=36)
    axes[0].scatter(img_sums, metrics["size"], label="Sum of Pixels", s=9)
    axes[0].set_xlabel("Size from ellipse fit/image sum")
    axes[0].set_ylabel("Size from EFA coeffs")

    axes[1].scatter(ellipse_ars, metrics["aspect_ratio"], s=36)

    for ax in axes:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot([x0, x1], [y0, y1], "k--")
        ax.set_xlim([x0, x1])
        ax.set_ylim([y0, y1])
    fig.tight_layout()
    fig.savefig(debug_plot_dir / "efa_size_ar_comparison.png")
    plt.close(fig)

    # - a grid of 16 randomly chosen scales, in order of bumpiness
    n_scales = 16
    percentiles = np.linspace(0, 100, n_scales, endpoint=True)
    values = np.percentile(metrics["bumpiness"], percentiles)
    indices = [np.argmin(np.abs(metrics["bumpiness"] - v)) for v in values]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for axis, i in zip(axes.flat, indices):
        axis.imshow(efa._fix_segmentation(mdata["path"][i]), cmap="binary")
        axis.set_title(f"bumpiness: {metrics['bumpiness'][i]:.4f}")
        axis.set_axis_off()
    fig.suptitle("Scales in order of bumpiness")
    fig.tight_layout()
    fig.savefig(debug_plot_dir / "bumpiness.png")
    plt.close(fig)

    # Coeff power spectrum
    fig, axis = plt.subplots()
    labels = np.array(labels)
    power = (coeffs**2).sum(axis=2).T
    for label in np.unique(labels):
        power_ = power[:, labels == label]
        x = np.arange(len(power_))
        axis.plot(x, np.median(power_, axis=1), label=label)
        lower, upper = np.quantile(power_, [0.25, 0.75], axis=1)
        axis.fill_between(x, lower, upper, alpha=0.3)
    axis.set_yscale("log")
    axis.axvline(harmonic_cutoff, color="k")
    axis.text(harmonic_cutoff + 0.5, axis.get_ylim()[0] * 5, "Bump harmonics->")

    fig.suptitle("EFA Fourier Power Spectrum")
    axis.set_title("Median/IQR of classes")
    axis.legend()
    fig.tight_layout()

    fig.savefig(debug_plot_dir / "fourier_power.png")
    plt.close(fig)


def main(
    *,
    segmentation_dir: pathlib.Path,
    classes: list[str],
    debug_plot_dir: pathlib.Path | None,
    output_dir: pathlib.Path,
    query: str | None,
    n_edge_points: int,
    n_total_harmonics: int,
    harmonic_cutoff: int,
    efa_cache_path: pathlib.Path | None,
):
    """
    Read in the scales, perform EFA and use the EFA coefficients
    to define some simple features that we can use to distinguish our classes.

    Optionally, make some debug plots showing a simple elliptical
    fit to the images, the correlation between our EFA features and
    the area/ellipse area/ellipse aspect ratio and also showing the
    Fourier power spectrum.
    """
    scale_paths = list(str(p) for p in segmentation_dir.glob("*.tif"))

    # Get the metadata
    mdata = metadata.df(scale_paths, default_stain="ALP")
    if query is not None:
        mdata = mdata.query(query).reset_index()
    del scale_paths  # Don't want to accidentally use this later now that we have sliced the df

    # Make sure that we've not messed anything up
    find_separability = mdata[classes].drop_duplicates().shape[0] > 1
    if not find_separability:
        unique = mdata[classes].drop_duplicates()
        warnings.warn(
            f"Only {unique.shape[0]} classes found in metadata ({unique}) based on labels for {classes}.\n"
            "We cannot find stats on separability here."
        )
        del unique

    # Get the EFA coeffs
    coeffs = _efa_coeffs(
        mdata["path"], efa_cache_path, n_edge_points, n_total_harmonics
    )

    # Let's just be sure and check that we have the right number of coeffs
    if len(coeffs) != len(mdata):
        raise ValueError(
            f"Got {len(coeffs)=} but {len(mdata)=}; "
            f"was the EFA coeff cache {efa_cache_path} used for a different "
            f"input dataset or with a different query ({query=} here)?"
        )

    # Get the metrics from the coeffs
    metrics = efa.shape_features(
        coeffs, mdata["magnification"], bump_threshold=harmonic_cutoff
    )
    mdata = mdata.join(metrics)

    # Make output dirs
    output_dir.mkdir(exist_ok=False, parents=True)
    if debug_plot_dir:
        debug_plot_dir = output_dir / debug_plot_dir
        debug_plot_dir.mkdir(exist_ok=False, parents=True)

    # Make pairplots of them
    # Ideally, use the tab10 colourmap
    # If we have more than 10 categories, instead use equally-spaced
    # CSS colours
    num_unique = mdata[classes].drop_duplicates().shape[0]
    colours = TABLEAU_COLORS if num_unique <= 10 else CSS4_COLORS
    colours = list(colours.keys())
    if num_unique > 10:
        idxs = np.linspace(0, len(colours) - 1, num_unique, dtype=int)
        colours = [colours[i] for i in idxs]

    fig = plotting.pair_plot(
        metrics.to_numpy(),
        mdata[classes],
        colours[:num_unique],
        axis_label="Metric",
        normalise=True,
    )
    fig.set_title(query)

    # Rename the axis labels
    axes = np.array(fig.get_axes()).reshape(3, 3)
    for axis, title in zip(axes[:, 0], ["Size", "Aspect Ratio", "Bumpiness"]):
        axis.set_ylabel(title)
    for axis, title in zip(axes[-1], ["Size", "Aspect Ratio", "Bumpiness"]):
        axis.set_xlabel(title)
    plt.savefig(output_dir / "pair_plot.png")
    plt.close(fig)

    # Print stats to a file
    labels = mdata[classes].astype(str).agg(" | ".join, axis=1)
    if find_separability:
        _statstests(mdata, labels, output_dir)

    # Optional - make any last debug plots
    if debug_plot_dir:
        _debug_plots(mdata, coeffs, metrics, harmonic_cutoff, debug_plot_dir, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "segmentation_dir",
        help="The directory of scale segmentations. These must be .tif files named such that we can extract"
        "the metadata from each filename.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "classes",
        nargs="+",
        type=str,
        help="Which classes to split the data based on",
        choices={"age", "sex", "growth"},
    )
    parser.add_argument(
        "--debug-plot-dir",
        type=pathlib.Path,
        help="If provided, will store extra images for debugging here."
        "This will greatly slow things down but will be useful for debugging."
        "The extra plots include:\n"
        " - a bar chart of the metadata\n"
        " - the EFA fits\n"
        " - the EFA fits, truncated up to the harmonic_cutoff\n"
        " - elliptical fits to the segmentation mask\n"
        " - plots comparing the segmentation mask size/elliptical fit size/area from EFA and aspect ratio from ellipse + EFA\n"
        " - a grid of 16 representative scales, in order of bumpiness\n",
        default=None,
    )

    default_out_dir = "outputs/3-quick_ellipse_analysis/"
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help=f"Where outputs get stored. Defaults to {default_out_dir}.",
        default=default_out_dir,
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional pandas query string to filter the dataframe "
        "e.g. 'sex == \"F\"' or '(sex != \"?\") & (age == 19)'",
        default=None,
    )

    parser.add_argument(
        "--n_edge_points",
        type=int,
        help="Number of points around the scales edge to use for defining its shape",
        default=300,
    )
    parser.add_argument(
        "--n_total_harmonics",
        type=int,
        help="The total number of EFA harmonics to use to describe the scales' shapes",
        default=25,
    )
    parser.add_argument(
        "--harmonic_cutoff",
        type=int,
        help="The number of harmonics above which we consider features"
        "'small', which means they will be considered 'bumps'",
        default=5,
    )
    parser.add_argument(
        "--efa_cache_path",
        type=pathlib.Path,
        help="If provided, will write to/attempt to read the EFA coeffs from here."
        "Useful for speeding up multiple runs",
        default=None,
    )

    main(**vars(parser.parse_args()))
