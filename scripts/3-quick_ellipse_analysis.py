"""
Measure the size and aspect ratio of scales directly.

To get a basic idea of how these features split our dataset,
we can simply fit ellipses to each of the segmentation masks
and then make some basic plots and output some simple statistics.

"""

import pathlib
import argparse

import cv2
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.feature_selection import f_classif
from statsmodels.multivariate.manova import MANOVA
from pingouin import pairwise_gameshowell

from scale_morphology.scales import metadata, ellipse_fit, plotting


def _plot_metadata_bars(mdata: pd.DataFrame, output_dir: pathlib.Path) -> None:
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


def _get_ellipse(
    img: np.ndarray, magnification: float, plot_path: pathlib.Path | None
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
    if plot_path is not None:
        # Convert to BGR for colored ellipse drawing
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(img_color, ellipse, (0, 200, 0), 5)
        cv2.imwrite(str(plot_path), img_color)

    return size, aspect_ratio


def _plot_scatter(
    mdata: pd.DataFrame, classes: list[str], output_dir: pathlib.Path, labels: pd.Series
) -> None:
    """
    Plot scatter plot of our classes along size/aspect ratio axes
    """
    # Get unique class labels and assign colors
    unique_labels = labels.unique()

    colours = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    colour_map = dict(zip(unique_labels, colours))

    plt.matplotlib.use("Agg")  # Not sure why I have to do this to run on scampi
    fig, ax = plt.subplots(figsize=(10, 8))

    plotting._plot_kde_scatter(
        ax, mdata["size"], mdata["aspect_ratio"], labels, colour_map
    )

    def sort_key(label):
        try:
            return (0, float(label))
        except ValueError:
            return (1, label)

    fig.legend(
        handles=[
            Patch(
                facecolor=colour_map[label],
                edgecolor="k",
                linewidth=0.5,
                label=str(label),
            )
            for label in sorted(unique_labels, key=sort_key)
        ],
        loc="center right",
        bbox_to_anchor=(0.97, 0.8),
        title="Group",
    )

    ax.set_xlabel("Size")
    ax.set_ylabel("Aspect Ratio")
    ax.set_title(f"Scale Size vs Aspect Ratio by {', '.join(classes)}")

    plt.tight_layout()
    plt.savefig(output_dir / f"ellipse_scatter_{'_'.join(classes)}.png", dpi=150)


def main(
    *,
    segmentation_dir: pathlib.Path,
    classes: list[str],
    plot_dir: pathlib.Path | None,
    output_dir: pathlib.Path,
):
    """
    Read in the scales, fit ellipses to them, plot histograms and scatter plots of these
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    if plot_dir:
        plot_dir = output_dir / plot_dir
        plot_dir.mkdir(exist_ok=True, parents=True)

    scale_paths = list(str(p) for p in segmentation_dir.glob("*.tif"))

    # Get the metadata
    mdata = metadata.df(scale_paths, default_stain="ALP")
    _plot_metadata_bars(mdata, output_dir)

    # Read in the scales
    sizes = []
    aspect_ratios = []
    for path, magnification in zip(tqdm(mdata["path"]), mdata["magnification"]):
        plot_path = (
            None
            if plot_dir is None
            else plot_dir / str(pathlib.Path(path).stem + ".png")
        )
        s, a = _get_ellipse(tifffile.imread(path), magnification, plot_path)
        sizes.append(s)
        aspect_ratios.append(a)

    mdata["size"] = sizes
    mdata["aspect_ratio"] = aspect_ratios

    labels = mdata[classes].astype(str).agg(" | ".join, axis=1)
    _plot_scatter(mdata, classes, output_dir, labels)

    # Run stats tests
    cols = ["size", "aspect_ratio"]

    # F-test
    f_stat, p_vals = f_classif(mdata[cols], labels)
    print("F-test: can we separate out at least one of our classes from the others?")
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
        {"size": mdata["size"], "aspect_ratio": mdata["aspect_ratio"], "label": labels}
    )
    for col in cols:
        print(f"{col}:")
        print("-" * 4)
        result = pairwise_gameshowell(data=df, dv=col, between="label")
        print(result)

    # MANOVA - joint comparison
    print()
    print("MANOVA test: do class means differ in the joint (size, aspect ratio) space?")
    print("Useful to see if our features interact meaningfully")
    print("Ignore the 'intercept' block below")
    print("=" * 8)
    manova = MANOVA.from_formula("size + aspect_ratio ~ label", data=df)
    print(manova.mv_test())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

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
        "--plot-dir",
        type=pathlib.Path,
        help="If provided, will store the images of the ellipse fits here.\n"
        "Massively slows things down, but useful for debugging. Relative to output-dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Where outputs get stored",
        default="outputs/3-quick_ellipse_analysis/",
    )

    main(**vars(parser.parse_args()))
