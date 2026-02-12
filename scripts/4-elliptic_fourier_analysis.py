"""
Run the EFA analysis, then attempt to interpret the coefficients
by using PCA and LDA for dimensionality reduction.

"""

import pathlib
import argparse

import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scale_morphology.scales import plotting, efa, metadata


def _efa_coeffs(
    segmentation_paths: pd.Series,
    efa_cache_path: pathlib.Path | None,
    magnifications: pd.Series,
    n_edge_points: int,
    n_harmonics: int,
) -> np.ndarray:
    """
    Get the EFA coefficients - either from a cache, or by reading the images from
    paths and running the analysis on them
    """
    if efa_cache_path is not None and efa_cache_path.is_file():
        return np.load(efa_cache_path)

    coeffs = efa.run_analysis(
        segmentation_paths,
        magnifications,
        n_points=n_edge_points,
        order=n_harmonics,
        n_threads=16,  # Hopefully this doesn't cause any trouble
    )

    if efa_cache_path is not None:
        np.save(efa_cache_path, coeffs)

    return coeffs


def _plot_efa(
    segmentation_paths: pd.Series,
    efa_coeffs: np.ndarray,
    plot_dir: pathlib.Path,
    n_edge_points: int,
    n_harmonics: int,
) -> None:
    """
    Plot the segmentations and the EFA fit and save them in the provided dir
    """
    assert len(segmentation_paths) == len(
        efa_coeffs
    ), f"Got {len(segmentation_paths)=} but {len(efa_coeffs)=}"

    for path, coeff in zip(tqdm(segmentation_paths), efa_coeffs):
        fig, axis = plt.subplots(figsize=(4, 4))
        img = tifffile.imread(path)

        axis.imshow(img, cmap="binary")
        axis.set_aspect("equal")

        plotting.plot_efa(
            np.mean(np.where(img > 0), axis=1),
            coeff,
            label=f"EFA best fit, {n_harmonics=}",
            linewidth=3,
            color="#ff00fa",
            axis=axis,
        )

        x, y = efa.points_around_edge(img, n_edge_points)
        axis.plot(
            x, y, "#00ff05", markersize=3, label="Edges", marker="o", linestyle="none"
        )

        axis.set_axis_off()
        axis.legend()
        fig.savefig(plot_dir / (pathlib.Path(path).with_suffix(".png")).name)
        plt.close(fig)


def _group_colours(grouping_df: pd.DataFrame):
    unique_groups = grouping_df.drop_duplicates()
    n_unique = len(unique_groups)

    colours = colors.TABLEAU_COLORS if n_unique <= 10 else colors.XKCD_COLORS
    return list(colours.values())[:n_unique]


def main(
    *,
    segmentation_dir: pathlib.Path,
    classes: list[str],
    output_dir: pathlib.Path,
    n_harmonics: int,
    n_edge_points: int,
    n_pcs: int,
    efa_cache_path: pathlib.Path | None,
):
    """
    Read in the segmentations, caching the coefficients if required,
    and perform EFA. Then run PCA, make some plots, run LDA, and make further
    plots
    """
    # Cache must be a numpy archive
    efa_cache_path = efa_cache_path.with_suffix(".npy")

    output_dir.mkdir(parents=True)

    # Get the metadata
    scale_paths = list(str(p) for p in segmentation_dir.glob("*.tif"))
    mdata = metadata.df(scale_paths, default_stain="ALP")
    plotting.plot_metadata_bars(mdata, output_dir)

    # Get the EFA coeffs
    coeffs = _efa_coeffs(
        mdata["path"],
        efa_cache_path,
        mdata["magnification"],
        n_edge_points,
        n_harmonics,
    )

    # We will use this dataframe to find which class each scales
    # belongs to
    grouping_df = mdata.loc[:, classes]
    colours = _group_colours(grouping_df)

    # Run PCA
    pca = PCA(n_components=n_pcs)
    pca_coeffs = pca.fit_transform(coeffs)

    # Plot PCA heatmap + feature importance
    fig = plotting.pca_barplot(pca.components_)
    fig.savefig(output_dir / "pca_barplot.png")
    plt.close(fig)

    fig = plotting.heatmap(pca.components_)
    fig.savefig(output_dir / "pca_heatmap.png")
    plt.close(fig)

    fig = plotting.feature_importance(pca)
    fig.savefig(output_dir / "pca_feature_importance.png")
    plt.close(fig)

    # Pairplot of PCA coeffs
    fig = plotting.pair_plot(
        pca_coeffs,
        grouping_df,
        colours,
        axis_label="PC",
        normalise=True,
    )
    fig.savefig(output_dir / "pca_pairplot.png")
    plt.close(fig)

    # Run LDA based on the labels
    # Get labels for our different categories - we need to label each row in our dataframe
    # with which group it belongs to
    labels, uniques = pd.factorize(
        grouping_df.apply(lambda row: tuple(row.values), axis=1)
    )
    # Now we cannot choose how many components to use in our dimensionality reduction;
    # LDA just finds the best (N-1) axes to distinguish our N classes.
    # Technically we could use any number less than N-1, but we want to keep all of them
    lda = LinearDiscriminantAnalysis()
    lda_coeffs = lda.fit_transform(pca_coeffs, labels)

    # Plot LDA heatmap
    fig = plotting.heatmap((pca.components_.T @ lda.scalings_).T)
    fig.savefig(output_dir / "lda_heatmap.png")
    plt.close(fig)

    fig = plotting.pair_plot(lda_coeffs, grouping_df, colours, axis_label="LDA")
    fig.savefig(output_dir / "lda_pairplot.png")
    plt.close(fig)

    # Plot LDA bar plot
    fig = plotting.pca_barplot((pca.components_.T @ lda.scalings_).T)
    fig.savefig(output_dir / "lda_barplot.png")
    plt.close(fig)

    # Run k-fold cross validation on LDA to get score


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

    default_out_dir = "outputs/4-elliptic_fourier_analysis/"
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help=f"Where outputs get stored. Defaults to {default_out_dir}.",
        default=default_out_dir,
    )
    parser.add_argument(
        "--n_harmonics",
        type=int,
        help="Number of points around the scales edge to use for defining its shape",
        default=30,
    )
    parser.add_argument(
        "--n_edge_points",
        type=int,
        help="Number of points around the scales edge to use for defining its shape",
        default=300,
    )
    parser.add_argument(
        "--n_pcs",
        type=int,
        help="Number of principal components to use when reducing the EFA coefficient dimensionality",
        default=10,
    )
    parser.add_argument(
        "--efa_cache_path",
        type=pathlib.Path,
        help="If provided, will write to/attempt to read the EFA coeffs from here."
        "Useful for speeding up multiple runs",
        default=None,
    )

    main(**vars(parser.parse_args()))
