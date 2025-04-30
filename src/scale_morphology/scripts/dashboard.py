"""
Create a dashboard to visualise the dimensionality reduction
of the EFA, autoencoder, or VAE coefficients.

"""

import warnings
import argparse

import numpy as np

from scale_morphology.scales import read, dim_reduction
from scale_morphology.scales import dashboard


def main(
    *,
    compression_method: str,
    dim_reduction_method: str,
    progress: bool,
    colour_coding: None | str,
) -> None:
    """
    Read in the specified coefficients then create the dashboard
    """
    # Read the coefficients
    coeffs = read.read_coeffs(compression_method)

    # Find the indices where the coeffs are NaN and drop them
    nan_rows = dim_reduction.nan_scale_mask(coeffs)
    if nan_rows.any():
        warnings.warn(
            f"{nan_rows.sum()} NaNs in the coefficients - these will be dropped"
        )

    # Perform the dimensionality reduction
    # We only need to flatten the EFA coefficients
    red_method = dim_reduction.get_dim_reduction(dim_reduction_method)
    reduced, fitter = red_method(
        coeffs, flatten=(compression_method == "efa"), drop=nan_rows
    )

    out_dir = read.output_dir() / "dashboards"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    plot_kw = {}
    if dim_reduction_method == "pca":
        plot_kw["x_axis_label"] = (
            f"PC1 ({100 * fitter.explained_variance_ratio_[0]:.1f}% variance)"
        )
        plot_kw["y_axis_label"] = (
            f"PC2 ({100 * fitter.explained_variance_ratio_[1]:.3f}% variance)"
        )

    plot_kw["colour_coding"] = colour_coding

    filename = f"{out_dir / '_'.join([compression_method, dim_reduction_method] + ([colour_coding] if colour_coding else []))}.html"
    dashboard.write_dashboard(
        reduced,
        filename,
        progress=progress,
        drop=nan_rows,
        **plot_kw,
    )


def cli():
    """
    Command line interface
    """
    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument(
        "compression_method",
        choices={"efa", "autoencoder", "vae"},
        help="The method used to compress the images to vectors",
    )
    parser.add_argument(
        "dim_reduction_method",
        type=str,
        help="The method used to reduce the dimensionality of the vectors",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bars",
    )
    parser.add_argument(
        "--colour_coding",
        type=str,
        help="The method used to color code the images",
        default=None,
        choices={"regen", "mutation"},
    )

    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli()
