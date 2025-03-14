"""
Create a dashboard to visualise the dimensionality reduction
of the EFA, autoencoder, or VAE coefficients.

"""

import warnings
import argparse

import numpy as np

from scale_morphology.scales import read, dim_reduction
from scale_morphology.scales import dashboard


def main(*, compression_method: str, dim_reduction_method: str, progress: bool) -> None:
    """
    Read in the specified coefficients then create the dashboard
    """
    # Read the coefficients
    coeffs = read.read_coeffs(compression_method)


    # Find the indices where the coeffs are NaN and drop them
    nan_rows = dim_reduction.nan_scale_mask(coeffs)
    if nan_rows.any():
        warnings.warn(f"{nan_rows.sum()} NaNs in the coefficients - these will be dropped")

    # Perform the dimensionality reduction
    # We only need to flatten the EFA coefficients
    red_method = dim_reduction.get_dim_reduction(dim_reduction_method)
    reduced = red_method(coeffs, flatten=(compression_method == "efa"), drop=nan_rows)

    out_dir = read.output_dir() / "dashboards"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    dashboard.write_dashboard(
        reduced,
        f"{out_dir / '_'.join([compression_method, dim_reduction_method])}.html",
        progress=progress,
        drop=nan_rows,
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

    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli()
