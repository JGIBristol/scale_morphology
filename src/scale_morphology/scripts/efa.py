"""
Elliptical Fourier Analysis (EFA) of scale shapes

"""

import argparse

from ..scales import read, processing, efa


def main(*, n_edge_points: int, progress: bool) -> None:
    """
    Read in the scale segmentations, find the edges, find evenly spaced
    points along the edges, use these points to perform the EFA
    and then save the coefficients in a numpy array.

    """
    segmented_scales = read.segmentations()

    edge_points = [
        efa.points_around_edge(scale, n_edge_points) for scale in segmented_scales
    ]

    # Perform EFA on the points

    # Store these in an array


def cli():
    """
    CLI for this script
    """
    parser = argparse.ArgumentParser(description="Perform EFA on scale shapes")

    parser.add_argument(
        "--n_edge_points",
        type=int,
        default=100,
        help="Number of equally spaced edge points to use ",
    )

    parser.add_argument("--progress", action="store_true", help="Show progress bars")

    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli()
