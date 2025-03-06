"""
Elliptical Fourier Analysis (EFA) of scale shapes

"""

import argparse

import numpy as np
from tqdm import tqdm

from ..scales import read, processing, efa


def main(*, n_edge_points: int, progress: bool, order: int) -> None:
    """
    Read in the scale segmentations, find the edges, find evenly spaced
    points along the edges, use these points to perform the EFA
    and then save the coefficients in a numpy array.

    """
    segmented_scales = read.greyscale_images(progress=progress)
    if progress:
        segmented_scales = tqdm(segmented_scales, desc="Processing segmentations")

    coeffs = []
    for scale in segmented_scales:
        try:
            coeffs.append(efa.coefficients(scale, n_edge_points, order))
        except efa.BadImgError as e:
            import matplotlib.pyplot as plt
            plt.imshow(scale)
            plt.savefig("tmp.png")
            print(e)

    coeffs = np.row_stack(coeffs)

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

    parser.add_argument(
        "--order", type=int, default=25, help="Number of harmonics to use"
    )

    parser.add_argument("--progress", action="store_true", help="Show progress bars")

    main(**vars(parser.parse_args()))


if __name__ == "__main__":
    cli()
