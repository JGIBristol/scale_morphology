"""
Plot the scales and the edges used for EFA
"""

import argparse

from tqdm import tqdm
import matplotlib.pyplot as plt

from scale_morphology.scales import read, plotting, efa


def main(n_imgs: int | None):
    """
    Read in the images and EFA coefficients, plot the scale and the EFA approximation
    Plot also the edges discovered by the edgefinding algorithm

    """
    out_dir = read.output_dir() / "efa_edges"
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    paths = read.greyscale_paths()
    images = read.greyscale_images(progress=True)

    paths = paths[:n_imgs]
    images = images[:n_imgs]

    coeffs = read.read_coeffs("efa")

    for path, image, coeff in zip(tqdm(paths), images, coeffs):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            sharey=True,
        )

        # Show image
        axes[0].imshow(image.T, cmap="gray", origin="lower")
        axes[1].set_aspect("equal")

        # Show EFA
        plotting.plot_efa(
            image,
            coeff,
            axis=axes[1],
            color="r",
            markersize=1,
            linestyle="-",
            linewidth=1,
            label="EFA Approximation",
        )

        # Show detected edges
        x, y = efa.points_around_edge(image, 300)
        axes[0].plot(x, y, "b.", markersize=2, label="Edges")

        axes[0].set_title("Original Image")
        axes[1].set_title("EFA")

        # Honestly no idea what's going on with the axis direction and labels
        # and whatever, but this seems to look right
        axes[1].set_xlim((x for x in axes[0].get_xlim()[::-1]))

        for axis in axes:
            axis.axis("off")
            axis.legend(loc="upper right")

        fig.suptitle(path.stem)
        fig.tight_layout()

        fig.savefig(out_dir / f"efa_{path.stem}.png")

        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__file__.__doc__)
    parser.add_argument("--n_imgs", type=int, default=None, help="n imgs to process")

    main(**vars(parser.parse_args()))
