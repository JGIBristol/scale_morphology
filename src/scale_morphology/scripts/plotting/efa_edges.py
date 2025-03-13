"""
Plot the scales and the edges used for EFA
"""

from tqdm import tqdm
import matplotlib.pyplot as plt

from scale_morphology.scales import read, plotting, efa


def main():
    """
    Read in the images and EFA coefficients, plot the scale and the EFA approximation
    Plot also the edges discovered by the edgefinding algorithm

    """
    out_dir = read.output_dir()
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    paths = read.greyscale_paths()
    images = read.greyscale_images(progress=True)

    coeffs = read.read_coeffs("efa")

    for path, image, coeff in zip(tqdm(paths), images, coeffs):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        # Show image
        axes[0].imshow(image.T, cmap="gray", origin="upper")
        axes[1].set_aspect("equal")

        # Show EFA
        plotting.plot_efa(
            image,
            coeff,
            axis=axes[1],
            color="r",
            label="EFA",
            marker=".",
            linestyle="none",
        )

        # Show detected edges
        x, y = efa.points_around_edge(image, 300)
        axes[0].plot(x, y, "b.", markersize=2, label="Edges")

        axes[0].set_title("Original Image")
        axes[1].set_title("EFA")

        fig.suptitle(path.stem)
        fig.tight_layout()

        fig.savefig(out_dir / f"efa_{path.stem}.png")

        plt.close(fig)


if __name__ == "__main__":
    main()
