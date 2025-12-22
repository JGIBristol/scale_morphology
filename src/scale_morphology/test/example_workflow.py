"""
An example workflow -- starting with segmented shapes, run EFA, PCA then LDA.

We will randomly generate some triangles/squares/ellipses (these will be our
different classes), run EFA to express the shapes in terms of their size
and outline shape then run PCA to reduce the dimensionality of the EFA space
and finally LDA to see if we can separate between our classes.

This is a toy version of the real workflow, where instead of
circles/squares/triangles we will have fish scales from fish of different ages/
genotypes/sex etc.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.draw import random_shapes

from scale_morphology.scales import efa, plotting


def _shape(shape_name: str, rng: np.random.Generator) -> np.ndarray:
    """
    Randomly generate an image
    """
    d = 128
    # Keep retrying to create a shape until we get one that
    # is fully inside the image - we don't want any cut-off
    # triangles or anything, or this will have a different
    # outline to the others
    img = None
    while img is None:
        img_, ((_, bbox),) = random_shapes(
            (d, d),
            max_shapes=1,
            min_shapes=1,
            num_channels=1,
            min_size=32,
            max_size=96,
            shape=shape_name,
            intensity_range=(0, 1),
            rng=rng,
        )

        bbox = np.array(bbox).ravel()
        if (bbox > 0).all() and (bbox < d).all():
            img = img_

    # Remove channel dimension
    img = img.squeeze()

    # Flip img to 0 bkg, 255 foreground
    img = img - img.min()
    img = img / img.max()
    img = ~img.astype(bool)

    return (img * 255).astype(np.uint8)


def _plot_shapes(triangles, rectangles, ellipses):
    """
    Plot the shapes we used as an example
    """
    kw = {"cmap": "binary"}
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for axis, triangle in zip(axes[0], triangles):
        axis.imshow(triangle, **kw)
    for axis, rectangle in zip(axes[1], rectangles):
        axis.imshow(rectangle, **kw)
    for axis, ellipse in zip(axes[2], ellipses):
        axis.imshow(ellipse, **kw)
    for axis in axes.flat:
        axis.set_xticklabels([])
        axis.set_yticklabels([])

    fig.tight_layout()
    fig.savefig("example_shapes.png")


def main():
    """
    Generate some random shapes, perform EFA/PCA/LDA, then plot a scatter plot
    and a plot showing a few example shapes.
    """
    rng = np.random.default_rng()

    # Generate some shapes
    n_low, n_high = 50, 100
    triangles = [_shape("triangle", rng) for _ in range(rng.integers(n_low, n_high))]
    rectangles = [_shape("rectangle", rng) for _ in range(rng.integers(n_low, n_high))]
    ellipses = [_shape("ellipse", rng) for _ in range(rng.integers(n_low, n_high))]

    # Record the labels, for later plotting/LDA
    shapes = np.vstack([triangles, rectangles, ellipses])
    labels = np.concatenate(
        [
            np.full(len(triangles), "Triangle"),
            np.full(len(rectangles), "Rectangle"),
            np.full(len(ellipses), "Ellipse"),
        ]
    )
    label_df = pd.DataFrame({"label": labels})

    # Run EFA
    coeffs = [efa.coefficients(shape, 100, 25) for shape in shapes]

    # Run PCA then LDA
    colours = ["blue", "orange", "green"]
    pca_coeffs = PCA(n_components=10).fit_transform(coeffs)
    lda_coeffs = LDA().fit_transform(pca_coeffs, labels)

    # Plot both of these
    plotting.pair_plot(pca_coeffs, label_df, colours, axis_label="PC")
    plt.gcf().savefig("example_pca.png")

    plotting.pair_plot(lda_coeffs, label_df, colours, axis_label="LD Axis")
    plt.gcf().savefig("example_LDA.png")

    _plot_shapes(triangles, rectangles, ellipses)


if __name__ == "__main__":
    main()
