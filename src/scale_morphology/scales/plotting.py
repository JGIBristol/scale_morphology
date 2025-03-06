"""
Utilities for data viz
"""

import numpy as np
import matplotlib.pyplot as plt

from . import efa


def plot_efa(scale: np.ndarray, coeffs: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the EFA of a scale

    """
    fig, axis = plt.subplots()

    # Find the centre of the scale
    locus = [np.average(x) for x in np.where(scale > 0)]

    x, y = efa.coeffs2points(coeffs, locus)

    axis.plot(x, y, "r.")
    axis.imshow(scale, cmap="gray")

    fig.tight_layout()

    return fig, axis
