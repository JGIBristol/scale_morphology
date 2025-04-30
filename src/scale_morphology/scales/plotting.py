"""
Utilities for data viz
"""

import numpy as np
import matplotlib.pyplot as plt

from . import efa


def plot_efa(
    scale: np.ndarray, coeffs: np.ndarray, *, axis: plt.Axes, **plot_kw
) -> None:
    """
    Plot the EFA of a scale

    """
    # Find the centre of the scale
    locus = [np.average(x) for x in np.where(scale > 0)]

    x, y = efa.coeffs2points(coeffs, locus[::-1])

    # For some reason it comes out flipped
    axis.plot(-y, x, **plot_kw)
