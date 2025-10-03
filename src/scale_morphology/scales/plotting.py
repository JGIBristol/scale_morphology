"""
Utilities for data viz
"""

import numpy as np
import matplotlib.pyplot as plt

from . import efa


def plot_efa(
    centroid: tuple[float, float], coeffs: np.ndarray, *, axis: plt.Axes, **plot_kw
) -> None:
    """
    Plot the EFA of a scale

    :param centroid: centre of the scale; can be calculataed from a mask using
                     `[np.average(x) for x in np.where(mask > 0)]`
    :param coeffs: EFD coefficients [[a, b, c, d]_1, [a, b, c, d]_2, ...]
    :param axis: axis to plot on
    :param plot_kw: further keywords to be passed to the plot fcn (e.g. color, linestyle)

    """
    x, y = efa.coeffs2points(coeffs, centroid)
    axis.plot(x, y, **plot_kw)
