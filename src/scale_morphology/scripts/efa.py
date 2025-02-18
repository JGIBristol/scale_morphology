"""
Elliptical Fourier Analysis (EFA) of scale shapes

"""


def main() -> None:
    """
    Read in the scale segmentations, find the edges, find evenly spaced
    points along the edges, use these points to perform the EFA
    and then save the coefficients in a numpy array.

    """
