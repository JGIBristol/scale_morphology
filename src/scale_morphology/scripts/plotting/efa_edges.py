"""
Plot the scales and the edges used for EFA
"""

from scale_morphology.scales import read


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

    coeffs = read.read_efa_coeffs()
    print(len(paths))
    print(len(images))
    print(coeffs.shape)


if __name__ == "__main__":
    main()
