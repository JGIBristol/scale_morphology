"""
Utilities for reading data, files, etc.

"""

import yaml
import imageio
import pathlib
import warnings

import numpy as np
from tqdm import tqdm
from PIL import Image
from readlif.reader import LifFile


class LIFError(Exception):
    """
    Something went wrong reading a LIF file
    """


def _thisfile() -> pathlib.Path:
    """
    Get the path of this file.

    """
    return pathlib.Path(__file__)


def _root() -> pathlib.Path:
    """
    Get the root directory of the project.

    """
    return _thisfile().parents[3]


def output_dir() -> pathlib.Path:
    """
    Where outputs go
    """
    return _root() / "output"


def config() -> dict:
    """
    Read the config file and return it as a dictionary.

    """
    with open(_root() / "config.yml") as file:
        return yaml.safe_load(file)


def _data_dir() -> pathlib.Path:
    """
    The directory holding data

    """
    return _root() / "data"


def raw_data_dir() -> pathlib.Path:
    """
    Get the directory holding the raw image data

    """
    return _data_dir() / config()["binary_img_dir"]


def _raw_paths() -> list[pathlib.Path]:
    """
    Get all the paths to the raw data

    """
    return sorted(list(raw_data_dir().glob("*.tif")))


def raw_segmentations(*, progress: bool = False) -> np.ndarray:
    """
    Get all the segmentations in the raw data directory

    """
    paths = _raw_paths()

    # If they greyscale images don't exist, create them
    if not paths:
        raise FileNotFoundError(f"No data found in {raw_data_dir()}")

    if progress:
        paths = tqdm(paths, desc="Reading raw segmentations")
    return [imageio.imread(path) for path in paths]


def greyscale_dir() -> pathlib.Path:
    """
    The directory holding the pre-processed segmentations

    """
    return _data_dir() / config()["processed_img_dir"]


def greyscale_paths() -> list[pathlib.Path]:
    """
    Get a list of paths to the greyscale images

    """
    return sorted(list(greyscale_dir().glob("*.tif")))


def _create_greyscale_tiffs(*, progress=True) -> None:
    """
    The raw segmentations on OneDrive will be RGB images,
    in which case we want to standardise them- save them all as
    greyscale, 8 bit unsigned tiffs

    """
    out_dir = greyscale_dir()
    if not out_dir.exists():
        out_dir.mkdir()

    input_paths = _raw_paths()
    raw_imgs = raw_segmentations(progress=progress)

    raw_imgs = tqdm(raw_imgs, desc="Saving greyscale images") if progress else raw_imgs
    for path, img in zip(input_paths, raw_imgs):
        imageio.imsave(out_dir / path.name, np.mean(img, axis=2).astype(np.uint8))


def greyscale_images(*, progress: bool = False) -> np.ndarray:
    """
    Get all the greyscale images

    """
    paths = greyscale_paths()

    if not paths:
        warnings.warn("No greyscale images found, creating them now")
        _create_greyscale_tiffs(progress=progress)
        paths = greyscale_paths()

    if progress:
        paths = tqdm(paths, desc="Reading greyscale images")
    return [imageio.imread(path) for path in paths]


def _coeff_dir() -> pathlib.Path:
    """
    Get the directory holding the coefficients

    """
    return _data_dir() / "coeffs"


def _efa_coeff_path() -> pathlib.Path:
    """
    Get the path to the EFA coefficients

    """
    return _coeff_dir() / "efa_coeffs.npy"


def _autoencoder_coeff_path() -> pathlib.Path:
    """
    Get the path to the autoencoder coefficients

    """
    return _coeff_dir() / "autoencoder_coeffs.npy"


def _vae_coeff_path() -> pathlib.Path:
    """
    Get the path to the VAE coefficients

    """
    return _coeff_dir() / "vae_coeffs.npy"


def coeff_path(compression_method: str) -> pathlib.Path:
    """
    Get the path to the coefficients

    """
    match compression_method:
        case "efa":
            return _efa_coeff_path()
        case "autoencoder":
            return _autoencoder_coeff_path()
        case "vae":
            return _vae_coeff_path()
        case _:
            raise ValueError(f"Unknown compression method: {compression_method}")


def write_coeffs(coeffs: np.ndarray, compression_method: str) -> None:
    """
    Write the EFA coefficients to disk

    """
    path = coeff_path(compression_method)
    if not path.parent.exists():
        path.parent.mkdir()

    np.save(path, coeffs)


def read_coeffs(compression_method: str) -> np.ndarray:
    """
    Read the EFA coefficients from disk

    """
    return np.load(coeff_path(compression_method))


def read_2d_lif(lif_path: pathlib.Path) -> list[Image]:
    """
    Read all the images from a LIF file, assuming the contents
    contain only one time-point and mosaic image.

    :param lif_path: path to the LIF file.

    :return: a list of the images, as PIL Images.
    :raises LIFError: if the LIF files do not conform to the above assumptions.
    """
    lif_file = LifFile(lif_path)
    all_images = [image for image in tqdm(lif_file.get_iter_image())]

    for i, image in enumerate(all_images):
        if image.channels != 3 or image.nz != 1 or image.nt != 1 or image.n_mosaic != 1:
            raise LIFError(
                "Expected RGB image with 1 timepoint and 1 z-img; got"
                f"\t{image.channels=}, {image.nz=}, {image.nt=}, {image.n_mosaic=} for image {i}"
                f"in {lif_path}"
            )

    return [image.get_frame(z=0, t=0, m=0)]
