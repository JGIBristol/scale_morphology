"""
Utilities for reading data, files, etc.

"""

import yaml
import imageio
import pathlib
import warnings

import numpy as np
from tqdm import tqdm


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
    return list(raw_data_dir().glob("*.tif"))


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
    return list(greyscale_dir().glob("*.tif"))


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
