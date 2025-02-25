"""
Utilities for reading data, files, etc.

"""

import yaml
import imageio
import pathlib

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


def segmentation_dir() -> pathlib.Path:
    """
    The directory holding the pre-processed segmentations

    """
    return _data_dir() / config()["processed_img_dir"]


def _create_greyscale_tiffs(*, progress=True):
    """
    The raw segmentations on OneDrive will be RGB images,
    in which case we want to standardise them- save them all as
    greyscale, 8 bit unsigned tiffs

    """

def segmentations(*, progress: bool = False) -> np.ndarray:
    """
    Get all the segmentations in the data directory

    """
    paths = list(raw_data_dir().glob("*.tif"))

    # If they greyscale images don't exist, create them
    if not paths:
        raise FileNotFoundError(f"No data found in {raw_data_dir()}")

    if progress:
        paths = tqdm(paths, desc="Reading segmentations")
    return [imageio.imread(path) for path in paths]


def rgb2uint8(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Convert a list of RGB images to uint8 greyscale

    """
    return [np.mean(image, axis=2).astype(np.uint8) for image in images]
