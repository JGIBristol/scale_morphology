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


def data_dir() -> pathlib.Path:
    """
    Get the directory holding the data

    """
    return _root() / "data" / config()["binary_img_dir"]


def segmentations(*, progress: bool = False) -> np.ndarray:
    """
    Get all the segmentations in the data directory

    """
    paths = list(data_dir().glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"No data found in {data_dir()}")

    if progress:
        paths = tqdm(paths, desc="Reading segmentations")
    return [imageio.imread(path) for path in paths]

