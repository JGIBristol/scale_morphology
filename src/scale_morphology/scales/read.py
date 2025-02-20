"""
Utilities for reading data, files, etc.

"""

import yaml
import imageio
import pathlib


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


def segmentations() -> list[pathlib.Path]:
    """
    Get all the segmentations in the data directory

    """
    return [imageio.imread(path) for path in data_dir().glob("*.tif")]
