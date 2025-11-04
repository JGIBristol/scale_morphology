"""
Get fish metadata from filepaths.

Quite empirical - based on how the files have come out - so
isn't that comprehensively documented. Just figure it out

"""

import re
from functools import cache

import numpy as np


def stain(name: str) -> str:
    """
    Get the stain from the path to an image (or segmentation).

    Basically just looks for the stain in the name - ALP, TRAP or KOSSA.
    """
    (stem, ending) = name.split(".tif")
    assert not ending, f"{stem}, {ending}"

    vk, alp, trap = False, False, False
    if "KOSSA" in stem or "VK" in stem:
        vk = True
    if "ALP" in stem:
        alp = True
    if "TRAP" in stem:
        trap = True

    if vk + alp + trap != 1:
        raise ValueError(f"Unknown stain for {stem}; {vk=}, {alp=}, {trap=}")

    if vk:
        return "KOSSA"
    if alp:
        return "ALP"
    return "TRAP"


def get_year(stem):
    """
    Extract years from a filepath
    """
    if "year" not in stem:
        return 0

    year_str, _ = stem.split("year")
    return int(year_str.strip()[-1])


def get_month(stem):
    """
    Extract months from a filepath
    """
    if not "month" in stem:
        return 0
    month_str, _ = stem.split("month")
    month_str = month_str.strip()

    # We might have a 2-digit month, so check for that here
    if month_str[-2].isdigit():
        return int(month_str[-2:])
    return int(month_str[-1])


def age(stem):
    """Get age in months"""
    return 12 * get_year(stem) + get_month(stem)


def sex(stem: str) -> str:
    """
    Get the sex from a filepath, if we know it (might be ?)
    """
    if "female" in stem:
        return "F"
    if "male" in stem:
        return "M"
    return "?"


@cache
def _mag_regex():
    """regex used to find magnification"""
    return re.compile(r"^.*(\d\.\d)X.*$")


def magnification(path: str) -> float:
    """
    Get the magnification from a filepath, if specified - else np.nan
    """
    if match := _mag_regex().match(path):
        return float(match.group(1))
    return np.nan
