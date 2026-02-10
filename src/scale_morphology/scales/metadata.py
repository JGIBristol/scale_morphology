"""
Get fish metadata from filepaths.

Quite empirical - based on how the files have come out - so
isn't that comprehensively documented. Just figure it out

"""

import re
import sys
import pathlib
from functools import cache

import numpy as np
import pandas as pd


def stain(name: str, default_stain: str | None) -> str:
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
        # Just return the default if we didn't find anything
        if default_stain and (vk + alp + trap == 0):
            return default_stain
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

    # We might have a _-separated filename
    month_str = month_str.strip("_")

    # We might have a 2-digit month, so check for that here
    if month_str[-2].isdigit():
        return int(month_str[-2:])
    return int(month_str[-1])


@cache
def _age_regex():
    return re.compile(r"^(\d+)M.*$")


def age(stem):
    """Get age in months"""
    # It might start with <number>M, in which case we can just pick that out
    if match_ := _age_regex().match(stem):
        return float(match_.group(1))
    return 12 * get_year(stem) + get_month(stem)


def sex(stem: str) -> str:
    """
    Get the sex from a filepath, if we know it (might be ?)
    """
    if "female" in stem or "Fem" in stem:
        return "F"
    if "male" in stem:
        return "M"
    return "?"


@cache
def _mag_regex():
    """regex used to find magnification"""
    # Expect <months>M(<mag>x)_<the rest>
    # Mag can be e.g. 4.0 or 20
    return re.compile(r"^\d+M_\(([0-9.]+)x\)_.*$")


def magnification(stem: str) -> float:
    """
    Get the magnification from a filepath, if specified - else default to 4.0x
    """
    if match := _mag_regex().match(stem):
        return float(match.group(1))
    raise ValueError(f"No magnification found for\n{path}")


@cache
def _growth_regex():
    """e.g. D10reg"""
    return re.compile(r"^.*D(\d+)reg.*$")


@cache
def _growth_regex2():
    """e.g.d10_reg"""
    return re.compile(r"^.*d(\d+)_?reg.*$")


def growth(path: str) -> float:
    """
    Growth status of a scale - onto(genetic) or regen(erating).
    Gives:
        - np.inf if onto
        - the number of days of regeneration, if regen
        - np.nan if not specified

    Assumes the path is labelled by "onto" or "reg", and that this
    is found in the lif filename (which is separated in `path`
    by a double underscore). If this isn't true, will
    """
    try:
        lif_name, _ = path.split("__")
    except ValueError:
        # the filename isn't separated with a double underscore, so just use the whole name
        lif_name = path

    lif_name = lif_name.lower()

    if "onto" in lif_name:
        return np.inf

    if "reg" in lif_name:
        try:
            return float(_growth_regex().match(path).group(1))
        except AttributeError as e:
            try:
                return float(_growth_regex2().match(lif_name).group(1))
            except AttributeError as e2:
                if "_reg_" in lif_name:
                    # We'll assume they're 10 days.. not sure if there's
                    # a strong justification for this, but this is what
                    # we think
                    return 10
                raise

    return np.nan


def no_scale(path: str) -> bool:
    """
    Whether this image contains no scale
    """
    return "no data" in path


def mutation(path: str) -> str:
    """
    Get the mutation, or WT (in all caps)
    """
    path = path.lower()

    # Wildtypes - do this first to
    # catch spp1 sibling wildtypes
    if "wt" in path:
        return "WT"

    # Various mutants
    if "omd" in path:
        return "OMD"
    if "spp1mutant_hom" in path:
        return "SPP1"

    # Fallback is wildtype
    return "WT"


def df(
    paths: list[str], drop_no_scale: bool = True, default_stain: str | None = None
) -> pd.DataFrame:
    """
    Get dataframe of all the metadata.

    The magnification defaults to 4.0x if not specified in the filepath.

    :param paths: list of paths pointing to each scale; the naming convention
                  should be such that we can get the metadata from them.
                  You shouldn't need to think too hard about this - look at the notebooks
                  for examples of what the scales should be named.
    :param drop_empty: whether to drop entries in the table that are tagged as containing
                       no scale.
    :param default_stain: sometimes we won't find a stain in the path - if specified, defaults to this
    """
    retval = pd.DataFrame(
        {
            "path": paths,
            "sex": (sex(p) for p in paths),
            "magnification": (magnification(pathlib.Path(p).stem) for p in paths),
            "age": (age(pathlib.Path(p).stem) for p in paths),
            "stain": (stain(p, default_stain) for p in paths),
            "growth": (growth(p) for p in paths),
            "mutation": (mutation(p) for p in paths),
            "no_scale": (no_scale(p) for p in paths),
        }
    )

    if drop_no_scale:
        return retval[~retval["no_scale"]]
    return retval
