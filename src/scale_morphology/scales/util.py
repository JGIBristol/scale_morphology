"""
General utilities
"""

import pathlib
from typing import Any
from functools import cache

import yaml


@cache
def config() -> dict[str, Any]:
    """
    Get the configuration.
    """
    with open(
        pathlib.Path(__file__).parents[3] / "config.yml", "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f)
