import pytest
import numpy as np

from ..scales import processing


@pytest.fixture
def img_with_holes():
    return 255 * np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


def test_has_holes(img_with_holes: np.ndarray):
    """
    Check we can detect holes properly

    """
    assert processing.has_holes(img_with_holes)

    img_without_holes = 255 * np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert not processing.has_holes(img_without_holes)


def test_fill_bkg(img_with_holes: np.ndarray):
    """
    Check we fill the bkg regions correctly

    """
    assert np.all(
        processing.fill_background(img_with_holes)
        == 255
        * np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
    )
