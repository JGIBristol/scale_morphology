import numpy as np

from ..scales import dim_reduction


def test_nan_2d():
    """
    Check we can correctly mask out NaNs for 2d indices
    """
    coeffs = np.array(
        [1, 2, 3],
        [1, np.nan, 3],
        [np.nan, np.nan, np.nan],
    )
    expected = np.array([False, True, True])

    assert (dim_reduction.nan_scale_mask(coeffs) == expected).all()


def test_nan_3d_wrong_shape():
    """
    Check the right error is raised if the 3d array doesnt have 4-length coeffs
    """


def test_nan_3d():
    """
    Check we can correctly mask out NaNs for 2d indices
    """


def test_flatten():
    """
    Check we get the right shape when we flatten an array of EFA coeffs

    """
    # 2 scales, 3 harmonics
    efa_coeffs = np.array(
        [
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            [
                [2, 3, 4, 5],
                [6, 7, 8, 9],
                [10, 11, 12, 13],
            ],
        ]
    )

    flattened = dim_reduction._flatten(efa_coeffs)

    assert flattened.shape == (2, 12)
    assert (
        flattened
        == np.array(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            ]
        )
    ).all()
