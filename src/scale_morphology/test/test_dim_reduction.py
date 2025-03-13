import numpy as np

from ..scales import dim_reduction


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
