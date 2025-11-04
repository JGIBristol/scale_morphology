"""
Test the metadata reading fcns
"""

import numpy as np

from scale_morphology.scales import metadata


def test_get_age_months():
    """
    Check we can get an age from a filepath containing months only
    """
    assert (
        metadata.age(
            "Fish2_Onto_scale009__OSX_mcherry_2021_4months_ALP_segmentation.tif"
        )
        == 4
    )


def test_get_age_month_year():
    """
    Check we can get an age from a filepath containing months and years
    """
    assert (
        metadata.age(
            "Fish1_male_D10reg_scale004__2022_1year6months_OMD_ALP_segmentation.tif"
        )
        == 18
    )


def test_get_sex():
    """
    Check we can get the sex from a filepath (including sex-not-found)
    """
    assert (
        metadata.sex(
            "Fish1_female_D10reg_scale005__2021_spp1mutant_HomMch_7months_ALP_segmentation.tif"
        )
        == "F"
    )
    assert (
        metadata.sex(
            "Fish1_male_D10reg_scale004__2022_1year6months_OMD_ALP_segmentation.tif"
        )
        == "M"
    )
    assert (
        metadata.sex(
            "Fish3_D10reg_scale001__OSX_mcherry_2021_4months_ALP_segmentation.tif"
        )
        == "?"
    )


def test_get_stain():
    """
    Get the stain (ALP, VK, TRAP) from a filepath
    """
    assert (
        metadata.stain(
            "Fish1_male_D10reg_scale004__2022_1year6months_OMD_ALP_segmentation.tif"
        )
        == "ALP"
    )


def test_get_magnification():
    """
    Check we can get the magnification from a filepath, where it exists
    """
    assert (
        metadata.magnification(
            "Fish1_male_Onto_3.2X_scale006__3year4months_ALP_segmentation.tif"
        )
        == 3.2
    )
    assert np.isnan(
        metadata.magnification(
            "Fish3_D10reg_scale001__OSX_mcherry_2021_4months_ALP_segmentation.tif"
        )
    )
