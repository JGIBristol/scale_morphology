"""
Test the metadata reading fcns
"""

import numpy as np
import pandas as pd

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


def test_get_growth():
    """
    Check we can get the onto/regen labels correctly
    """
    name = "Fish3_male_D21reg_scale002__2021_spp1mutant_HomMch_7months_ALP_segmentation.tif"
    assert metadata.growth(name) == "regen"

    name = "Fish1_Onto_scale006__OSX_mcherry_GFP_2021_10_months_ALP_segmentation.tif"
    assert metadata.growth(name) == "onto"

    # lowercase o
    name = "Fish1_onto_scale006__OSX_mcherry_GFP_2021_10_months_ALP_segmentation.tif"
    assert metadata.growth(name) == "onto"

    name = "Fish7_3.2X_scale008__3year4months_ALP_segmentation.tif"
    assert metadata.growth(name) == "?"


def test_no_data():
    """
    Check we can correctly identify an image with a missing scale
    """
    assert not metadata.no_scale(
        "Fish7_3.2X_scale008__3year4months_ALP_segmentation.tif"
    )
    assert metadata.no_scale(
        "Fish5_male_D10reg_scale_no data001__2021_spp1mutant_HomMch_7months_ALP.tif"
    )


def test_df():
    """
    Check we build up a dataframe of metadata correctly
    """
    name = "Fish1_male_Onto_3.2X_scale006__3year4months_ALP_segmentation.tif"
    expected = pd.DataFrame(
        {
            "path": [name],
            "sex": "M",
            "magnification": 3.2,
            "age": 40,
            "stain": "ALP",
            "growth": "onto",
            "no_scale": False,
        }
    )

    assert (metadata.df([name]) == expected).all().all()
