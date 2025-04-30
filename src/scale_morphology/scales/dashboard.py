"""
The best way to visualise the PCA will be to have a simple interactive dashboard,
so that we can see the shape of the scale and how it relates to the points in the
reduced-dimension space.

"""

import re
import base64
import pathlib
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.resources import INLINE

from scale_morphology.scales import errors, read


def embeddable_image(image: np.ndarray, *, thumbnail_size: int = (64, 64)) -> str:
    """
    Downsample an image and encode it as a base64 string so that it will
    be embeddable in a webpage.

    :param image: the greycale image
    :param thumbnail_size: size of the embedded images

    """
    errors.check_binary_img(image)

    resized = 255 * resize(image, thumbnail_size, anti_aliasing=False)

    buffered = BytesIO()
    img = Image.fromarray(resized)
    img = img.convert("RGB")
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def extract_mutation(name: str):
    """
    Extract mutation type from filename using regex pattern matching
    """
    pattern = r"^((?:SPP1|SOST)_(?:HOM|WT(?:\sSIB)?))|(?:(OSX)[_-](m[CX]H))"

    match = re.match(pattern, name)
    if match:
        if match.group(1):  # SPP1_HOM or SOST_HOM or SOST_WT
            return match.group(1)
        else:  # OSX marker type
            return f"{match.group(2)}_{match.group(3)}"

    raise ValueError(f"No mutation found in {name}")


def dashboard_df(
    coeffs: np.ndarray,
    *,
    progress: bool,
    drop: np.ndarray,
    colour_coding: str | None = None,
) -> pd.DataFrame:
    """
    Build a dataframe holding the information we need to create the dashboard.
    This includes the dimension-reduced co-ords, the original image, and the
    filename.

    """
    # Get greyscale image paths
    # Convert to a np array so we can use the mask for indexing
    paths = np.array(read.greyscale_paths())[~drop]

    # Check that we have the right number of coeffs
    if len(coeffs) != len(paths):
        raise ValueError(
            "Number of images and PCA coefficients don't match:"
            f"{len(coeffs)} vs {len(paths)}"
        )

    # Get the image names
    names = [path.name.replace("_rois.tif", "") for path in paths]

    # Create a boolean array based on some criteria
    if colour_coding:
        match colour_coding:
            case "regen":
                colour = ["REGEN" if "REGEN" in name else "not regen" for name in names]
            case "mutation":
                colour = [extract_mutation(name) for name in names]
            case _:
                raise ValueError(f"Unknown colour coding criterion: {colour_coding}")
    else:
        # Just make everything the same colour
        colour = ["" for _ in names]

    # Convert images to strings
    images = [
        embeddable_image(image)
        for image in np.array(read.greyscale_images(progress=progress))[~drop]
    ]

    # Build the dataframe
    df = pd.DataFrame(coeffs, columns=["x", "y"])

    df["image"] = images
    df["filename"] = names
    df["colour"] = colour

    return df


def write_dashboard(
    coeffs: np.ndarray,
    filename: str | pathlib.Path,
    *,
    colour_coding: str | None,
    progress: bool = False,
    drop: np.ndarray | None = None,
    **fig_kw,
) -> None:
    """
    Create a dashboard to visualise the PCA of the EFA coefficients

    :param coeffs: the PCA coefficients
    :param filename: the HTML file to save the dashboard
    :param colour_coding: criterion to use for colour coding, if specified
    :param progress: whether to show progress bars
    :param drop: 1d N-length boolean mask of scales to exclude from the dashboard

    """
    if not coeffs.shape[1] == 2:
        raise ValueError("Dim reduced co-ords should be 2D")
    if drop is None:
        drop = np.zeros(coeffs.shape[0], dtype=bool)

    if isinstance(filename, pathlib.Path):
        filename = str(filename)
    if not filename.endswith(".html"):
        filename = filename + ".html"

    # Build the dataframe
    df = dashboard_df(coeffs, progress=progress, drop=drop, colour_coding=colour_coding)

    # Create a mapping for colours
    factors = np.unique(df["colour"])
    mapper = CategoricalColorMapper(
        factors=factors,
        # minimum number of colours in our cmap is 3
        palette=f"Category10_{max(len(factors), 3)}" if colour_coding else ["#000000"],
    )

    # Create the figure
    datasource = ColumnDataSource(df)
    fig = figure(
        title="Dimension-reduced Scale Dataset",
        width=800,
        height=800,
        tools="pan, wheel_zoom, box_zoom, reset",
        **fig_kw,
    )

    fig.add_tools(
        (
            HoverTool(
                tooltips="""
<div>
    <div>
        <img src="@image" style="float: left; margin: 5px 5px 5px 5px;">
    </div>
    <div>
        <span style="font-size: 17px; font-weight: bold;">@filename</span>
    </div>
</div>
"""
            )
        )
    )

    kw = {}
    if colour_coding:
        kw["legend_field"] = "colour"
    fig.scatter(
        x="x",
        y="y",
        source=datasource,
        size=4,
        color={"field": "colour", "transform": mapper},
        **kw,
    )

    save(
        fig,
        filename=filename,
        title=pathlib.Path(filename.replace(".html", "")).name,
        resources=INLINE,
    )
