"""
The best way to visualise the PCA will be to have a simple interactive dashboard,
so that we can see the shape of the scale and how it relates to the points in the
reduced-dimension space.

"""

import base64
import pathlib
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

from bokeh.plotting import figure, save
from bokeh.models import ColumnDataSource, HoverTool

from scale_morphology.scales import errors


def embeddable_image(image: np.ndarray, *, thumbnail_size: int = (64, 64)) -> str:
    """
    Downsample an image and encode it as a base64 string so that it will
    be embeddable in a webpage.

    :param image: the greycale image
    :param thumbnail_size: size of the embedded images

    """
    errors.check_binary_img(image)

    resized = resize(image, thumbnail_size, anti_aliasing=False)

    buffered = BytesIO()
    Image.fromarray(resized).save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"


def dashboard_df(coeffs: np.ndarray) -> pd.DataFrame:
    """
    Build a dataframe holding the information we need to create the dashboard.
    This includes the dimension-reduced co-ords, the original image, and the
    filename.

    """
    # Check the coeffs are 2d
    # Get greyscale image paths
    # Check that we have the right number of coeffs
    # Get the image names
    # Convert images to strings
    # Build the dataframe


def dashboard(coeffs: np.ndarray, filename: str | pathlib.Path) -> None:
    """
    Create a dashboard to visualise the PCA of the EFA coefficients

    :param coeffs: the PCA coefficients
    :param filename: the HTML file to save the dashboard

    """
    if isinstance(filename, pathlib.Path):
        filename = str(filename)
    if not filename.endswith(".html"):
        filename = filename + ".html"

    # Build the dataframe
    df = dashboard_df(coeffs)

    # Create the figure
    datasource = ColumnDataSource(df)
    fig = figure(
        title="Dimension-reduced Scale Dataset",
        plot_width=800,
        plot_height=800,
        tools="pan, wheel_zoom, box_zoom, reset",
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

    fig.scatter(
        x="x",
        y="y",
        source=datasource,
        size=4,
        color="black",
    )

    save(fig, filename=filename)
