"""
The best way to visualise the PCA will be to have a simple interactive dashboard,
so that we can see the shape of the scale and how it relates to the points in the
reduced-dimension space.

"""

import re
import sys
import base64
import pathlib
from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
from scipy.spatial import ConvexHull

from bokeh.plotting import figure, save
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    CategoricalColorMapper,
    GlyphRenderer,
)
from bokeh.resources import INLINE

from scale_morphology.scales import errors, read


class HullError(Exception):
    """Can't draw a convex hull around these points"""


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


def _dashboard_df(
    coeffs: np.ndarray,
    images: np.ndarray,
    names: list[str],
    colour_coding: np.ndarray,
) -> pd.DataFrame:
    """
    Build a dataframe holding the information we need to create the dashboard.
    This includes the dimension-reduced co-ords, the original image, and the
    filename.

    """
    # Convert images to strings
    images = [embeddable_image(image) for image in images]

    # Build the dataframe
    df = pd.DataFrame(coeffs, columns=["x", "y"])

    df["image"] = images
    df["names"] = names
    df["colour"] = colour_coding

    return df


def _dashboard_figure(
    df, colour_mapper, colour_coding, title
) -> tuple[figure, GlyphRenderer]:
    """
    Create the dashboard figure - plot the points on it
    """
    # Create a mapping for colours
    unique_colours = np.unique(colour_coding)

    # Create the figure
    datasource = ColumnDataSource(df)
    fig = figure(
        title=title,
        width=800,
        height=800,
        tools="pan, wheel_zoom, box_zoom, reset",
    )

    # Plot the scatter points
    point_renderer = fig.scatter(
        x="x",
        y="y",
        source=datasource,
        size=4,
        color={"field": "colour", "transform": colour_mapper},
        legend_field="colour",
    )

    return fig, point_renderer


def _plot_hull(
    fig: figure,
    df: pd.DataFrame,
    colour_value: str,
    mapper: CategoricalColorMapper,
    i: int,
):
    """
    Plot a convex hull around points of the given colour

    :raises HullError: if we don't have enough points to plot a hull
    """
    group_points = df[df["colour"] == colour_value][["x", "y"]].values

    # Need at least 3 points for a convex hull
    if len(group_points) < 3:
        raise HullError(
            f"Not enough points to create a convex hull for {colour_value} (got {len(group_points)})"
        )

    hull = ConvexHull(group_points)
    vertices = group_points[hull.vertices]

    # Close the polygon by adding the first point at the end
    vertices = np.vstack([vertices, vertices[0]])

    hull_color = mapper.palette[i % len(mapper.palette)]
    fig.patch(
        x=vertices[:, 0],
        y=vertices[:, 1],
        alpha=0.2,
        line_color=hull_color,
        line_width=2,
        fill_color=hull_color,
    )


def write_dashboard(
    reduced_coeffs: np.ndarray,
    images: np.ndarray,
    colour_coding: np.ndarray,
    names: list[str],
    filename: str | pathlib.Path,
    title: str,
) -> None:
    """
    Create a dashboard to visualise a 2d projection of some co-ordinates.

    Basically just a colour-coded 2d scatter plot, showing the point corresponding to each image
    when you hover over it.

    :param coeffs: the PCA coefficients
    :param: images: images to embed in the dashboard
    :param filename: path to save the HTML dashboard
    :param colour_coding: array of unique labels for each datapoint
    :param names: names of each scale, to be displayed in the dashboard
    :param title: plot title

    """
    if not reduced_coeffs.shape[1] == 2:
        raise ValueError("Dim reduced co-ords should be 2D")
    if len(colour_coding) != reduced_coeffs.shape[0]:
        raise ValueError(
            f"coeffs + labels should be same length; got {reduced_coeffs.shape=} and {len(colour_coding)=}"
        )

    # Build the dataframe
    df = _dashboard_df(reduced_coeffs, images, names, colour_coding)

    # minimum number of colours in our cmap is 3
    unique_colours = np.unique(colour_coding)
    mapper = CategoricalColorMapper(
        factors=unique_colours,
        palette=(f"Category10_{max(len(unique_colours), 3)}"),
    )

    # Plot the figure
    fig, point_renderer = _dashboard_figure(df, mapper, colour_coding, title)

    # Enable showing the images on hover
    fig.add_tools(
        (
            HoverTool(
                renderers=[point_renderer],
                tooltips="""
<div>
    <div>
        <img src="@image" style="float: left; margin: 5px 5px 5px 5px;">
    </div>
    <div>
        <span style="font size: 14px; font-weight: bold;">@names</span>
    </div>
</div>
""",
            )
        )
    )

    # For each color group, draw a convex hull
    for i, colour_value in enumerate(unique_colours):
        try:
            _plot_hull(fig, df, colour_value, mapper, i)
        except HullError:
            pass

    save(
        fig,
        filename=filename,
        title=filename,
        resources=INLINE,
    )
