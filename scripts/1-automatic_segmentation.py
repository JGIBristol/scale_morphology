"""
Segment some scales out from microscopy images.

Intended for use with ALP stained scales, but may also work with confocal scales.
ALP scales are segmented by first performing a rough segmentation and using this as a prior for the model;
confocal scales are segmented by taking some points near the centre of the images and using that as the prior.

Confocal segmentation is less tested than ALP, so might not work right.
"""

import pathlib
import argparse
import requests
import tifffile

import cv2
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from segment_anything import sam_model_registry, SamPredictor

from scale_morphology.scales import segmentation


def _get_scales(scale_dir: pathlib.Path) -> list[pathlib.Path]:
    """
    Get a list of scale paths from a directory, assuming they're all TIFs.
    """
    return sorted(list(scale_dir.glob("*.tif")))


def _get_out_paths(
    out_dir: pathlib.Path, in_paths: list[pathlib.Path]
) -> list[pathlib.Path]:
    """
    Get the list of paths to save the segmentations to
    """
    return [out_dir / (name.stem + "_segmentation.tif") for name in in_paths]


def _get_model(device: str) -> torch.nn.Module:
    """
    Get the model used for segmentation, caching it if necessary
    """
    # Use this URL. change it if you want
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    sam_dir = pathlib.Path("segmentation_checkpoints/")
    sam_dir.mkdir(exist_ok=True)
    sam_path = sam_dir / "sam_vit_h_4b8939.pth"

    if not sam_path.is_file():
        with open(sam_path, "wb") as f:
            f.write(requests.get(url).content)

    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device).eval()

    return SamPredictor(sam)


def _alp_prior(img) -> dict:
    """
    ALP prior - bounding box and point labels
    """
    rough_segmentation = segmentation.rough_segment_alp(img)

    pts = segmentation.sample_pos_points(rough_segmentation)
    box = segmentation.bbox_from_mask(rough_segmentation)

    return {
        "point_coords": pts,
        "point_labels": np.ones((pts.shape[0],), dtype=np.int32),
        "box": box,
    }


def _confocal_prior(img_shape) -> dict:
    """
    Confocal prior - points near the centre of the image
    """
    centre = np.array([img_shape[1] // 2, img_shape[0] // 2])

    # How far to move our points
    offset = 500
    points = np.array(
        [
            centre,
            centre + [0, offset],
            centre + [0, -offset],
            centre + [offset, 0],
            centre + [-offset, 0],
        ]
    )

    return {
        "point_coords": points,
        "point_labels": [1 for _ in points],
    }


def _image_prior(img: np.ndarray, prior_type: str) -> dict:
    """
    Get the prior as kwargs suitable for passing into sam model.predict
    """
    if prior_type == "ALP":
        return _alp_prior(img)
    return _confocal_prior(img.shape)


def _plot_prior(img: np.ndarray, prior: dict, path: pathlib.Path):
    """
    Show the prior (bounding box + points or points)

    """
    fig, axis = plt.subplots()
    axis.imshow(img, cmap="grey")

    # Plot the points
    points = prior["point_coords"]
    axis.scatter(points[:, 0], points[:, 1], c="red", s=50, marker="o")

    # For ALP segmentation, we also have a bounding box prior
    if "box" in prior:
        box = prior["box"]
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
        axis.add_patch(rect)

    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(
    *,
    scale_dir: pathlib.Path,
    out_dir: pathlib.Path,
    prior_type: str,
    debug_plot_dir: pathlib.Path | None,
    device: str,
):
    """
    Read in the scales, find a "prior" for the segmentation and then feed
    the images and prior into an AI model for segmentation.

    The prior is the "dark magic" part of this script - it is relatively simple
    to run the AI model on an image, but finding a prior that roughly segments
    the scale out is a bit more difficult. The ones here are priors that I found
    work, approximately at least, for the scales we have, but if you take new images
    you might need to make your own prior function.

    Caches the segmentation model to disk.
    Saves the segmentations to the specified directory.

    """
    scale_dir = scale_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    assert scale_dir.is_dir(), f"{scale_dir} is not a directory"

    out_dir.mkdir(exist_ok=True, parents=True)

    # Read the scale images in and get the paths to save them to
    scale_img_paths = _get_scales(scale_dir)
    out_paths = _get_out_paths(out_dir, scale_img_paths)

    # Get segmentation model
    model = _get_model(device)

    # Set up dirs for debug plots
    make_debug_plots = debug_plot_dir is not None
    if make_debug_plots:
        preprocessed_dir = debug_plot_dir / "preprocessed"
        preprocessed_dir.mkdir(exist_ok=True, parents=True)

        prior_dir = debug_plot_dir / "prior"
        prior_dir.mkdir(exist_ok=True, parents=True)

    is_alp = prior_type == "ALP"
    for in_path, out_path in zip(tqdm(scale_img_paths), out_paths, strict=True):
        img = tifffile.imread(in_path)
        if is_alp:
            img = rgb2gray(img)

        # Preprocess image
        preprocessed = segmentation.preprocess_greyscale(img)
        if make_debug_plots:
            fig, axis = plt.subplots()
            axis.imshow(preprocessed, cmap="grey")
            axis.set_axis_off()
            fig.tight_layout()
            fig.savefig(preprocessed_dir / (in_path.with_suffix(".png")).name)
            plt.close(fig)

        # Find prior - either points for confocal or points and bbox for ALP
        prior_kw = _image_prior(img, prior_type)
        if make_debug_plots:
            _plot_prior(
                preprocessed, prior_kw, prior_dir / (in_path.with_suffix(".png")).name
            )

        # Run segmentation model
        # Turns the greyscale image back to RGB, since this is what SAM expects
        # It probably would have been better to not do this, since the SAM model can presumably
        # use the colour channels to pick out the dyed scale, but I started with the confocal
        # scales which are greyscale and i cba to change it now
        model.set_image(cv2.cvtColor(img.astype(np.float32), cv2.COLOR_GRAY2RGB))

        masks, _, _ = model.predict(**prior_kw, multimask_output=False)

        # Save images
        tifffile.imwrite(out_path, masks[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "scale_dir", help="directory of .tif images to segment.", type=pathlib.Path
    )
    parser.add_argument(
        "out_dir", help="Where to save segmentations.", type=pathlib.Path
    )

    parser.add_argument(
        "--prior-type",
        choices={"ALP", "confocal"},
        help="We segment ALP and confocal scales slightly differently - set the type here.",
        default="ALP",
    )
    parser.add_argument(
        "--debug-plot-dir",
        help="If specified, will save additional plots here for debugging purposes",
        type=pathlib.Path,
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Which device to run this code on."
        "Defaults to GPU because it's annoyingly slow on CPU, but you can specify CPU if you really want.",
        choices={"cuda", "cpu"},
        default="cuda",
    )

    main(**vars(parser.parse_args()))
