"""
Segment the scale from an unprocessed image
"""

import sys
import pathlib
import warnings
import torch
from segment_anything import sam_model_registry, SamPredictor
from functools import cache

import cv2
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_mean, threshold_minimum
from skimage.segmentation import clear_border
from skimage.exposure import equalize_adapthist
from skimage.morphology import (
    binary_opening,
    disk,
    erosion,
    reconstruction,
    binary_closing,
)


def _largest_connected_component(binary_array):
    """
    Return the largest connected component of a binary array, as a binary array

    :param binary_array: Binary array.
    :returns: Largest connected component.

    """
    labelled, _ = ndimage.label(binary_array, np.ones((3, 3)))

    # Find the size of each component
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0

    retval = labelled == np.argmax(sizes)
    return retval


def largest_connected_component(binary_array):
    return _largest_connected_component(binary_array)


def preprocess_greyscale(grey_img):
    """
    Preprocess confocal img
    """
    blurred = gaussian(grey_img, sigma=3)
    # Giant kernel seems to work best, since the scale is also very large
    # Unfortunately this does make things slow
    return equalize_adapthist(blurred, kernel_size=2001)


def preprocess_img(img: np.ndarray) -> np.ndarray:
    """
    Blurs + enhances contrast.

    Used as input for both the rough and SAM segmentation.
    """
    grey = rgb2gray(img)
    return preprocess_greyscale(grey)


def _clear_border_keep_large(img):
    """
    Clear border but dont do anything if it would remove too much
    """
    sum_before = np.sum(img)
    cleared = clear_border(img)

    if np.sum(cleared) < 0.1 * sum_before:
        print("large obj touching border", file=sys.stderr)
        return img
    return cleared


@cache
def _elem():
    """structuring element for binary opening"""
    return disk(10)


def rough_segment_alp(enhanced_img: np.ndarray) -> np.ndarray:
    """
    Roughly segment an RGB image of a scale stained with ALP.

    :param enhanced_img: the image to segment, as a 3-channel RGB array.
                         Probably should have a Gaussian blur + contrast enhancement applied.
    :returns: binary segmentation mask

    """
    # Threshold
    try:
        min_th = threshold_minimum(enhanced_img)
    except RuntimeError as e:
        warnings.warn(f"Failed to find min threshold:\n{str(e)}")
        min_th = 0

    mean_th = threshold_mean(enhanced_img)
    thresholded = enhanced_img < max(min_th, mean_th)

    # Clear the border
    cleared = _clear_border_keep_large(thresholded)

    # Perform binary opening
    opened = binary_opening(cleared, _elem())

    # Fill holes
    filled = ndimage.binary_fill_holes(opened)

    # Take the largest connected component
    return _largest_connected_component(filled)


def bbox_from_mask(m, pad=32):
    """
    Get bounding box from a mask
    """
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    return np.array(
        [
            max(0, x0 - pad),
            max(0, y0 - pad),
            min(m.shape[1] - 1, x1 + pad),
            min(m.shape[0] - 1, y1 + pad),
        ],
        dtype=np.int32,
    )


def sample_pos_points(m, n=10):
    """
    Get some positive-labelled points from deep inside a mask
    """
    mm = ndimage.binary_erosion(m, iterations=100)
    ys, xs = np.nonzero(mm if mm.any() else m)

    idx = np.linspace(0, ys.size - 1, num=min(n, ys.size)).astype(int)
    pts = np.stack([xs[idx], ys[idx]], axis=1)
    return pts


@cache
def _sam(model_type: str, model_checkpoint: str, device: str) -> SamPredictor:
    """
    Get the model
    """
    sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
    sam.to(device).eval()

    return SamPredictor(sam)


def sam_segmentation(
    enhanced_img: np.ndarray,
    prior: np.ndarray,
    *,
    device: str,
    model_type: str,
    model_checkpoint: pathlib.Path,
) -> np.ndarray:
    """
    Perform segmentation with Meta's SAM.

    Requies the model weights to be downloaded.

    :param enhanced_img: the image to segment, as a 3-channel RGB array.
                         Probably should have a Gaussian blur + contrast enhancement applied.
    :param prior: a rough segmentation that will be used as the prior for the model.
    :param device: "cuda" or "cpu"
    :param model_type: e.g. "vit_h"
    :param model_checkpoint: path to the downloaded weights.

    :returns: binary segmentation mask
    """
    pts = sample_pos_points(prior)
    box = bbox_from_mask(prior)

    # Turn the greyscale image back to RGB i guess
    grey = cv2.cvtColor(enhanced_img.astype(np.float32), cv2.COLOR_GRAY2RGB)

    model = _sam(model_type, str(model_checkpoint), device)
    model.set_image(grey)

    masks, scores, _ = model.predict(
        point_coords=pts,
        point_labels=np.ones((pts.shape[0],), dtype=np.int32),
        box=box,
        multimask_output=True,
    )

    return masks[np.argmax(scores)]


def segment_alp(
    img: np.ndarray,
    *,
    device: str,
    model_type: str,
    model_checkpoint: pathlib.Path,
) -> np.ndarray:
    """
    Enhancement, rough segmentation then SAM segmentation.
    """
    enhanced = preprocess_img(img)

    return sam_segmentation(
        enhanced,
        rough_segment_alp(enhanced),
        device=device,
        model_type=model_type,
        model_checkpoint=model_checkpoint,
    )
