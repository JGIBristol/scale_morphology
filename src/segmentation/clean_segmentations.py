# /// script
# requires-python = "==3.12.2"
# dependencies = ["napari[pyqt5]", "tifffile", "numpy", "pyyaml"]
# ///
"""
This will open a GUI for editing the masks:
- pressing "s" will Save the new mask in a segmentations_cleaned/ directory on the RDSF
- pressing "n" will save and advance to the Next image/mask
- pressing "b" will go Back to the previous image/mask

"""

import pathlib

import yaml
import napari
import tifffile
import numpy as np

cfg_path = pathlib.Path(__file__).resolve().parent / "config.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# The image dir contains lots of subfolders that each contain images
# The segmentation names are based on these but dont have the same directory structure
# So, we'll read in all the image paths and all the segmentation paths
# And match them up together by seeing which segmentation filename has the same ending as
# each image filename.

img_dir = pathlib.Path(cfg["input_tif_dir"]).expanduser()
mask_dir = pathlib.Path(cfg["auto_segmentation_dir"]).expanduser()
assert img_dir.is_dir()
assert mask_dir.is_dir()

cleaned_mask_dir = img_dir.parent / "segmentations_cleaned"
cleaned_mask_dir.mkdir(exist_ok=True)

all_img_dirs = [d for d in img_dir.glob("**/*") if d.is_dir()]
img_paths = []
for d in all_img_dirs:
    if "D21Reg" not in str(d):
        img_paths += list(d.glob("*.tif"))


all_mask_paths = list(mask_dir.glob("*.tif"))
assert len(all_mask_paths) == len(img_paths), f"{len(all_mask_paths)=}, {len(img_paths)=}"


# Get the corresponding masks
def get_mask(img_path: pathlib.Path, l: list[pathlib.Path]):
    """Get the name of the mask given an image"""
    matching_masks = [m for m in l if img_path.name in m.name]

    if len(matching_masks) != 1:
        print(f"{len(matching_masks)=}:\n\t{matching_masks}")
    (retval,) = matching_masks

    return retval


mask_paths = [get_mask(i, all_mask_paths) for i in img_paths]

# Which image to start at
start = 0


def load_index(i):
    name = mask_paths[i].name
    im = tifffile.imread(img_paths[i])
    mask = tifffile.imread(mask_paths[i]).astype(np.uint8)

    if state["labels"] is None:
        state["image"] = viewer.add_image(im, name="image")
        state["labels"] = viewer.add_labels(mask, name="mask", opacity=0.5)
    else:
        state["image"].data = im
        state["labels"].data = mask
    viewer.title = f"{i+1}/{len(mask_paths)} : {name}"


def save_current():
    name = mask_paths[state["i"]].name
    out_path = cleaned_mask_dir / name
    tifffile.imwrite(out_path, (state["labels"].data > 0).astype(np.uint8) * 255)
    print(f"Saved {out_path}")


state = {"i": start, "viewer": None, "labels": None}
viewer = napari.Viewer()
state["viewer"] = viewer


@viewer.bind_key("s")
def _save(v):
    save_current()


@viewer.bind_key("n")
def _next(v):
    save_current()
    if state["i"] < len(mask_paths) - 1:
        state["i"] += 1
        load_index(state["i"])


@viewer.bind_key("b")
def _prev(v):
    save_current()
    if state["i"] > 0:
        state["i"] -= 1
        load_index(state["i"])


load_index(state["i"])
