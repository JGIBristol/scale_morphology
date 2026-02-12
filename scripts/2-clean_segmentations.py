"""
Clean the segmentations.

This will open a GUI for editing the masks:
    - pressing "s" will Save the new mask in a segmentations_cleaned/ directory on the RDSF
    - pressing "n" will save and advance to the Next image/mask
    - pressing "Shift-p" will go back to the Previous image/mask

"""

import sys
import pathlib
import argparse

import napari
import tifffile
import numpy as np


def main(*, img_dir: pathlib.Path, mask_dir: pathlib.Path, output_dir: pathlib.Path):
    """
    Get the lists of masks and images and open a GUI to edit them
    """
    assert img_dir.is_dir(), f"{img_dir} does not exist"
    assert mask_dir.is_dir(), f"{mask_dir} does not exist"

    try:
        output_dir.mkdir()
    except FileExistsError as e:
        print(f"Please move or delete `{output_dir}`", file=sys.stderr)
        print("="*79)
        raise e

    img_paths = sorted(list(img_dir.glob("*.tif")))
    mask_paths = sorted(list(mask_dir.glob("*.tif")))
    assert len(img_paths) == len(
        mask_paths
    ), f"Got different numbers of imgs ({len(img_paths)}) and masks ({len(mask_paths)})"

    # Check that all the images and masks match up
    for img_path, mask_path in zip(img_paths, mask_paths):
        assert (
            img_path.name == mask_path.name.replace("_segmentation", "")
        ), f"Got img-mask mismatch: {img_path.name} vs {mask_path.name}"

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
        out_path = output_dir / name
        tifffile.imwrite(
            pathlib.Path(str(out_path).replace(",", "")),
            (state["labels"].data > 0).astype(np.uint8) * 255,
        )
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

    @viewer.bind_key("Shift-p", overwrite=True)
    def _prev(v):
        save_current()
        if state["i"] > 0:
            state["i"] -= 1
            load_index(state["i"])

    load_index(state["i"])
    napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            )

    parser.add_argument(
        "--img_dir",
        type=pathlib.Path,
        help="Directory containing scale images. Must be tif",
        required=True,
    )
    parser.add_argument(
        "--mask_dir",
        type=pathlib.Path,
        help="Directory containing scale images. Must be tif",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Where to save cleaned segmentation masks.",
        required=True,
    )

    main(**vars(parser.parse_args()))
