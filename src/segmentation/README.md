Segmentation
----
We do the segmentation in two steps:
1. Automated
2. Manual pass

### Step 1, Automated: `segmentation.ipynb`
This is the first pass of the segmentation pipeline.

> [!TIP]
> You likely won't need to run this notebook, since it writes the segmentations directly
> to the RDSF.
> It currently only runs on the ALP stained scales; if you want to do the others, you will
> need to edit the notebook.

We use the meta Segment Anything Model (SAM) to roughly segment out the scale - this
is much faster than doing them manually, and it usually gets them reasonably correct.
Since we only care about the shape of the scale here, this is usually good enough for us.

It works by roughly thresholding the scale out and using this as a prior for the model.
More specifically, we threshold out the scale, clear artifacts from the image's edge
(since the scale shouldn't touch the edge of the image), remove small noise with a
binary opening, fill holes and then take the largest remaining object in the image.
This is usually (roughly) the scale, so we can use it as a prior for the SAM model.

The SAM model can take a bounding box or some known positive points as a prior.
We can find both of these using our image prior.
The bounding box can be found directly from our roughly segmented scale.
We find the points in interior of the scale by performing a lot of successive binary erosions
on our prior image and taking (some of) the remaining points.

> [!WARNING]
> Finding the prior segmentation (the rough threshold) works well for the ALP scales,
> but might need to be tuned more if you want to segment the other images, since they have
> less contrast with the background.

However, sometimes it doesn't work, so we also want to do a manual pass over the scales.


### Step 2, Manual: `clean_segmentations.py`
Sometimes the automated segmentation will pick up e.g. a bubble, some dirt, some floating dye
as well as/instead of the scale.
To correct this, we have to look through all the scales + segmentations manually.
This is a relatively slow process, but it goes relatively quickly by using the GUI in the
provided script.

Detailed instructions on how to use this cleaning script are provided in [how_to_clean_segs.md](../../docs/how_to_clean_segs.md).

This will open a user interface which will allow you to add/remove from the segmentation masks.
Most of them will require no action, from our experience with the non-SOST ALP scales roughly
~10% of the segmentations required some manual correction (usually small).
