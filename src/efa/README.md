EFA
====

Once we have segmented out some scales, we can run EFA on them to analyse their shapes.

The pipeline looks like this:

1. Segment out the scales (this should already be done)
2. Run EFA on the scales
3. Run PCA on the EFA coefficients
4. Run LDA on the dimension-reduced coefficients
5. Interpret the LDA axes

It's a bit fiddly - we need to perform the EFA correctly (choose the right number of points around the shape, and the right number of EFA
harmonics), and
we need to find the right number of PCs to keep (using lots keeps the most information but also keeps a lot of noise).

There's two files here:
- `verbose_efa_analysis.ipynb`: contains lots of extra steps and explanations as to what's going on
- `simple_efa_analysis.ipynb`: just does the EFA/PCA/LDA and makes some plots.

#### Future directions?
It's possible that the interesting shape variation might be hidden in the high-frequency harmonics that are thrown away by the PCA.
This has been addressed in this paper: https://arxiv.org/html/2507.01009v1#S3, where they basically train a model on the distance
transform of an image.
This will be a lot of work but might be worth trying, as a better way to parameterise our shapes.

It is also possible that we might want to use a different classifier (a random forest, or a neural network...) to separate out the scales
by shape - either directly from the images, or from the EFA coefficients, or from something else.
