"""
Train and evaluate an autoencoder model for scale shape

"""

import torch


class Autoencoder(torch.nn.Module):
    """
    Simple autoencoder
    """

    def __init__(self):
        """
        Work out the feature map sizes and find the feature map sizes etc.
        """


def main() -> None:
    """
    Read in the scale segmentations, create a dataloader from them
    and train an autoencoder model then save the model to disk.

    """
