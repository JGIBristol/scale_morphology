"""
The best way to visualise the PCA will be to have a simple interactive dashboard,
so that we can see the shape of the scale and how it relates to the points in the
reduced-dimension space.

"""

import base64
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.transform import resize

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
