# Copyright (c) 2026 EarthBridge Team.
# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Image preprocessing utilities.

Ported from ``vendor/Img2Image-Turbo/src/image_prep.py``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def canny_from_pil(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    """Compute a Canny edge map from a PIL image and return as a 3-channel PIL image."""
    import cv2

    image_arr = np.array(image)
    edges = cv2.Canny(image_arr, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    return Image.fromarray(edges)
