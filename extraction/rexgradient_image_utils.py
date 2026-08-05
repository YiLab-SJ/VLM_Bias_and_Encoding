"""
rexgradient_image_utils.py

Utility to properly load RexGradient 16-bit PNG images.
RexGradient images are I;16 mode (16-bit unsigned integer grayscale).
Direct PIL .convert('RGB') produces washed-out images because it divides by 256
without windowing. This utility normalizes to full 8-bit dynamic range first.
"""

import numpy as np
from PIL import Image


def load_rexgradient_image(path):
    """
    Load a RexGradient image and return a properly normalized PIL Image in 'L' mode.
    Handles both 16-bit (I;16) and 8-bit (L/RGB) images correctly.
    """
    img = Image.open(path)
    if img.mode in ('I;16', 'I'):
        arr = np.array(img, dtype=np.float32)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    elif img.mode not in ('L', 'RGB'):
        img = img.convert('L')
    return img
