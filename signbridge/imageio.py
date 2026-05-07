"""Shared image-loading helpers.

Centralised so the recognizer, smoke test, gold-set harness, and backend
all behave the same way on alpha-channel images (e.g. SVG-rendered PNGs
with transparent backgrounds — those would otherwise come out solid black
after a naive `.convert("RGB")` and the VLM sees nothing).
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def load_rgb(source: str | Path | bytes | io.IOBase) -> np.ndarray:
    """Load an image as an RGB ndarray, compositing any alpha onto white.

    Also applies EXIF rotation so phone-camera photos arrive at the VLM
    upright. Accepts a filesystem path, raw bytes, or any file-like object
    PIL knows how to open.
    """
    from PIL import Image, ImageOps

    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(source)))
    else:
        img = Image.open(source)

    # Force-load before EXIF transpose so the image is in memory and the
    # source file/buffer can be released. Required because the rest of the
    # pipeline holds the array, not the PIL handle.
    img.load()
    # Honour EXIF orientation. Phone cameras often store landscape rotation
    # in EXIF rather than rotating the pixel data; without this every
    # portrait-mode photo arrives at the VLM rotated 90°/180°/270°.
    img = ImageOps.exif_transpose(img)
    return _composite_to_rgb(img)


def array_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert an arbitrary-shape ndarray (H,W,3 or H,W,4) to RGB on white.

    Used at the recognizer's API boundary in case a caller hands us a
    pre-decoded RGBA array. Float arrays in [0, 1] are scaled to uint8;
    naive `.astype(np.uint8)` would truncate to all-zeros (the same
    black-frame failure mode the alpha fix already eliminated for paths).
    """
    from PIL import Image

    if arr.ndim == 2:
        img = Image.fromarray(_to_uint8(arr)).convert("RGB")
        return np.asarray(img)
    if arr.ndim != 3:
        raise ValueError(f"unsupported array shape for RGB conversion: {arr.shape}")
    if arr.shape[-1] == 3:
        return _to_uint8(arr)
    if arr.shape[-1] == 4:
        img = Image.fromarray(_to_uint8(arr), mode="RGBA")
        return _composite_to_rgb(img)
    raise ValueError(f"unsupported array shape for RGB conversion: {arr.shape}")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce an ndarray to uint8 without truncating float [0, 1] to zero."""
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        # Heuristic: if max is ≤ 1.0, it's a normalised [0, 1] image.
        # Otherwise assume the caller already scaled to 0–255.
        if arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _composite_to_rgb(img) -> np.ndarray:  # noqa: ANN001
    from PIL import Image

    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.getchannel("A") if img.mode == "RGBA" else img.split()[-1]
        bg.paste(img.convert("RGB"), mask=alpha)
        img = bg
    elif img.mode == "P" and "transparency" in img.info:
        # Palette image with transparent index — also composite.
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)
