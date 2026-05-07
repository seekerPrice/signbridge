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

    Accepts a filesystem path, raw bytes, or any file-like object PIL
    knows how to open.
    """
    from PIL import Image

    if isinstance(source, (str, Path)):
        img = Image.open(source)
    elif isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(source)))
    else:
        img = Image.open(source)

    return _composite_to_rgb(img)


def array_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert an arbitrary-shape ndarray (H,W,3 or H,W,4) to RGB on white.

    Used at the recognizer's API boundary in case a caller hands us a
    pre-decoded RGBA array.
    """
    from PIL import Image

    if arr.ndim == 2:
        img = Image.fromarray(arr).convert("RGB")
        return np.asarray(img)
    if arr.shape[-1] == 3:
        return arr if arr.dtype == np.uint8 else arr.astype(np.uint8)
    if arr.shape[-1] == 4:
        img = Image.fromarray(arr, mode="RGBA")
        return _composite_to_rgb(img)
    raise ValueError(f"unsupported array shape for RGB conversion: {arr.shape}")


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
