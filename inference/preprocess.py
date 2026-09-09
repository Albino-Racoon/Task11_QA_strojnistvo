"""Image preprocessing helpers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Ne morem prebrati slike: {path}")
    return image


def load_rgb(path: str | Path) -> np.ndarray:
    return cv2.cvtColor(load_bgr(path), cv2.COLOR_BGR2RGB)


def normalize_for_model(
    image_bgr: np.ndarray,
    size: int = 640,
) -> np.ndarray:
    """Letterbox-like resize preserving aspect ratio."""
    h, w = image_bgr.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        # assume BGR from OpenCV callers that pass BGR
        pass
    return image.astype(np.uint8)


def save_image(path: str | Path, image_bgr: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image_bgr)
    return str(path)


def pil_from_bgr(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
