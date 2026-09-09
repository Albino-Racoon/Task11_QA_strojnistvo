"""Camera capture helper."""
from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def capture_frame(device_index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Kamera ni dosegljiva (index={device_index}). "
            "Preverite USB kamero ali uporabite upload slike."
        )
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Zajem s kamere ni uspel.")
    return frame


def list_cameras(max_index: int = 5) -> list[dict]:
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({"index": i, "available": True})
            cap.release()
    return found
