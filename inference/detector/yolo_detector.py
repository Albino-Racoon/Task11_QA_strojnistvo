"""YOLO defect detector for GC10-DET classes."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from decision_engine.engine import DefectDetection

logger = logging.getLogger(__name__)

GC10_CLASSES = [
    "punching_hole",
    "welding_line",
    "crescent_gap",
    "water_spot",
    "oil_spot",
    "silk_spot",
    "inclusion",
    "rolled_pit",
    "crease",
    "waist_folding",
]

# MSDD casting-part defects (when using weights/yolo_msdd.pt)
MSDD_CLASSES = [
    "misrun",
    "inclusion",
    "dent",
    "parting_line_crack",
    "stamp_collapse",
    "pockmarks",
    "mould_scuffing",
    "cut_marks",
]

# Friendly SI-ish labels for UI mapping (API keeps EN keys)
CLASS_ALIASES = {
    "punching_hole": "punching_hole",
    "welding_line": "welding_line",
    "crescent_gap": "crescent_gap",
    "water_spot": "water_spot",
    "oil_spot": "oil_spot",
    "silk_spot": "silk_spot",
    "inclusion": "inclusion",
    "rolled_pit": "rolled_pit",
    "crease": "crease",
    "waist_folding": "waist_folding",
    # NEU-style
    "scratch": "scratch",
    "scratches": "scratch",
    "crazing": "crazing",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "rolled-in_scale": "rolled_in_scale",
    # MSDD
    "misrun": "misrun",
    "dent": "dent",
    "parting_line_crack": "parting_line_crack",
    "stamp_collapse": "stamp_collapse",
    "pockmarks": "pockmarks",
    "mould_scuffing": "mould_scuffing",
    "cut_marks": "cut_marks",
}


class DefectDetector:
    def __init__(self, weights_path: Path, conf_threshold: float = 0.35):
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self.available = False
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            logger.warning("YOLO uteži niso na voljo: %s", self.weights_path)
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.weights_path))
            self.available = True
            logger.info("YOLO naložen: %s", self.weights_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Napaka pri nalaganju YOLO: %s", exc)
            self.available = False

    def predict(self, image_bgr: np.ndarray) -> list[DefectDetection]:
        if not self.available or self.model is None:
            return []

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        detections: list[DefectDetection] = []
        h, w = image_bgr.shape[:2]
        image_area = float(h * w)

        for result in results:
            names = result.names or {}
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                area = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
                raw_name = names.get(cls_id, f"class_{cls_id}")
                name = CLASS_ALIASES.get(str(raw_name), str(raw_name))
                detections.append(
                    DefectDetection(
                        type=name,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                        affected_area=area,
                    )
                )
        return detections

    def draw(self, image_bgr: np.ndarray, defects: list[DefectDetection]) -> np.ndarray:
        out = image_bgr.copy()
        for d in defects:
            x1, y1, x2, y2 = map(int, d.bbox)
            color = (40, 40, 220) if d.severity == "high" else (0, 165, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.type} {d.confidence * 100:.0f}%"
            cv2.putText(
                out,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": "yolo_gc10",
            "available": self.available,
            "weights": str(self.weights_path),
            "classes": GC10_CLASSES,
        }
