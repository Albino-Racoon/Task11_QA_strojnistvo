"""Inspection orchestration service."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from api.config import Settings, get_settings
from api.models.inspection import Inspection
from decision_engine.engine import decide
from inference.anomaly.patchcore import PatchCoreAnomalyDetector
from inference.demo_fallback import demo_anomaly_heuristic, demo_defect_heuristic
from inference.detector.yolo_detector import DefectDetector


class ModelRegistry:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        yolo_path = self.settings.yolo_weights
        if self.settings.yolo_msdd_weights.exists():
            yolo_path = self.settings.yolo_msdd_weights
        self.detector = DefectDetector(
            yolo_path,
            conf_threshold=self.settings.yolo_conf_threshold,
        )
        self.anomaly = PatchCoreAnomalyDetector(self.settings.anomaly_weights)

    def reload(self) -> None:
        yolo_path = self.settings.yolo_weights
        if self.settings.yolo_msdd_weights.exists():
            yolo_path = self.settings.yolo_msdd_weights
        self.detector = DefectDetector(
            yolo_path,
            conf_threshold=self.settings.yolo_conf_threshold,
        )
        self.anomaly = PatchCoreAnomalyDetector(self.settings.anomaly_weights)

    def health(self) -> dict:
        return {
            "detector": self.detector.info(),
            "anomaly": self.anomaly.info(),
            "demo_fallback": self.settings.allow_demo_fallback,
        }


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def _public_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    settings = get_settings()
    try:
        rel = path.relative_to(settings.storage_dir)
        return f"/api/files/{rel.as_posix()}"
    except ValueError:
        return str(path)


def run_inspection(
    db: Session,
    image_bgr: np.ndarray,
    *,
    part_id: str | None = None,
    line: str = "demo",
    source_name: str = "upload",
) -> dict:
    settings = get_settings()
    registry = get_registry()
    started = time.perf_counter()

    part_id = part_id or f"P{uuid.uuid4().hex[:8].upper()}"
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    raw_name = f"{stamp}_{part_id}.jpg"
    raw_path = settings.images_dir / raw_name
    cv2.imwrite(str(raw_path), image_bgr)

    defects = registry.detector.predict(image_bgr)
    heatmap_path = settings.heatmaps_dir / f"{stamp}_{part_id}_heat.jpg"
    anomaly = registry.anomaly.predict(image_bgr, heatmap_out=heatmap_path)

    used_fallback = False
    if (
        settings.allow_demo_fallback
        and not registry.detector.available
        and not registry.anomaly.available
    ):
        used_fallback = True
        defects = demo_defect_heuristic(image_bgr)
        anomaly = demo_anomaly_heuristic(image_bgr, heatmap_out=heatmap_path)
    elif settings.allow_demo_fallback and not registry.detector.available and not defects:
        # anomaly only available
        pass
    elif settings.allow_demo_fallback and not registry.anomaly.available:
        anomaly = demo_anomaly_heuristic(image_bgr, heatmap_out=heatmap_path)
        used_fallback = True

    result = decide(
        defects,
        anomaly,
        yolo_conf_threshold=settings.yolo_conf_threshold,
        anomaly_fail_threshold=settings.anomaly_fail_threshold,
        anomaly_review_threshold=settings.anomaly_review_threshold,
    )

    annotated = registry.detector.draw(image_bgr, result.defects) if result.defects else image_bgr.copy()
    # status banner
    color = {"PASS": (40, 180, 40), "FAIL": (40, 40, 220), "REVIEW": (0, 165, 255)}[result.status]
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 36), color, -1)
    cv2.putText(
        annotated,
        f"{result.status} | score {result.quality_score} | {part_id}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    ann_path = settings.images_dir / f"{stamp}_{part_id}_ann.jpg"
    cv2.imwrite(str(ann_path), annotated)

    if result.status == "FAIL":
        def_path = settings.defective_dir / raw_name
        cv2.imwrite(str(def_path), image_bgr)

    latency_ms = (time.perf_counter() - started) * 1000.0
    payload = result.to_dict()
    if payload["anomaly"]["heatmap_url"]:
        payload["anomaly"]["heatmap_url"] = _public_url(payload["anomaly"]["heatmap_url"])

    record = Inspection(
        part_id=part_id,
        status=result.status,
        decision=result.decision,
        quality_score=result.quality_score,
        line=line,
        defects_json=json.dumps(payload["defects"]),
        anomaly_score=result.anomaly.score,
        latency_ms=latency_ms,
        image_path=str(raw_path),
        annotated_path=str(ann_path),
        heatmap_path=str(heatmap_path) if Path(heatmap_path).exists() else "",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "part_id": part_id,
        "line": line,
        "source": source_name,
        "status": result.status,
        "decision": result.decision,
        "quality_score": result.quality_score,
        "reason": result.reason,
        "defects": payload["defects"],
        "anomaly": payload["anomaly"],
        "latency_ms": round(latency_ms, 1),
        "image_url": _public_url(raw_path),
        "annotated_url": _public_url(ann_path),
        "created_at": record.created_at.isoformat() + "Z",
        "demo_fallback": used_fallback,
        "models": {
            "detector_available": registry.detector.available,
            "anomaly_available": registry.anomaly.available,
        },
    }
