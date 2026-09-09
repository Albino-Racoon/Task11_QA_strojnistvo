"""Heuristic demo fallback when trained weights are missing."""
from __future__ import annotations

import cv2
import numpy as np

from decision_engine.engine import AnomalyResult, DefectDetection


def demo_defect_heuristic(image_bgr: np.ndarray) -> list[DefectDetection]:
    """Detect dark elongated regions as pseudo-scratches for UI demos without weights."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    detections: list[DefectDetection] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if area < (h * w) * 0.002 or area > (h * w) * 0.25:
            continue
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect < 2.5:
            continue
        conf = float(np.clip(0.45 + min(aspect, 8) * 0.05, 0.45, 0.92))
        detections.append(
            DefectDetection(
                type="scratch",
                confidence=conf,
                bbox=[float(x), float(y), float(x + bw), float(y + bh)],
                affected_area=area / float(h * w),
            )
        )
        if len(detections) >= 3:
            break
    return detections


def demo_anomaly_heuristic(image_bgr: np.ndarray, heatmap_out=None) -> AnomalyResult:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    energy = float(np.var(lap))
    # normalize roughly: metal textures vary; map into 0..1
    score = float(np.clip((energy - 50) / 400.0, 0.0, 1.0))
    heatmap_path = None
    if heatmap_out is not None:
        abs_lap = np.abs(lap)
        abs_lap = abs_lap / (abs_lap.max() + 1e-6)
        heat = (abs_lap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.55, colored, 0.45, 0)
        heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(heatmap_out), overlay)
        heatmap_path = str(heatmap_out)
    return AnomalyResult(score=score, heatmap_path=heatmap_path, is_anomaly=score >= 0.4)
