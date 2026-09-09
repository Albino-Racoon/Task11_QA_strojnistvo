#!/usr/bin/env python3
"""Evaluate detector + anomaly metrics and latency."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.config import get_settings
from decision_engine.engine import decide
from inference.anomaly.patchcore import PatchCoreAnomalyDetector
from inference.detector.yolo_detector import DefectDetector
from training.train_anomaly import _kolektor_split


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.array(vals, dtype=np.float64), p))


def eval_latency(detector: DefectDetector, anomaly: PatchCoreAnomalyDetector, images: list[Path]) -> dict:
    times = []
    for p in images[:40]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        t0 = time.perf_counter()
        detector.predict(img)
        anomaly.predict(img)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "n": len(times),
        "p50_ms": round(percentile(times, 50), 1),
        "p95_ms": round(percentile(times, 95), 1),
        "mean_ms": round(float(np.mean(times)) if times else 0.0, 1),
    }


def eval_anomaly_binary(anomaly: PatchCoreAnomalyDetector, ok_imgs: list[Path], nok_imgs: list[Path]) -> dict:
    y_true = []
    y_score = []
    for p in ok_imgs[:80]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        y_true.append(0)
        y_score.append(anomaly.predict(img).score)
    for p in nok_imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        y_true.append(1)
        y_score.append(anomaly.predict(img).score)

    if not y_true:
        return {"error": "ni slik"}

    settings = get_settings()
    thr = settings.anomaly_fail_threshold
    preds = [1 if s >= thr else 0 for s in y_score]
    tp = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, preds) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, preds) if yt == 1 and yp == 0)
    defect_recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_reject = fp / (fp + tn) if (fp + tn) else 0.0
    false_accept = fn / (fn + tp) if (fn + tp) else 0.0

    try:
        from sklearn.metrics import roc_auc_score

        auroc = float(roc_auc_score(y_true, y_score))
    except Exception:  # noqa: BLE001
        auroc = None

    ok_scores = [s for s, y in zip(y_score, y_true) if y == 0]
    nok_scores = [s for s, y in zip(y_score, y_true) if y == 1]

    return {
        "threshold": thr,
        "n_ok": len(ok_scores),
        "n_nok": len(nok_scores),
        "mean_score_ok": round(float(np.mean(ok_scores)), 4) if ok_scores else None,
        "mean_score_nok": round(float(np.mean(nok_scores)), 4) if nok_scores else None,
        "defect_recall": round(defect_recall, 4),
        "false_reject_rate": round(false_reject, 4),
        "false_accept_rate": round(false_accept, 4),
        "auroc": None if auroc is None else round(auroc, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def eval_yolo_map(weights: Path, data_yaml: Path) -> dict:
    if not weights.exists() or not data_yaml.exists():
        return {"skipped": True, "reason": "manjkajo weights ali data.yaml"}
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        metrics = model.val(data=str(data_yaml), verbose=False)
        return {
            "mAP50": round(float(metrics.box.map50), 4),
            "mAP50_95": round(float(metrics.box.map), 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "training" / "eval_report.json"))
    args = parser.parse_args()

    settings = get_settings()
    detector = DefectDetector(settings.yolo_weights)
    anomaly = PatchCoreAnomalyDetector(settings.anomaly_weights)

    ok, nok = _kolektor_split(ROOT / "training" / "datasets" / "kolektor")
    sample_imgs = (ok[:20] + nok[:20]) if ok else []

    report = {
        "detector_available": detector.available,
        "anomaly_available": anomaly.available,
        "latency": eval_latency(detector, anomaly, sample_imgs) if sample_imgs else {},
        "anomaly_binary": eval_anomaly_binary(anomaly, ok, nok)
        if anomaly.available and ok
        else {"skipped": True},
        "yolo_map": eval_yolo_map(
            settings.yolo_weights,
            ROOT / "training" / "datasets" / "gc10_yolo" / "data.yaml",
        ),
        "decision_smoke": None,
    }

    if sample_imgs:
        img = cv2.imread(str(nok[0] if nok else sample_imgs[0]))
        defects = detector.predict(img)
        an = anomaly.predict(img)
        decision = decide(defects, an)
        report["decision_smoke"] = decision.to_dict()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
