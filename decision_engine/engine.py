"""Dual-model decision engine: known defects (YOLO) + unknown anomalies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefectDetection:
    type: str
    confidence: float
    bbox: list[float]  # x1, y1, x2, y2
    severity: str = "medium"
    affected_area: float = 0.0


@dataclass
class AnomalyResult:
    score: float
    heatmap_path: str | None = None
    is_anomaly: bool = False


@dataclass
class DecisionResult:
    status: str  # PASS | FAIL | REVIEW
    decision: str  # ACCEPT | REJECT | MANUAL_REVIEW
    quality_score: int
    defects: list[DefectDetection] = field(default_factory=list)
    anomaly: AnomalyResult = field(default_factory=lambda: AnomalyResult(score=0.0))
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": self.decision,
            "quality_score": self.quality_score,
            "reason": self.reason,
            "defects": [
                {
                    "type": d.type,
                    "confidence": round(d.confidence, 4),
                    "bbox": [round(v, 2) for v in d.bbox],
                    "severity": d.severity,
                    "affected_area": round(d.affected_area, 4),
                }
                for d in self.defects
            ],
            "anomaly": {
                "score": round(self.anomaly.score, 4),
                "heatmap_url": self.anomaly.heatmap_path,
                "is_anomaly": self.anomaly.is_anomaly,
            },
        }


def _severity(confidence: float, affected_area: float) -> str:
    if confidence >= 0.85 or affected_area >= 0.05:
        return "high"
    if confidence >= 0.55 or affected_area >= 0.02:
        return "medium"
    return "low"


def _quality_score(
    defects: list[DefectDetection],
    anomaly_score: float,
) -> int:
    if not defects and anomaly_score < 0.25:
        return max(90, int(100 - anomaly_score * 40))
    score = 100.0
    for d in defects:
        score -= d.confidence * 28
        score -= d.affected_area * 100
    score -= anomaly_score * 35
    return int(max(0, min(100, round(score))))


def decide(
    defects: list[DefectDetection],
    anomaly: AnomalyResult,
    *,
    yolo_conf_threshold: float = 0.35,
    anomaly_fail_threshold: float = 0.65,
    anomaly_review_threshold: float = 0.40,
) -> DecisionResult:
    known = [d for d in defects if d.confidence >= yolo_conf_threshold]
    for d in known:
        d.severity = _severity(d.confidence, d.affected_area)

    anomaly.is_anomaly = anomaly.score >= anomaly_review_threshold
    score = _quality_score(known, anomaly.score)

    if known:
        top = max(known, key=lambda d: d.confidence)
        status = "FAIL"
        decision = "REJECT" if top.confidence >= 0.7 else "MANUAL_REVIEW"
        reason = (
            f"FAIL: znana napaka »{top.type}« "
            f"(zaupanje {top.confidence:.0%}, prag ≥ {yolo_conf_threshold:.0%})"
        )
        return DecisionResult(
            status=status,
            decision=decision,
            quality_score=score,
            defects=known,
            anomaly=anomaly,
            reason=reason,
        )

    if anomaly.score >= anomaly_fail_threshold:
        return DecisionResult(
            status="FAIL",
            decision="MANUAL_REVIEW",
            quality_score=score,
            defects=[],
            anomaly=anomaly,
            reason=(
                f"FAIL: neznano odstopanje (indeks {anomaly.score:.2f} ≥ "
                f"prag FAIL {anomaly_fail_threshold:.2f})"
            ),
        )

    if anomaly.score >= anomaly_review_threshold:
        return DecisionResult(
            status="REVIEW",
            decision="MANUAL_REVIEW",
            quality_score=score,
            defects=[],
            anomaly=anomaly,
            reason=(
                f"REVIEW: mejno odstopanje (indeks {anomaly.score:.2f} med "
                f"{anomaly_review_threshold:.2f} in {anomaly_fail_threshold:.2f})"
            ),
        )

    return DecisionResult(
        status="PASS",
        decision="ACCEPT",
        quality_score=score,
        defects=[],
        anomaly=anomaly,
        reason=(
            f"PASS: ni zaznanih napak, indeks odstopanja {anomaly.score:.2f} < "
            f"prag REVIEW {anomaly_review_threshold:.2f}"
        ),
    )
