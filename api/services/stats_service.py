"""Stats helpers."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models.inspection import Inspection


def inspections_today_filter():
    start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    return Inspection.created_at >= start


def get_today_stats(db: Session) -> dict:
    q = select(Inspection).where(inspections_today_filter())
    rows = db.scalars(q).all()
    if not rows:
        # fall back to all-time if today empty (useful for demo)
        rows = db.scalars(select(Inspection).order_by(Inspection.created_at.desc()).limit(500)).all()
        scope = "all_time_fallback"
    else:
        scope = "today"

    produced = len(rows)
    passed = sum(1 for r in rows if r.status == "PASS")
    rejected = sum(1 for r in rows if r.status == "FAIL")
    review = sum(1 for r in rows if r.status == "REVIEW")
    defect_rate = (rejected / produced * 100.0) if produced else 0.0

    defect_counter: Counter[str] = Counter()
    line_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"produced": 0, "failed": 0})
    for r in rows:
        line_stats[r.line]["produced"] += 1
        if r.status == "FAIL":
            line_stats[r.line]["failed"] += 1
        try:
            defects = json.loads(r.defects_json or "[]")
        except json.JSONDecodeError:
            defects = []
        if not defects and r.status == "FAIL":
            defect_counter["other_anomaly"] += 1
        for d in defects:
            defect_counter[d.get("type", "other")] += 1

    top_defects = [
        {"type": name, "count": count}
        for name, count in defect_counter.most_common(10)
    ]
    lines = []
    for line, vals in sorted(line_stats.items()):
        rate = (vals["failed"] / vals["produced"] * 100.0) if vals["produced"] else 0.0
        lines.append(
            {
                "line": line,
                "produced": vals["produced"],
                "failed": vals["failed"],
                "defect_rate": round(rate, 2),
                "warning": rate >= 5.0,
            }
        )

    avg_latency = float(np_mean([r.latency_ms for r in rows])) if rows else 0.0

    return {
        "scope": scope,
        "produced": produced,
        "passed": passed,
        "rejected": rejected,
        "review": review,
        "defect_rate": round(defect_rate, 2),
        "top_defects": top_defects,
        "lines": lines,
        "avg_latency_ms": round(avg_latency, 1),
    }


def np_mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _reconstruct_reason(status: str, defects: list, anomaly_score: float) -> str:
    """Human-readable reason for history rows (reason not stored in DB)."""
    fail_t = 0.54
    review_t = 0.49
    conf_t = 0.35
    if defects:
        top = max(defects, key=lambda d: float(d.get("confidence") or 0))
        conf = float(top.get("confidence") or 0)
        return (
            f"{'FAIL' if status == 'FAIL' else status}: znana napaka »{top.get('type')}« "
            f"(zaupanje {conf:.0%}, prag ≥ {conf_t:.0%})"
        )
    if status == "FAIL" or anomaly_score >= fail_t:
        return (
            f"FAIL: neznano odstopanje (indeks {anomaly_score:.2f} ≥ "
            f"prag FAIL {fail_t:.2f})"
        )
    if status == "REVIEW" or anomaly_score >= review_t:
        return (
            f"REVIEW: mejno odstopanje (indeks {anomaly_score:.2f} med "
            f"{review_t:.2f} in {fail_t:.2f})"
        )
    return (
        f"PASS: ni zaznanih napak, indeks odstopanja {anomaly_score:.2f} < "
        f"prag REVIEW {review_t:.2f}"
    )


def list_inspections(
    db: Session,
    status: str | None = None,
    line: str | None = None,
    limit: int = 100,
) -> list[dict]:
    stmt = select(Inspection).order_by(Inspection.created_at.desc()).limit(limit)
    if status:
        stmt = stmt.where(Inspection.status == status.upper())
    if line:
        stmt = stmt.where(Inspection.line == line)
    rows = db.scalars(stmt).all()
    from api.services.inspection_service import _public_url

    out = []
    for r in rows:
        defects = json.loads(r.defects_json or "[]")
        out.append(
            {
                "id": r.id,
                "part_id": r.part_id,
                "status": r.status,
                "decision": r.decision,
                "quality_score": r.quality_score,
                "line": r.line,
                "defects": defects,
                "anomaly_score": r.anomaly_score,
                "latency_ms": r.latency_ms,
                "reason": _reconstruct_reason(r.status, defects, float(r.anomaly_score or 0)),
                "image_url": _public_url(r.image_path),
                "annotated_url": _public_url(r.annotated_path),
                "heatmap_url": _public_url(r.heatmap_path) if r.heatmap_path else None,
                "created_at": r.created_at.isoformat() + "Z",
            }
        )
    return out


def list_inspection_lines(db: Session) -> list[str]:
    rows = db.scalars(select(Inspection.line).distinct().order_by(Inspection.line)).all()
    return [r for r in rows if r]


def defect_frequency(db: Session, limit: int = 500) -> list[dict]:
    rows = db.scalars(select(Inspection).order_by(Inspection.created_at.desc()).limit(limit)).all()
    counter: Counter[str] = Counter()
    for r in rows:
        defects = json.loads(r.defects_json or "[]")
        if not defects and r.status in {"FAIL", "REVIEW"}:
            counter["other_anomaly"] += 1
        for d in defects:
            counter[d.get("type", "other")] += 1
    return [{"type": k, "count": v} for k, v in counter.most_common()]
