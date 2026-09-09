"""Seed demo inspection history for dashboard demos."""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.database import SessionLocal, init_db
from api.models.inspection import Inspection

DEFECT_TYPES = [
    "scratch",
    "inclusion",
    "oil_spot",
    "welding_line",
    "rolled_pit",
    "crease",
    "punching_hole",
]


def seed(n: int = 120) -> None:
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(Inspection).count()
        if existing >= 50:
            print(f"Seed preskočen — že {existing} zapisov.")
            return
        now = datetime.utcnow()
        lines = ["Linija A", "Linija B", "Linija C"]
        for i in range(n):
            status_roll = random.random()
            if status_roll < 0.86:
                status, decision = "PASS", "ACCEPT"
                defects = []
                anomaly = random.uniform(0.02, 0.25)
                score = random.randint(88, 99)
            elif status_roll < 0.94:
                status, decision = "FAIL", "REJECT"
                dtype = random.choice(DEFECT_TYPES)
                defects = [
                    {
                        "type": dtype,
                        "confidence": round(random.uniform(0.7, 0.98), 3),
                        "bbox": [40, 40, 180, 120],
                        "severity": "medium",
                        "affected_area": round(random.uniform(0.01, 0.05), 4),
                    }
                ]
                anomaly = random.uniform(0.4, 0.9)
                score = random.randint(40, 75)
            else:
                status, decision = "REVIEW", "MANUAL_REVIEW"
                defects = []
                anomaly = random.uniform(0.42, 0.62)
                score = random.randint(70, 85)

            line = random.choices(lines, weights=[0.4, 0.3, 0.3])[0]
            # bias Linija B toward more defects
            if line == "Linija B" and status == "PASS" and random.random() < 0.15:
                status, decision = "FAIL", "REJECT"
                defects = [
                    {
                        "type": "scratch",
                        "confidence": 0.91,
                        "bbox": [30, 50, 160, 90],
                        "severity": "high",
                        "affected_area": 0.03,
                    }
                ]
                score = 62
                anomaly = 0.7

            rec = Inspection(
                part_id=f"A{90000 + i}",
                status=status,
                decision=decision,
                quality_score=score,
                line=line,
                defects_json=json.dumps(defects),
                anomaly_score=anomaly,
                latency_ms=round(random.uniform(120, 420), 1),
                image_path="",
                annotated_path="",
                heatmap_path="",
                created_at=now - timedelta(minutes=random.randint(0, 60 * 12)),
            )
            db.add(rec)
        db.commit()
        print(f"Dodanih {n} demo inšpekcij.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
