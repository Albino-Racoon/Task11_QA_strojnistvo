"""API contract smoke tests."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.database import init_db
from api.main import app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    init_db()
    return TestClient(app)


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "models" in body


def test_inspect_upload(client, tmp_path):
    img = np.full((240, 240, 3), 130, np.uint8)
    cv2.line(img, (20, 30), (200, 180), (10, 10, 10), 2)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), img)
    with path.open("rb") as f:
        res = client.post(
            "/api/inspect",
            files={"file": ("sample.jpg", f, "image/jpeg")},
            data={"line": "Linija A"},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"PASS", "FAIL", "REVIEW"}
    assert "quality_score" in body
    assert "latency_ms" in body
    assert "defects" in body
    assert "anomaly" in body


def test_stats_and_history(client):
    stats = client.get("/api/stats/today")
    assert stats.status_code == 200
    hist = client.get("/api/inspections")
    assert hist.status_code == 200
    assert "items" in hist.json()
