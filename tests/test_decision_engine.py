"""Unit tests for decision engine."""
from decision_engine.engine import AnomalyResult, DefectDetection, decide


def test_pass_when_clean():
    result = decide([], AnomalyResult(score=0.1))
    assert result.status == "PASS"
    assert result.decision == "ACCEPT"
    assert result.quality_score >= 90


def test_fail_on_known_defect():
    defects = [
        DefectDetection(type="scratch", confidence=0.96, bbox=[1, 2, 3, 4], affected_area=0.02)
    ]
    result = decide(defects, AnomalyResult(score=0.2))
    assert result.status == "FAIL"
    assert result.defects[0].type == "scratch"
    assert result.decision in {"REJECT", "MANUAL_REVIEW"}


def test_unknown_anomaly_fail():
    result = decide([], AnomalyResult(score=0.9))
    assert result.status == "FAIL"
    assert "odstopanje" in result.reason.lower() or "neznan" in result.reason.lower()


def test_review_band():
    result = decide([], AnomalyResult(score=0.5), anomaly_fail_threshold=0.54, anomaly_review_threshold=0.49)
    assert result.status == "REVIEW"
    assert result.decision == "MANUAL_REVIEW"


def test_pass_reason_mentions_threshold():
    result = decide([], AnomalyResult(score=0.2), anomaly_review_threshold=0.49)
    assert result.status == "PASS"
    assert "PASS" in result.reason
