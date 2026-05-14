"""Tests for evaluation metrics."""

from __future__ import annotations

from datetime import datetime, timedelta

from sentria_pm.data.schemas import EnsembleScore, FailureLabel, FailureType
from sentria_pm.evaluation.metrics import compute_metrics


def _alert(ts: datetime, machine_id: str, score: float, threshold: float = 0.5) -> EnsembleScore:
    return EnsembleScore(
        timestamp=ts,
        machine_id=machine_id,
        score=score,
        threshold=threshold,
        alert=score >= threshold,
    )


def test_perfect_detection() -> None:
    """One alert 30 min before failure → TP, no FP, no FN."""
    failure = datetime(2026, 1, 1, 12, 0)
    labels = [
        FailureLabel(
            machine_id="M-001",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    alerts = [_alert(failure - timedelta(minutes=30), "M-001", 0.9)]
    res = compute_metrics(alerts, labels, lookback_window_minutes=60)
    assert res.n_true_positives == 1
    assert res.n_false_positives == 0
    assert res.n_false_negatives == 0
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.lead_time_minutes_p50 == 30.0


def test_alert_outside_window_is_fp() -> None:
    failure = datetime(2026, 1, 1, 12, 0)
    labels = [
        FailureLabel(
            machine_id="M-001",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    # Alert 5 hours before — outside 60-min lookback.
    alerts = [_alert(failure - timedelta(hours=5), "M-001", 0.9)]
    res = compute_metrics(alerts, labels, lookback_window_minutes=60)
    assert res.n_true_positives == 0
    assert res.n_false_positives == 1
    assert res.n_false_negatives == 1


def test_no_alert_is_fn() -> None:
    failure = datetime(2026, 1, 1, 12, 0)
    labels = [
        FailureLabel(
            machine_id="M-001",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    res = compute_metrics([], labels, lookback_window_minutes=60)
    assert res.n_true_positives == 0
    assert res.n_false_negatives == 1
    assert res.recall == 0.0


def test_multiple_alerts_one_failure_count_as_one_tp() -> None:
    failure = datetime(2026, 1, 1, 12, 0)
    labels = [
        FailureLabel(
            machine_id="M-001",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    alerts = [
        _alert(failure - timedelta(minutes=45), "M-001", 0.8),
        _alert(failure - timedelta(minutes=30), "M-001", 0.85),
        _alert(failure - timedelta(minutes=15), "M-001", 0.9),
    ]
    res = compute_metrics(alerts, labels, lookback_window_minutes=60)
    assert res.n_true_positives == 1
    assert res.n_false_positives == 0
    # Lead time is from the FIRST TP alert.
    assert res.lead_time_minutes_p50 == 45.0


def test_machine_specific_tp() -> None:
    """An alert on M-001 should NOT count toward an M-002 failure."""
    failure = datetime(2026, 1, 1, 12, 0)
    labels = [
        FailureLabel(
            machine_id="M-002",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    alerts = [_alert(failure - timedelta(minutes=20), "M-001", 0.9)]
    res = compute_metrics(alerts, labels, lookback_window_minutes=60)
    assert res.n_true_positives == 0
    assert res.n_false_positives == 1
    assert res.n_false_negatives == 1


def test_false_alarm_rate_computed() -> None:
    """Two FPs spanning 2 days on 1 machine → FP rate = 1 per machine-day."""
    failure = datetime(2026, 1, 5, 0, 0)  # far away → both alerts are FPs
    labels = [
        FailureLabel(
            machine_id="M-001",
            failure_time=failure,
            failure_type=FailureType.BEARING,
            downtime_minutes=60,
        )
    ]
    alerts = [
        _alert(datetime(2026, 1, 1, 0, 0), "M-001", 0.9),
        _alert(datetime(2026, 1, 3, 0, 0), "M-001", 0.9),
    ]
    res = compute_metrics(alerts, labels, lookback_window_minutes=60, n_machines=1)
    assert res.n_false_positives == 2
    # Span is 4 days (Jan 1 to Jan 5), n_machines=1 → FP rate = 2/4 = 0.5
    assert res.false_alarm_rate_per_machine_day == 0.5


def test_empty_inputs() -> None:
    res = compute_metrics([], [])
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.f1 == 0.0
    assert res.lead_time_minutes_p50 is None
