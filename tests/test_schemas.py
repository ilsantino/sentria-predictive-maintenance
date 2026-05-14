"""Schema validation tests — Pydantic contracts."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from sentria_pm.data.schemas import (
    AnomalyAlert,
    EnsembleScore,
    EvaluationResult,
    FailureLabel,
    FailureType,
    Incident,
    IncidentSeverity,
    SyntheticConfig,
    TelemetryReading,
)


class TestTelemetryReading:
    def test_valid_reading(self) -> None:
        r = TelemetryReading(
            timestamp=datetime(2026, 1, 1, 12, 0),
            machine_id="M-001",
            sensor_id="S-00",
            value=42.5,
            unit="°C",
        )
        assert r.machine_id == "M-001"
        assert r.value == 42.5

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryReading(
                timestamp=datetime(2026, 1, 1),
                machine_id="M-001",
                sensor_id="S-00",
                value=float("nan"),
            )

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValidationError):
            TelemetryReading(
                timestamp=datetime(2026, 1, 1),
                machine_id="M-001",
                sensor_id="S-00",
                value=float("inf"),
            )

    def test_immutable(self) -> None:
        r = TelemetryReading(
            timestamp=datetime(2026, 1, 1),
            machine_id="M-001",
            sensor_id="S-00",
            value=1.0,
        )
        with pytest.raises(ValidationError):
            r.value = 2.0  # type: ignore[misc]


class TestAnomalyAlert:
    def test_score_must_be_in_range(self) -> None:
        with pytest.raises(ValidationError):
            AnomalyAlert(
                timestamp=datetime(2026, 1, 1),
                machine_id="M-001",
                score=1.5,
                model_name="x",
            )
        with pytest.raises(ValidationError):
            AnomalyAlert(
                timestamp=datetime(2026, 1, 1),
                machine_id="M-001",
                score=-0.1,
                model_name="x",
            )


class TestEnsembleScore:
    def test_alert_consistent_with_threshold(self) -> None:
        # score >= threshold AND alert=True is OK
        EnsembleScore(
            timestamp=datetime(2026, 1, 1),
            machine_id="M-001",
            score=0.8,
            threshold=0.7,
            alert=True,
        )
        # score < threshold AND alert=False is OK
        EnsembleScore(
            timestamp=datetime(2026, 1, 1),
            machine_id="M-001",
            score=0.5,
            threshold=0.7,
            alert=False,
        )

    def test_alert_inconsistent_raises(self) -> None:
        with pytest.raises(ValidationError):
            EnsembleScore(
                timestamp=datetime(2026, 1, 1),
                machine_id="M-001",
                score=0.8,
                threshold=0.7,
                alert=False,
            )


class TestFailureLabel:
    def test_negative_downtime_raises(self) -> None:
        with pytest.raises(ValidationError):
            FailureLabel(
                machine_id="M-001",
                failure_time=datetime(2026, 1, 1),
                failure_type=FailureType.BEARING,
                downtime_minutes=-5,
            )


class TestIncident:
    def test_severity_enum(self) -> None:
        inc = Incident(
            incident_id="I-001",
            timestamp=datetime(2026, 1, 1),
            machine_id="M-001",
            severity="critical",  # type: ignore[arg-type]
        )
        assert inc.severity == IncidentSeverity.CRITICAL


class TestEvaluationResult:
    def test_metrics_in_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(
                model_name="x",
                n_alerts=0,
                n_true_positives=0,
                n_false_positives=0,
                n_false_negatives=0,
                precision=1.5,
                recall=0.5,
                f1=0.5,
                false_alarm_rate_per_machine_day=0.0,
                threshold=0.5,
            )


class TestSyntheticConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = SyntheticConfig()
        assert cfg.n_machines >= 1
        assert cfg.failure_rate <= 1.0

    def test_zero_noise_invalid(self) -> None:
        with pytest.raises(ValidationError):
            SyntheticConfig(noise_std=0.0)
