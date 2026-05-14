"""Tests for the ensemble detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentria_pm.models.ensemble import EnsembleDetector
from sentria_pm.models.isolation_forest import IsolationForestDetector


class _FakeDetector:
    """Pre-fit fake that returns a known constant score per row."""

    is_fitted = True

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    def fit(self, features):  # pragma: no cover - not exercised
        return self

    def score(self, features):
        return np.full(len(features), self.value)


def test_mean_fusion() -> None:
    f1 = _FakeDetector("a", 0.2)
    f2 = _FakeDetector("b", 0.8)
    ens = EnsembleDetector([f1, f2], strategy="mean")  # type: ignore[list-item]
    features = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"), "machine_id": ["M-1"] * 3}
    )
    scores = ens.score(features)
    assert np.allclose(scores, [0.5, 0.5, 0.5])


def test_max_fusion() -> None:
    f1 = _FakeDetector("a", 0.2)
    f2 = _FakeDetector("b", 0.8)
    ens = EnsembleDetector([f1, f2], strategy="max")  # type: ignore[list-item]
    features = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=2, freq="5min"), "machine_id": ["M-1"] * 2}
    )
    scores = ens.score(features)
    assert np.allclose(scores, [0.8, 0.8])


def test_weighted_fusion() -> None:
    f1 = _FakeDetector("a", 0.0)
    f2 = _FakeDetector("b", 1.0)
    ens = EnsembleDetector(
        [f1, f2], strategy="weighted", weights={"a": 1.0, "b": 3.0}  # type: ignore[list-item]
    )
    features = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=1, freq="5min"), "machine_id": ["M-1"]}
    )
    scores = ens.score(features)
    # weighted mean: (0*1 + 1*3) / 4 = 0.75
    assert np.allclose(scores, [0.75])


def test_alerts_respect_threshold(features_synthetic) -> None:
    det = IsolationForestDetector(n_estimators=50, random_state=1)
    ens = EnsembleDetector([det], default_threshold=0.7)
    ens.fit(features_synthetic)
    ens.calibrate_threshold(features_synthetic, target_false_alarm_quantile=0.95)
    alerts = ens.alerts(features_synthetic)
    assert len(alerts) == len(features_synthetic)
    # Some alerts should fire on a calibrated 0.95 quantile threshold.
    n_alerts = sum(1 for a in alerts if a.alert)
    # About 5% of rows should be alerts (calibration at q95).
    assert 0 < n_alerts < len(alerts)


def test_empty_detectors_raises() -> None:
    try:
        EnsembleDetector([])
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
