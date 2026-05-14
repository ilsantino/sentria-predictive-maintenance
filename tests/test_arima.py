"""Tests for ARIMA baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentria_pm.data.schemas import SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator
from sentria_pm.models.arima_baseline import ARIMABaseline


def _wide_dataset() -> pd.DataFrame:
    cfg = SyntheticConfig(n_machines=3, n_sensors=3, trajectory_length=80, seed=4, failure_rate=1.0)
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, _ = gen.generate()
    return gen.to_wide(long_df)


def test_fit_and_score() -> None:
    wide = _wide_dataset()
    det = ARIMABaseline(order=(1, 0, 0))
    det.fit(wide)
    assert det.is_fitted
    scores = det.score(wide)
    assert scores.shape == (len(wide),)
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_constant_series_handled() -> None:
    """ARIMA on a constant series must not crash — fallback path engaged."""
    n = 60
    wide = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="5min"),
            "machine_id": ["M-001"] * n,
            "S-00": [50.0] * n,
            "S-01": [50.0] * n,
        }
    )
    det = ARIMABaseline()
    det.fit(wide)
    scores = det.score(wide)
    assert np.all(np.isfinite(scores))


def test_score_before_fit_raises() -> None:
    det = ARIMABaseline()
    try:
        det.score(pd.DataFrame())
        raise AssertionError("Expected RuntimeError")
    except RuntimeError:
        pass
