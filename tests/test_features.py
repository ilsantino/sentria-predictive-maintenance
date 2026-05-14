"""Tests for the rolling feature extractor."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sentria_pm.data.features import FeatureConfig, RollingFeatureExtractor


def _make_simple_wide(n: int = 50) -> pd.DataFrame:
    ts = [datetime(2026, 1, 1) + timedelta(minutes=5 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": ts * 2,
            "machine_id": ["M-001"] * n + ["M-002"] * n,
            "S-00": list(np.linspace(50, 60, n)) + list(np.linspace(20, 25, n)),
            "S-01": list(np.linspace(10, 12, n)) + list(np.linspace(100, 110, n)),
        }
    )


def test_basic_features_added() -> None:
    wide = _make_simple_wide()
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5,)))
    feats = extractor.fit_transform(wide)
    for col in ("S-00_roll5_mean", "S-00_roll5_std", "S-00_lag1", "S-00_zscore"):
        assert col in feats.columns


def test_no_cross_machine_leakage() -> None:
    wide = _make_simple_wide()
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5,)))
    feats = extractor.fit_transform(wide)

    # Row 10 of M-002 has fully-warm rolling stats from only M-002 history.
    # Sensor S-00 has very different value ranges between M-001 and M-002.
    m2_row = feats[feats["machine_id"] == "M-002"].iloc[10]
    m1_last = feats[feats["machine_id"] == "M-001"].iloc[-1]
    assert abs(m2_row["S-00_roll5_mean"] - m1_last["S-00_roll5_mean"]) > 1.0


def test_drop_warmup_removes_n_rows_per_machine() -> None:
    wide = _make_simple_wide(n=40)
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5,)))
    feats = extractor.fit_transform(wide)
    warmup = 5
    trimmed = RollingFeatureExtractor.drop_warmup(feats, warmup=warmup)
    for _, group in trimmed.groupby("machine_id"):
        assert len(group) == 40 - warmup


def test_fill_remaining_eliminates_nan() -> None:
    wide = _make_simple_wide()
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5,)))
    feats = extractor.fit_transform(wide)
    trimmed = RollingFeatureExtractor.drop_warmup(feats, warmup=5)
    filled = RollingFeatureExtractor.fill_remaining(trimmed)
    feature_cols = [c for c in filled.columns if c not in ("timestamp", "machine_id")]
    assert not filled[feature_cols].isna().any().any()


def test_cross_ratios_optional() -> None:
    wide = _make_simple_wide()
    extractor = RollingFeatureExtractor(
        FeatureConfig(rolling_windows=(5,), cross_ratios=True)
    )
    feats = extractor.fit_transform(wide)
    assert "ratio_S-00_S-01" in feats.columns


def test_transform_without_fit_raises() -> None:
    wide = _make_simple_wide()
    extractor = RollingFeatureExtractor()
    try:
        extractor.transform(wide)
        raise AssertionError("Expected RuntimeError")
    except RuntimeError:
        pass


def test_no_sensor_columns_raises() -> None:
    bad = pd.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1)],
            "machine_id": ["M-001"],
            "X": [1.0],
        }
    )
    extractor = RollingFeatureExtractor()
    try:
        extractor.fit_transform(bad)
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
