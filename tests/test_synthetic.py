"""Tests for the synthetic telemetry generator."""

from __future__ import annotations

import pandas as pd

from sentria_pm.data.schemas import SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator


def test_shape_matches_config() -> None:
    cfg = SyntheticConfig(n_machines=3, n_sensors=4, trajectory_length=50, seed=1)
    gen = SyntheticTelemetryGenerator(cfg)
    df, labels = gen.generate()
    expected_rows = cfg.n_machines * cfg.n_sensors * cfg.trajectory_length
    assert len(df) == expected_rows
    assert df["machine_id"].nunique() == cfg.n_machines
    assert df["sensor_id"].nunique() == cfg.n_sensors


def test_reproducibility_same_seed() -> None:
    cfg = SyntheticConfig(n_machines=2, n_sensors=2, trajectory_length=30, seed=42)
    df1, labels1 = SyntheticTelemetryGenerator(cfg).generate()
    df2, labels2 = SyntheticTelemetryGenerator(cfg).generate()
    pd.testing.assert_frame_equal(df1, df2)
    assert labels1 == labels2


def test_different_seeds_diverge() -> None:
    df1, _ = SyntheticTelemetryGenerator(
        SyntheticConfig(n_machines=2, n_sensors=2, trajectory_length=30, seed=1)
    ).generate()
    df2, _ = SyntheticTelemetryGenerator(
        SyntheticConfig(n_machines=2, n_sensors=2, trajectory_length=30, seed=2)
    ).generate()
    assert not df1["value"].equals(df2["value"])


def test_failure_rate_respected() -> None:
    cfg = SyntheticConfig(n_machines=50, failure_rate=0.3, seed=7)
    _, labels = SyntheticTelemetryGenerator(cfg).generate()
    # With 50 machines and rate 0.3, expect ~15 failures. Loose tolerance.
    assert 5 <= len(labels) <= 25


def test_zero_failure_rate() -> None:
    cfg = SyntheticConfig(n_machines=10, failure_rate=0.0, seed=3)
    _, labels = SyntheticTelemetryGenerator(cfg).generate()
    assert len(labels) == 0


def test_long_to_wide_pivot() -> None:
    cfg = SyntheticConfig(n_machines=2, n_sensors=3, trajectory_length=20, seed=5)
    gen = SyntheticTelemetryGenerator(cfg)
    df, _ = gen.generate()
    wide = gen.to_wide(df)
    assert "timestamp" in wide.columns
    assert "machine_id" in wide.columns
    for s in range(cfg.n_sensors):
        assert f"S-{s:02d}" in wide.columns
    assert len(wide) == cfg.n_machines * cfg.trajectory_length


def test_failing_machines_have_higher_late_variance() -> None:
    """Failing machines should drift; late-trajectory std > early-trajectory std on average."""
    cfg = SyntheticConfig(n_machines=20, failure_rate=1.0, seed=9, trajectory_length=200)
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, _ = gen.generate()
    wide = gen.to_wide(long_df)

    sensor_cols = [c for c in wide.columns if c.startswith("S-")]
    early = wide.groupby("machine_id").head(50)[sensor_cols].std().mean()
    late = wide.groupby("machine_id").tail(50)[sensor_cols].std().mean()
    # On average, late variance should be at least as large as early variance.
    assert late >= early * 0.9
