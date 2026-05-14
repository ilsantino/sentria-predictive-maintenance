"""Tests for walk-forward evaluation harness."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from sentria_pm.evaluation.walk_forward import WalkForwardEvaluator


def _toy_frame(n_machines: int = 2, n_steps: int = 30) -> pd.DataFrame:
    rows = []
    for m in range(n_machines):
        for i in range(n_steps):
            rows.append(
                {
                    "timestamp": datetime(2026, 1, 1) + timedelta(minutes=5 * i),
                    "machine_id": f"M-{m:03d}",
                    "S-00": float(i),
                }
            )
    return pd.DataFrame(rows)


def test_n_folds_respected() -> None:
    ev = WalkForwardEvaluator(n_folds=3, min_train_fraction=0.4, test_fraction=0.2)
    splits = ev.splits(_toy_frame())
    assert len(splits) == 3


def test_train_test_disjoint() -> None:
    ev = WalkForwardEvaluator(n_folds=2)
    splits = ev.splits(_toy_frame())
    for split in splits:
        intersection = set(split.train_indices.tolist()) & set(split.test_indices.tolist())
        assert not intersection


def test_train_precedes_test() -> None:
    ev = WalkForwardEvaluator(n_folds=2)
    df = _toy_frame()
    splits = ev.splits(df)
    for split in splits:
        if len(split.train_indices) == 0 or len(split.test_indices) == 0:
            continue
        train_max_ts = df.loc[split.train_indices, "timestamp"].max()
        test_min_ts = df.loc[split.test_indices, "timestamp"].min()
        # Allow equal because boundary index — strict less-or-equal is enough.
        assert train_max_ts <= test_min_ts


def test_invalid_fractions_raise() -> None:
    with pytest.raises(ValueError):
        WalkForwardEvaluator(n_folds=0)
    with pytest.raises(ValueError):
        WalkForwardEvaluator(min_train_fraction=0.0)
    with pytest.raises(ValueError):
        WalkForwardEvaluator(test_fraction=0.0)
    with pytest.raises(ValueError):
        WalkForwardEvaluator(min_train_fraction=0.8, test_fraction=0.3)


def test_short_machines_excluded() -> None:
    """Machines with <10 steps should be dropped from splits."""
    short = _toy_frame(n_machines=1, n_steps=5)
    long = _toy_frame(n_machines=1, n_steps=40)
    long["machine_id"] = "M-100"
    df = pd.concat([short, long], ignore_index=True)
    ev = WalkForwardEvaluator(n_folds=1)
    splits = ev.splits(df)
    # All resulting indices belong to the long machine
    for split in splits:
        machines_in_split = set(df.loc[split.train_indices, "machine_id"]) | set(
            df.loc[split.test_indices, "machine_id"]
        )
        assert "M-000" not in machines_in_split
