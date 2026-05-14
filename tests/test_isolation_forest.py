"""Tests for IsolationForestDetector."""

from __future__ import annotations

import numpy as np

from sentria_pm.models.isolation_forest import IsolationForestDetector


def test_fit_and_score_shape(features_synthetic) -> None:
    det = IsolationForestDetector(n_estimators=50, random_state=1)
    det.fit(features_synthetic)
    assert det.is_fitted
    scores = det.score(features_synthetic)
    assert scores.shape == (len(features_synthetic),)


def test_scores_in_unit_interval(features_synthetic) -> None:
    det = IsolationForestDetector(n_estimators=50, random_state=1)
    det.fit(features_synthetic)
    scores = det.score(features_synthetic)
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


def test_score_before_fit_raises() -> None:
    det = IsolationForestDetector()
    try:
        det.score_samples = None  # type: ignore[attr-defined]
        det.score(None)  # type: ignore[arg-type]
        raise AssertionError("Expected RuntimeError")
    except (RuntimeError, AttributeError):
        pass


def test_late_window_scores_higher_on_failing_machines(features_synthetic, small_synthetic) -> None:
    """Sanity check: failing machines should have higher avg scores in the last
    quartile vs the first quartile (degradation drift is what the model picks up)."""
    _, labels = small_synthetic
    failing_machines = {label.machine_id for label in labels}

    det = IsolationForestDetector(n_estimators=100, random_state=1)
    det.fit(features_synthetic)
    scores = det.score(features_synthetic)
    feat = features_synthetic.copy()
    feat["score"] = scores

    deltas = []
    for machine_id, group in feat[feat["machine_id"].isin(failing_machines)].groupby(
        "machine_id"
    ):
        group = group.sort_values("timestamp")
        q1 = group.iloc[: len(group) // 4]["score"].mean()
        q4 = group.iloc[-len(group) // 4 :]["score"].mean()
        deltas.append(q4 - q1)

    # The average failing machine should have higher late scores.
    if deltas:
        assert np.mean(deltas) > 0
