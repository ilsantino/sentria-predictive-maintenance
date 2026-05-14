"""Ensemble of heterogeneous anomaly detectors with score fusion.

Fusion strategies
-----------------
- ``mean``: simple average of normalized scores.
- ``weighted``: caller-supplied weights per detector.
- ``max``: most pessimistic detector wins.

Thresholding is per-machine to allow each line to have its own
calibration. ``calibrate_threshold`` uses a percentile of the score
distribution on a held-out *normal* set to fix an operational
false-alarm budget.

The ensemble produces structured ``EnsembleScore`` objects, which include
component scores so downstream consumers (dashboards, operator UI) can
attribute the decision back to specific detectors.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from sentria_pm.data.schemas import EnsembleScore
from sentria_pm.models.base import AnomalyDetector

FusionStrategy = Literal["mean", "weighted", "max"]


class EnsembleDetector:
    """Fuses scores from multiple ``AnomalyDetector`` instances."""

    def __init__(
        self,
        detectors: list[AnomalyDetector],
        weights: dict[str, float] | None = None,
        strategy: FusionStrategy = "mean",
        default_threshold: float = 0.7,
    ) -> None:
        if not detectors:
            raise ValueError("EnsembleDetector requires at least one detector.")
        self.detectors = detectors
        self.weights = weights or {d.name: 1.0 for d in detectors}
        self.strategy = strategy
        self.default_threshold = default_threshold
        self._per_machine_threshold: dict[str, float] = {}

    @property
    def is_fitted(self) -> bool:
        return all(d.is_fitted for d in self.detectors)

    def fit(self, features: pd.DataFrame) -> "EnsembleDetector":
        """Fit all component detectors on the same feature frame."""
        for detector in self.detectors:
            detector.fit(features)
        return self

    def calibrate_threshold(
        self, features: pd.DataFrame, target_false_alarm_quantile: float = 0.99
    ) -> "EnsembleDetector":
        """Set per-machine thresholds at the given quantile of training scores.

        Default 0.99 means the top 1% of normal-data scores trigger an
        alert in production — a starting budget that should be retuned
        once real alert volume is measured.
        """
        if not 0.5 <= target_false_alarm_quantile <= 0.9999:
            raise ValueError("Quantile must be in [0.5, 0.9999].")
        fused = self._fuse_per_row(features)
        feat_with_score = features[["timestamp", "machine_id"]].copy()
        feat_with_score["score"] = fused
        for machine_id, group in feat_with_score.groupby("machine_id", sort=False):
            self._per_machine_threshold[machine_id] = float(
                np.quantile(group["score"], target_false_alarm_quantile)
            )
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return fused scores for each row."""
        return self._fuse_per_row(features)

    def alerts(self, features: pd.DataFrame) -> list[EnsembleScore]:
        """Produce structured ``EnsembleScore`` objects per row, with the
        ``alert`` flag set per the calibrated per-machine threshold."""
        per_detector = {
            d.name: np.asarray(d.score(features), dtype=np.float64) for d in self.detectors
        }
        fused = self._fuse_arrays(per_detector)
        out: list[EnsembleScore] = []
        for i, (_, row) in enumerate(features.iterrows()):
            machine_id = row["machine_id"]
            threshold = self._per_machine_threshold.get(machine_id, self.default_threshold)
            score = float(fused[i])
            component = {name: float(per_detector[name][i]) for name in per_detector}
            ts = row["timestamp"]
            if not isinstance(ts, datetime):
                ts = pd.to_datetime(ts).to_pydatetime()
            out.append(
                EnsembleScore(
                    timestamp=ts,
                    machine_id=machine_id,
                    score=score,
                    threshold=threshold,
                    alert=score >= threshold,
                    component_scores=component,
                )
            )
        return out

    def _fuse_per_row(self, features: pd.DataFrame) -> np.ndarray:
        per_detector = {
            d.name: np.asarray(d.score(features), dtype=np.float64) for d in self.detectors
        }
        return self._fuse_arrays(per_detector)

    def _fuse_arrays(self, scores: dict[str, np.ndarray]) -> np.ndarray:
        arrays = list(scores.values())
        if self.strategy == "max":
            return np.max(np.vstack(arrays), axis=0)

        if self.strategy == "mean":
            return np.mean(np.vstack(arrays), axis=0)

        # weighted
        weights = np.array([self.weights.get(name, 1.0) for name in scores], dtype=np.float64)
        weights = weights / max(weights.sum(), 1e-9)
        stacked = np.vstack(arrays)
        return np.average(stacked, axis=0, weights=weights)
