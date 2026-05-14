"""Abstract base class for anomaly detectors.

All detectors share the same contract:

- ``fit(features)`` — learn normal behavior from a DataFrame of features
  with ``timestamp``, ``machine_id``, and numeric feature columns.
- ``score(features)`` — return a NumPy array of anomaly scores in ``[0, 1]``
  (1.0 = maximum anomaly), aligned to the input row order.

The detectors do not produce ``AnomalyAlert`` objects directly — the
ensemble layer is responsible for thresholding and structured output.
This keeps each detector cheap to compose and test in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AnomalyDetector(ABC):
    """Contract for all anomaly detectors."""

    name: str = "abstract"

    @abstractmethod
    def fit(self, features: pd.DataFrame) -> "AnomalyDetector":
        """Learn normal behavior. Returns self for chaining."""

    @abstractmethod
    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1], aligned to feature rows."""

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fit. Defaults to False; subclasses
        with internal state should override."""
        return False

    @staticmethod
    def _select_numeric(features: pd.DataFrame) -> pd.DataFrame:
        """Drop non-numeric columns (timestamp, machine_id) for modeling."""
        drop_cols = [c for c in ("timestamp", "machine_id") if c in features.columns]
        return features.drop(columns=drop_cols).select_dtypes(include=[np.number])

    @staticmethod
    def _normalize_scores(raw: np.ndarray) -> np.ndarray:
        """Min-max normalize raw scores into [0, 1]. Robust to constant input."""
        raw = np.asarray(raw, dtype=np.float64)
        if raw.size == 0:
            return raw
        lo = np.nanmin(raw)
        hi = np.nanmax(raw)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            return np.zeros_like(raw)
        return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
