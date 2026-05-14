"""Isolation Forest anomaly detector.

Reference
---------
Liu, F. T., Ting, K. M., Zhou, Z. H. *Isolation Forest.* ICDM 2008.

Tree-based density estimator that isolates anomalies by random splits.
Inexpensive, handles high-dimensional rolling-feature vectors well,
and serves as a strong baseline. Score is the average path length in
the isolation tree ensemble — inverted and normalized to [0, 1] here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from sentria_pm.models.base import AnomalyDetector


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest wrapped to the SENTRIA PM detector contract."""

    name = "isolation_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        max_samples: int | str = "auto",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state
        self._model: IsolationForest | None = None
        self._feature_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, features: pd.DataFrame) -> "IsolationForestDetector":
        """Fit on the numeric feature columns of ``features``."""
        X = self._select_numeric(features)
        if X.empty or X.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit IsolationForest.")
        self._feature_names = list(X.columns)
        X_filled = X.ffill().fillna(0.0).to_numpy()
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(X_filled)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Score samples. Higher = more anomalous."""
        if self._model is None:
            raise RuntimeError("IsolationForestDetector must be fit before scoring.")
        X = self._select_numeric(features)
        # Align columns to training-time order
        missing = [c for c in self._feature_names if c not in X.columns]
        for col in missing:
            X[col] = 0.0
        X = X[self._feature_names]
        X_filled = X.ffill().fillna(0.0).to_numpy()

        # sklearn returns the anomaly score: higher = more normal. Invert.
        raw = -self._model.score_samples(X_filled)
        return self._normalize_scores(raw)
