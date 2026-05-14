"""ARIMA univariate baseline.

Trains one ARIMA model per sensor per machine on the *raw* sensor series.
At score time, each sensor's residual (one-step-ahead forecast error,
normalized by training residual std) is computed, and the per-row score
is the max absolute z-score across sensors.

This serves as a deliberately simple baseline: any modern detector must
beat this to justify its complexity.

Implementation notes
--------------------
- We fit per (machine, sensor) to avoid forcing cross-machine
  stationarity assumptions.
- We use ARIMA(1, 0, 0) by default (i.e., AR(1)) for speed and
  robustness. Higher orders are configurable but slow.
- Fitting silently handles degenerate series (constant or near-constant)
  by skipping ARIMA and using rolling-z-score residuals instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA

from sentria_pm.models.base import AnomalyDetector


class ARIMABaseline(AnomalyDetector):
    """Per-(machine, sensor) ARIMA baseline."""

    name = "arima_baseline"

    def __init__(self, order: tuple[int, int, int] = (1, 0, 0)) -> None:
        self.order = order
        self._sensor_columns: list[str] = []
        self._residual_stds: dict[tuple[str, str], float] = {}
        self._machine_models: dict[tuple[str, str], object] = {}
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, features: pd.DataFrame) -> "ARIMABaseline":
        """Fit one ARIMA per (machine, sensor) pair."""
        self._sensor_columns = [c for c in features.columns if c.startswith("S-") and "_" not in c]
        if not self._sensor_columns:
            raise ValueError(
                "ARIMABaseline expects raw sensor columns (S-XX), not derived features."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            for machine_id, group in features.groupby("machine_id", sort=False):
                group = group.sort_values("timestamp").reset_index(drop=True)
                for sensor in self._sensor_columns:
                    series = group[sensor].to_numpy(dtype=np.float64)
                    if np.nanstd(series) < 1e-6 or len(series) < 10:
                        self._residual_stds[(machine_id, sensor)] = max(np.nanstd(series), 1e-6)
                        self._machine_models[(machine_id, sensor)] = None
                        continue
                    try:
                        model = ARIMA(series, order=self.order, enforce_stationarity=False)
                        fit_result = model.fit()
                        residuals = np.asarray(fit_result.resid, dtype=np.float64)
                        resid_std = float(np.nanstd(residuals))
                        self._residual_stds[(machine_id, sensor)] = max(resid_std, 1e-6)
                        self._machine_models[(machine_id, sensor)] = fit_result
                    except Exception:
                        # Convergence failure: fall back to rolling-zscore proxy
                        self._residual_stds[(machine_id, sensor)] = max(
                            float(np.nanstd(series)), 1e-6
                        )
                        self._machine_models[(machine_id, sensor)] = None

        self._is_fitted = True
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Score each row by max |z| of one-step-ahead residuals across sensors."""
        if not self._is_fitted:
            raise RuntimeError("ARIMABaseline must be fit before scoring.")

        scores = np.zeros(len(features), dtype=np.float64)
        row_idx = features.index.to_numpy()

        for machine_id, group in features.groupby("machine_id", sort=False):
            group = group.sort_values("timestamp")
            machine_positions = np.array([row_idx.tolist().index(i) for i in group.index])
            per_row_max_z = np.zeros(len(group), dtype=np.float64)

            for sensor in self._sensor_columns:
                if sensor not in group.columns:
                    continue
                series = group[sensor].to_numpy(dtype=np.float64)
                key = (machine_id, sensor)
                resid_std = self._residual_stds.get(key, 1.0)
                model = self._machine_models.get(key)

                if model is None:
                    # Fallback: rolling-zscore using a 20-step window
                    rolling_mean = pd.Series(series).rolling(20, min_periods=2).mean().to_numpy()
                    rolling_std = (
                        pd.Series(series).rolling(20, min_periods=2).std().to_numpy()
                    )
                    rolling_std = np.where(
                        np.isnan(rolling_std) | (rolling_std < 1e-6),
                        resid_std,
                        rolling_std,
                    )
                    z = np.abs((series - rolling_mean) / rolling_std)
                    z = np.nan_to_num(z, nan=0.0)
                else:
                    # One-step-ahead forecasts: yhat_t = sum(phi * past); residual = y - yhat
                    # Use the fitted in-sample residuals if length matches, else
                    # forecast by applying AR coefficients.
                    fitted = np.asarray(getattr(model, "fittedvalues", series), dtype=np.float64)
                    if fitted.shape[0] != series.shape[0]:
                        fitted = np.concatenate(
                            [np.array([series[0]]), fitted[: series.shape[0] - 1]]
                        )
                    residuals = series - fitted
                    z = np.abs(residuals) / resid_std
                    z = np.nan_to_num(z, nan=0.0)

                per_row_max_z = np.maximum(per_row_max_z, z)

            scores[machine_positions] = per_row_max_z

        return self._normalize_scores(scores)
