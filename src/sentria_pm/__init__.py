"""SENTRIA Predictive Maintenance.

Anomaly detection layer for industrial production lines. Read-only alerting
on top of multivariate sensor telemetry, with ensemble of Isolation Forest,
LSTM autoencoder, and ARIMA baseline.
"""

__version__ = "0.1.0a0"

from sentria_pm.data.schemas import (
    AnomalyAlert,
    EvaluationResult,
    FailureLabel,
    Incident,
    TelemetryReading,
)

__all__ = [
    "AnomalyAlert",
    "EvaluationResult",
    "FailureLabel",
    "Incident",
    "TelemetryReading",
    "__version__",
]
