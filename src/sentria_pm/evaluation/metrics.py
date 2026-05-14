"""Evaluation metrics for predictive maintenance.

A predictive-maintenance evaluation is *not* the same as point
classification. Standard precision/recall make sense only after we
define what "the right answer" means when alerts arrive minutes or
hours before the labeled failure.

Rules used here
---------------
- An alert is a **true positive** if it occurs in
  ``[failure_time - lookback_window, failure_time]`` for a labeled
  failure of the same machine. Multiple TP alerts inside the window
  for the same failure count as ONE event-level TP (we do not
  reward an alert-storm).
- An alert outside any lookback window is a **false positive**.
- A failure with NO alerts in its lookback window is a **false negative**.
- ``lead_time`` = (failure_time - first_TP_alert_time), reported per
  failure that we caught.
- ``false_alarm_rate_per_machine_day`` = FP / (n_machines * n_days).

The window-based event evaluation matches industry practice for
PdM (see Saxena 2008 NASA C-MAPSS prognostic metrics).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from sentria_pm.data.schemas import EnsembleScore, EvaluationResult, FailureLabel


@dataclass
class PredictiveMaintenanceMetrics:
    """Detailed breakdown of evaluation. Public for inspection in tests."""

    n_alerts: int = 0
    n_true_positives: int = 0
    n_false_positives: int = 0
    n_false_negatives: int = 0
    lead_times_minutes: list[float] = field(default_factory=list)
    n_machine_days: float = 0.0

    @property
    def precision(self) -> float:
        denom = self.n_true_positives + self.n_false_positives
        return self.n_true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.n_true_positives + self.n_false_negatives
        return self.n_true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def lead_time_p50(self) -> float | None:
        return float(np.median(self.lead_times_minutes)) if self.lead_times_minutes else None

    @property
    def lead_time_p95(self) -> float | None:
        return (
            float(np.quantile(self.lead_times_minutes, 0.95))
            if self.lead_times_minutes
            else None
        )

    @property
    def false_alarm_rate_per_machine_day(self) -> float:
        if self.n_machine_days <= 0:
            return 0.0
        return self.n_false_positives / self.n_machine_days

    def to_evaluation_result(self, model_name: str, threshold: float) -> EvaluationResult:
        return EvaluationResult(
            model_name=model_name,
            n_alerts=self.n_alerts,
            n_true_positives=self.n_true_positives,
            n_false_positives=self.n_false_positives,
            n_false_negatives=self.n_false_negatives,
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            lead_time_minutes_p50=self.lead_time_p50,
            lead_time_minutes_p95=self.lead_time_p95,
            false_alarm_rate_per_machine_day=self.false_alarm_rate_per_machine_day,
            threshold=threshold,
        )


def compute_metrics(
    alerts: list[EnsembleScore],
    labels: list[FailureLabel],
    lookback_window_minutes: float = 180.0,
    evaluation_period_minutes: float | None = None,
    n_machines: int | None = None,
    threshold: float = 0.7,
    model_name: str = "ensemble",
) -> EvaluationResult:
    """Compute event-level metrics for a stream of alerts.

    Parameters
    ----------
    alerts : list[EnsembleScore]
        Output of ``EnsembleDetector.alerts(...)``. Only rows with
        ``alert == True`` are considered.
    labels : list[FailureLabel]
        Ground truth failure events.
    lookback_window_minutes : float
        How far before a failure an alert can be and still count as TP.
    evaluation_period_minutes : float | None
        Total time span covered by evaluation (used for false-alarm rate).
        If None, inferred from the alert timestamps.
    n_machines : int | None
        Number of distinct machines in the evaluation. If None,
        inferred from alerts + labels.
    """
    metrics = PredictiveMaintenanceMetrics()

    alert_only = [a for a in alerts if a.alert]
    metrics.n_alerts = len(alert_only)

    labels_by_machine: dict[str, list[FailureLabel]] = defaultdict(list)
    for label in labels:
        labels_by_machine[label.machine_id].append(label)

    # Track which failures have been "caught"
    caught_failures: set[tuple[str, datetime]] = set()
    failure_first_tp_ts: dict[tuple[str, datetime], datetime] = {}

    # Sort alerts chronologically per machine
    alerts_by_machine: dict[str, list[EnsembleScore]] = defaultdict(list)
    for a in alert_only:
        alerts_by_machine[a.machine_id].append(a)
    for v in alerts_by_machine.values():
        v.sort(key=lambda x: x.timestamp)

    lookback = timedelta(minutes=lookback_window_minutes)

    for machine_id, machine_alerts in alerts_by_machine.items():
        machine_labels = labels_by_machine.get(machine_id, [])
        for alert in machine_alerts:
            matched_failure = None
            for label in machine_labels:
                if label.failure_time - lookback <= alert.timestamp <= label.failure_time:
                    matched_failure = label
                    break

            if matched_failure is None:
                metrics.n_false_positives += 1
                continue

            failure_key = (matched_failure.machine_id, matched_failure.failure_time)
            if failure_key in caught_failures:
                # Already counted; subsequent alerts for the same failure
                # are NOT additional TPs (event-level scoring).
                continue
            caught_failures.add(failure_key)
            failure_first_tp_ts[failure_key] = alert.timestamp
            metrics.n_true_positives += 1
            lead_minutes = (matched_failure.failure_time - alert.timestamp).total_seconds() / 60.0
            metrics.lead_times_minutes.append(lead_minutes)

    # False negatives: any failure not caught
    for machine_id, machine_labels in labels_by_machine.items():
        for label in machine_labels:
            if (machine_id, label.failure_time) not in caught_failures:
                metrics.n_false_negatives += 1

    # Compute machine-days
    if not alert_only and not labels:
        metrics.n_machine_days = 0.0
    else:
        all_timestamps = [a.timestamp for a in alert_only] + [
            label.failure_time for label in labels
        ]
        period_min = (
            (max(all_timestamps) - min(all_timestamps)).total_seconds() / 60.0
            if len(all_timestamps) > 1
            else 0.0
        )
        if evaluation_period_minutes is not None:
            period_min = evaluation_period_minutes
        n_m = n_machines or len({a.machine_id for a in alert_only} | set(labels_by_machine))
        metrics.n_machine_days = (period_min / (60 * 24)) * max(n_m, 1)

    return metrics.to_evaluation_result(model_name=model_name, threshold=threshold)
