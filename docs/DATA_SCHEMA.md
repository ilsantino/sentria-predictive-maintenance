# Data Schema

Authoritative spec for every data type that crosses module boundaries.

## TelemetryReading

One sensor reading at one point in time. Long-format. Source-of-truth
input to the pipeline.

| Field | Type | Constraints |
|---|---|---|
| `timestamp` | `datetime` | UTC; monotonic per (machine, sensor) |
| `machine_id` | `str` | 1-64 chars; production convention `M-XXX` |
| `sensor_id` | `str` | 1-64 chars; production convention `S-XX` |
| `value` | `float` | Must be finite — no NaN, no Inf |
| `unit` | `str?` | Optional; max 16 chars (e.g., "°C", "bar", "rpm") |

**Notes**: Frozen / immutable. Use `model_copy(update={...})` for
derived rows. Multiple readings per timestamp are expected (one per
sensor channel); the wide-format pivot collapses them by `(timestamp,
machine_id)`.

## Incident

An incident logged in SENTRIA core. May or may not correspond to a
real failure.

| Field | Type | Constraints |
|---|---|---|
| `incident_id` | `str` | Unique within SENTRIA |
| `timestamp` | `datetime` | When the incident was reported |
| `machine_id` | `str` | Matches a `TelemetryReading.machine_id` |
| `severity` | `IncidentSeverity` | `minor` / `major` / `critical` |
| `description` | `str` | Up to 2,048 chars |
| `failure_type` | `FailureType?` | Optional; only set after reconciliation |
| `is_unplanned_stop` | `bool` | True iff the incident produced unplanned downtime |

## FailureLabel

Ground truth. Used to compute precision / recall and
lead-time-before-failure.

| Field | Type | Constraints |
|---|---|---|
| `machine_id` | `str` | — |
| `failure_time` | `datetime` | Confirmed failure timestamp |
| `failure_type` | `FailureType` | `bearing`/`thermal`/`electrical`/`lubrication`/`unknown` |
| `downtime_minutes` | `float` | ≥ 0 |
| `notes` | `str` | Free-form |

## AnomalyAlert

Output of a single model at a single point in time.

| Field | Type | Constraints |
|---|---|---|
| `timestamp` | `datetime` | — |
| `machine_id` | `str` | — |
| `score` | `float` | In `[0.0, 1.0]`; 1.0 = max anomaly |
| `model_name` | `str` | Identifies which detector produced it |
| `contributing_features` | `list[str]` | Top-k feature names contributing |
| `metadata` | `dict[str, float]` | Detector-specific numeric metadata |

## EnsembleScore

Fused decision from multiple detectors. The output type the alert
delivery layer consumes.

| Field | Type | Constraints |
|---|---|---|
| `timestamp` | `datetime` | — |
| `machine_id` | `str` | — |
| `score` | `float` | Fused score in `[0, 1]` |
| `threshold` | `float` | Per-machine threshold in `[0, 1]` |
| `alert` | `bool` | Must equal `score >= threshold` (cross-validated) |
| `component_scores` | `dict[str, float]` | Per-detector scores in `[0, 1]` |

## EvaluationResult

Aggregated metrics for one model over one evaluation run.

| Field | Type |
|---|---|
| `model_name` | `str` |
| `n_alerts` | `int` |
| `n_true_positives` | `int` |
| `n_false_positives` | `int` |
| `n_false_negatives` | `int` |
| `precision` | `float ∈ [0, 1]` |
| `recall` | `float ∈ [0, 1]` |
| `f1` | `float ∈ [0, 1]` |
| `lead_time_minutes_p50` | `float?` (None if no TPs) |
| `lead_time_minutes_p95` | `float?` |
| `false_alarm_rate_per_machine_day` | `float ≥ 0` |
| `threshold` | `float ∈ [0, 1]` |

## SyntheticConfig

Configuration for the synthetic telemetry generator.

| Field | Default | Constraints |
|---|---|---|
| `n_machines` | 5 | ≥ 1 |
| `n_sensors` | 8 | ≥ 1 |
| `trajectory_length` | 200 | ≥ 10 |
| `failure_rate` | 0.6 | `[0.0, 1.0]` |
| `drift_start_fraction` | 0.7 | `[0.0, 1.0]` |
| `noise_std` | 0.1 | > 0.0 |
| `seed` | 42 | int |
| `sampling_interval_minutes` | 5.0 | > 0.0 |
