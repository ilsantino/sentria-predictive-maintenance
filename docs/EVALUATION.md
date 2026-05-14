# Evaluation Plan

This document fixes evaluation methodology *before* real data is
available. The Q3 2026 pilot will produce a model card following
exactly this template.

## Hypotheses

H1
:   The ensemble of three heterogeneous detectors achieves F1 strictly
    greater than the best single detector at the same threshold.

H2
:   Median lead-time-before-failure is at least 30 minutes for caught
    failures.

H3
:   False-alarm rate per machine-day is below 1.0 across the pilot
    lines once threshold is calibrated at q=0.99.

H4
:   Cross-sensor coupling features (ratios) improve recall by at least
    five percentage points for bearing-type failures versus the same
    pipeline without them.

H5
:   Drift between training (months 1-2) and deployment (months 3-6)
    requires re-training to maintain F1 within 10% of initial value.
    Frequency to be determined empirically.

Each hypothesis becomes a test that either passes or fails in the model
card and shapes the production decision.

## Metrics

| Metric | Definition | Target |
|---|---|---|
| Precision (event) | TP / (TP + FP) | ≥ 0.60 |
| Recall (event) | TP / (TP + FN) | ≥ 0.50 |
| F1 (event) | 2·P·R / (P + R) | ≥ 0.55 |
| Lead-time p50 | Median minutes from first TP to failure | ≥ 30 |
| Lead-time p95 | 95th percentile lead time | report |
| FP rate | FP / (n_machines × n_days) | ≤ 1.0 |

Targets are **initial** — to be revised after the pilot reveals what
is achievable on this specific plant.

## Datasets

| Dataset | Source | Use |
|---|---|---|
| `synthetic-v1` | `SyntheticTelemetryGenerator` (NASA C-MAPSS inspired) | Pipeline integration tests, design validation |
| `pilot-train` | First two months of pilot data | Model training |
| `pilot-val` | Month 3 of pilot data | Threshold calibration |
| `pilot-test` | Months 4-6 of pilot data | Headline metrics for the model card |

Walk-forward splits within `pilot-train` are used for hyperparameter
tuning. No tuning ever touches `pilot-test`.

## Ablation matrix

The model card reports the following ablation rows:

| # | Configuration |
|---|---|
| 1 | Isolation Forest only |
| 2 | ARIMA only |
| 3 | LSTM Autoencoder only |
| 4 | Ensemble (mean fusion) |
| 5 | Ensemble (weighted fusion, weights from val) |
| 6 | Ensemble (max fusion) |
| 7 | Ensemble without environmental features |
| 8 | Ensemble without incident-frequency feature |
| 9 | Ensemble without cross-sensor ratios |
| 10 | Ensemble with single rolling window (15 only) |
| 11 | Lookback window 60 min |
| 12 | Lookback window 240 min |
| 13 | Threshold quantile 0.90 |
| 14 | Threshold quantile 0.95 |

For each row, the table reports precision, recall, F1, lead-time p50,
lead-time p95, and FP rate. The same `pilot-test` data is used
throughout — no data snooping into other ablations' tuning.

## Failure mode analysis

For each false positive and false negative in the test set:

- **For false positives**: which detector(s) contributed most to the
  score? What was happening on the line at that timestamp? Cluster the
  FPs into recognizable patterns (e.g., "30% of FPs are at end-of-shift
  cleaning cycles").
- **For false negatives**: was the failure visible in any sensor
  channel in the lookback window? If yes, *why* did the model miss it?
  If no, the data itself was uninformative for that failure mode and
  no model with the same inputs could have caught it.

The failure-mode analysis is the most informative part of the model
card. It is where future work (sensors to add, features to engineer,
operator workflow changes) lives.

## Operator-facing evaluation

The pilot will also collect operator feedback per alert:

- **Useful**: alert led to a check or pre-emptive action.
- **Already known**: operator was already aware.
- **False alarm**: nothing wrong.
- **Confusing**: alert fired but contributing features made no sense.

These categories are tracked separately from the offline TP/FP/FN
labels because they measure *operational value*, not just statistical
correctness. A model with high F1 but high "Confusing" rate will be
ignored.

## Decision rule

At the end of the pilot, the model goes to one of three states:

1. **Production**: F1 ≥ 0.55, FP rate ≤ 1.0, "Confusing" rate < 20%.
   The alert flow stays on; we move to Phase 2 (additional lines).
2. **Iteration**: one of the above thresholds missed but failure-mode
   analysis points to a fixable cause. We do one more round of
   feature engineering / model adjustment and re-evaluate on a fresh
   test slice.
3. **Shelve**: model fails twice. The data does not support a useful
   predictive maintenance layer on these lines with this sensor
   instrumentation. Recommend instrumentation upgrade or different
   problem framing.

The shelve decision is **deliberately included**. Predictive maintenance
is not always the right answer. Having a clear "shelve" rule prevents
sunk-cost commitment to a model that does not deliver value.
