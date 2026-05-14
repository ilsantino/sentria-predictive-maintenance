# Methodology

How models, features and evaluation work — separated from architecture
(which is about *where* code lives) and from evaluation results (which
do not exist yet because no real data has touched this system).

## 1. Features

The feature engineering layer (`sentria_pm.data.features`) converts a
wide-format DataFrame of raw sensor signals into a richer feature
frame by adding, **per machine**, time-since-now rolling statistics
and lag features.

| Feature family | Definition | Windows |
|---|---|---|
| Rolling mean | μ over the last k samples | k ∈ {5, 15, 60} |
| Rolling std | σ over the last k samples | k ∈ {5, 15, 60} |
| Rolling min/max | range bounds | k ∈ {5, 15, 60} |
| Lag features | x[t - k] | k ∈ {1, 5, 15} |
| Rolling z-score | (x[t] - μ_60) / σ_60 | 60 samples |
| Cross-sensor ratios | x_i / x_j (optional) | — |

All rolling features use `min_periods = window * 0.5` to allow partial
windows at the start of the trajectory rather than producing NaN.

**Warmup region**: the first `max_window` rows of every machine are
dropped via `RollingFeatureExtractor.drop_warmup` because they contain
mostly NaN-padded statistics that bias models toward thinking
"everything looks like a partial window".

**Why rolling z-score?** Sensor calibration drifts on the order of days.
A global z-score against training statistics ages poorly. Rolling
z-score against a recent window is stationary against slow
calibration drift while still flagging fast departures.

## 2. Models

### 2.1 Isolation Forest (`sentria_pm.models.isolation_forest`)

- **Type**: Tree-based density estimator.
- **Idea**: anomalies require few random splits to isolate.
- **Input**: the full engineered feature frame.
- **Hyperparameters**: ``n_estimators=100``, ``contamination="auto"``,
  ``max_samples="auto"``, ``random_state`` configurable.
- **Score normalization**: sklearn's ``-score_samples`` (higher =
  more anomalous) min-max scaled to [0, 1] per scoring call.

**Reference**: Liu, Ting, Zhou. *Isolation Forest.* ICDM 2008.

### 2.2 ARIMA per-sensor (`sentria_pm.models.arima_baseline`)

- **Type**: Univariate autoregressive forecast residual.
- **Idea**: forecast the next value of each sensor independently;
  large residual = anomaly. Aggregate across sensors with `max(|z|)`.
- **Input**: raw sensor columns (`S-XX`). Not engineered features —
  the model itself encapsulates the temporal model.
- **Order**: default ARIMA(1, 0, 0) (i.e., AR(1)) for speed. Higher
  orders configurable but expensive across the full machine fleet.
- **Fallback**: for constant or near-constant series where ARIMA
  cannot fit, the module uses a rolling z-score over a 20-step
  window — a deliberately uninformative baseline so the model
  never crashes on edge cases.

**Reference**: Box, Jenkins. *Time Series Analysis.* 5th ed., 2015.

### 2.3 LSTM Autoencoder (`sentria_pm.models.lstm_autoencoder`)

- **Type**: Seq2seq LSTM encoder-decoder, reconstruction-error scoring.
- **Idea**: train on (mostly normal) windows, score by how poorly the
  decoder reconstructs each window. High reconstruction error = unusual.
- **Architecture**:
  - Encoder: 1 LSTM layer (hidden_dim=32 by default).
  - Latent compression: linear `hidden_dim → latent_dim`.
  - Decoder: LSTM running autoregressively from a learned start
    token; output projected linearly to `n_features`.
- **Window size**: 16 samples by default.
- **Training**: Adam, MSE loss, 20 epochs, batch size 32.
- **Score attribution**: window error is attributed to the row at the
  **end** of the window (the row most recently observed).

**Reference**: Malhotra et al. *LSTM-based Encoder-Decoder for
Multi-sensor Anomaly Detection.* ICML AD Workshop 2016.

### 2.4 Ensemble (`sentria_pm.models.ensemble`)

Three fusion strategies are supported:

| Strategy | Formula | When to use |
|---|---|---|
| `mean` | (s_1 + s_2 + s_3) / 3 | Default; equal trust across detectors |
| `weighted` | Σ w_i · s_i / Σ w_i | When prior performance ranking is known |
| `max` | max(s_1, s_2, s_3) | When false negatives are more costly than false positives |

**Threshold calibration**: `EnsembleDetector.calibrate_threshold(features, q)`
sets the alert threshold to the **q-th quantile** of the fused score
distribution on training data, **per machine**. Default q = 0.99 →
top 1% of training scores are alerts in production.

This is a starting calibration that should be retuned once real alert
volume is measured against operator feedback.

## 3. Evaluation

### 3.1 Event-level metrics

Definitions (see `sentria_pm.evaluation.metrics`):

- **True positive (TP)**: an alert at time `t` such that there exists
  a failure label `F` for the same machine with
  `t ∈ [F.failure_time - L, F.failure_time]` where `L` is the
  lookback window (default: 180 minutes). Each failure counts as **at
  most one TP** regardless of how many alerts fall in its window —
  preventing alert-storm gaming.
- **False positive (FP)**: an alert that does not match any failure
  in lookback.
- **False negative (FN)**: a failure for which no alert appeared in
  its lookback window.
- **Precision, recall, F1**: standard formulae applied to TP, FP, FN.
- **Lead-time-before-failure**: time difference between the first TP
  alert and the failure. Reported as p50 and p95 over the failures
  the model caught.
- **False-alarm rate per machine-day**: FP / (n_machines × n_days).

### 3.2 Walk-forward evaluation

Walk-forward splitting (`sentria_pm.evaluation.walk_forward`) produces
`n_folds` train/test splits where:

- Each split's train set ends before its test set begins (no future
  leak).
- Each split's training fraction grows monotonically with fold
  index (more data available later).
- Per-machine splits are then concatenated, so no machine's future
  leaks into another machine's training either.

### 3.3 Ablations to run (Q3 2026 deliverable)

Each baseline isolated:

1. Isolation Forest only.
2. ARIMA only.
3. LSTM AE only.
4. Full ensemble.

Feature ablations:

5. Without environmental variables.
6. Without incident-frequency signal.
7. Without cross-sensor ratios.
8. Sensitivity to rolling window size (5 only / 15 only / 60 only / all).

Window/threshold ablations:

9. Lookback windows of 60, 120, 240 minutes.
10. Threshold calibration quantiles of 0.90, 0.95, 0.99.

Each ablation produces a row in the model card with precision, recall,
F1, lead-time p50/p95, false-alarm rate.

## 4. Limitations

Documented because honesty about limitations is part of the deliverable:

1. **Label noise.** Failure labels come from operator reports and may
   be mis-categorized or missing entirely (under-reporting of minor
   failures). The model trains on noisy labels and inherits their bias.
2. **Censoring.** Preventive interventions that avoided a failure
   appear as "non-events" but actually carry strong signal. The
   current design has no way to use them.
3. **Drift.** Distribution shift between training and deployment
   (lubricant change, SKU change, seasonal effects) requires
   periodic re-training. Frequency to be determined empirically.
4. **Generalization.** Trained on three pilot lines; transferring
   to other lines requires per-line calibration at minimum.
5. **Read-only by design.** The model cannot prevent failures it
   detects — only inform the operator who must act. Speed of operator
   response is a bottleneck the model cannot reduce.

## 5. References

- Box, G. E. P., Jenkins, G. M. *Time Series Analysis: Forecasting and
  Control.* 5th ed., 2015.
- Hundman, K. et al. *Detecting Spacecraft Anomalies Using LSTMs and
  Nonparametric Dynamic Thresholding.* KDD 2018.
- Liu, F. T., Ting, K. M., Zhou, Z. H. *Isolation Forest.* ICDM 2008.
- Malhotra, P. et al. *LSTM-based Encoder-Decoder for Multi-sensor
  Anomaly Detection.* ICML AD Workshop 2016.
- Pang, G. et al. *Deep Learning for Anomaly Detection: A Review.*
  ACM Computing Surveys, 2021.
- Saxena, A. et al. *Damage propagation modeling for aircraft engine
  run-to-failure simulation.* PHM 2008. (NASA C-MAPSS dataset; design
  inspiration for the synthetic generator.)
