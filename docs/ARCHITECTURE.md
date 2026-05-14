# Architecture

## High-level diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                       Data sources                                │
│                                                                   │
│   ┌─────────────┐   ┌─────────────────┐   ┌──────────────────┐    │
│   │ Telemetría  │   │ Frecuencia de   │   │ Variables        │    │
│   │ de máquina  │   │ incidencias por │   │ ambientales      │    │
│   │ (sensores)  │   │ línea (SENTRIA) │   │ (temp/hum/lote)  │    │
│   └─────┬───────┘   └────────┬────────┘   └────────┬─────────┘    │
│         │                    │                     │              │
└─────────┼────────────────────┼─────────────────────┼──────────────┘
          │                    │                     │
          ▼                    ▼                     ▼
┌───────────────────────────────────────────────────────────────────┐
│   Ingesta + ETL: Lambda → DynamoDB Streams → S3 parquet           │
│   (particionado por (machine_id, day))                            │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│   Feature engineering (`sentria_pm.data.features`)                │
│   - Rolling stats {5, 15, 60} samples                             │
│   - Lag features {1, 5, 15}                                       │
│   - Rolling z-score (60 samples)                                  │
│   - Optional cross-sensor ratios                                  │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│   Ensemble (`sentria_pm.models.ensemble`)                         │
│                                                                   │
│   ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│   │ IsolationForest  │  │ LSTM Autoencoder │  │ ARIMA(1,0,0)   │  │
│   │ (sklearn)        │  │ (PyTorch)        │  │ (statsmodels)  │  │
│   │ density-based    │  │ reconstruction   │  │ residual zscore│  │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬───────┘  │
│            │                     │                     │          │
│            └────────────┬────────┴─────────┬───────────┘          │
│                         ▼                  ▼                      │
│              Score fusion (mean / weighted / max)                 │
│                            │                                      │
│                            ▼                                      │
│        Per-machine threshold (calibrated at quantile)             │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────┐
│   Alert delivery (read-only) — WhatsApp / SES → operador          │
│   No shutdown automatizado. Operador valida y actúa.              │
└───────────────────────────────────────────────────────────────────┘
```

## Module map

```
src/sentria_pm/
├── data/
│   ├── schemas.py          # Pydantic contracts (telemetry, alert, label)
│   ├── synthetic.py        # SyntheticTelemetryGenerator
│   └── features.py         # RollingFeatureExtractor + drop_warmup helpers
├── models/
│   ├── base.py             # AnomalyDetector ABC
│   ├── isolation_forest.py # IsolationForestDetector
│   ├── arima_baseline.py   # ARIMABaseline
│   ├── lstm_autoencoder.py # LSTMAutoencoderDetector
│   └── ensemble.py         # EnsembleDetector + threshold calibration
├── evaluation/
│   ├── metrics.py          # Event-level precision/recall/F1, lead-time
│   └── walk_forward.py     # Walk-forward train/test splitting
├── utils/
│   └── logging.py          # Rich logging configuration
├── cli.py                  # Typer CLI: generate / evaluate / score / version
└── __main__.py             # python -m sentria_pm
```

## Design decisions and trade-offs

### Why an ensemble of three?

Each model captures a different anomaly signature:

- **Isolation Forest** is density-based. Strong when the anomaly is a
  rare combination of feature values (high temperature AND high
  vibration). Cheap and stable. Default contamination set to ``"auto"``
  to avoid forcing a prior.
- **LSTM autoencoder** captures temporal coupling. Strong when the
  failure manifests as a change in the time-correlation structure
  between sensors (e.g., bearing degradation breaks the normal
  phase relationship between vibration and current draw).
- **ARIMA baseline** is the simple-but-strong reference. Per-sensor
  one-step-ahead residuals normalized by training residual std. Hard
  to beat with much more complex models on calm operating regimes.

The ensemble is **deliberately heterogeneous**: if all three models
agree, the alert is high-confidence; if only one fires, the operator
gets attribution via ``component_scores`` and can judge.

### Why read-only?

Shutting down a production line on a model output is a high-stakes
action. Three reasons we operate in alert-only mode:

1. **Liability containment.** A false positive that triggers a shutdown
   costs the plant tens of thousands of pesos. We do not own that
   risk while the model is uncalibrated.
2. **Operator trust.** Operators need to see the model agree with
   their intuition repeatedly before they accept its recommendations.
   Closed-loop control without trust gets disabled the first time the
   model is wrong.
3. **Verifiability.** Read-only operation produces a labelable stream
   of (alert, outcome) pairs without confounding from the model's own
   intervention. Closed-loop control would create a causal feedback
   loop that's much harder to evaluate.

### Why event-level scoring instead of point-wise?

A model that fires three alerts in the 30 minutes before a failure is
NOT three times better than a model that fires one. Both caught the
failure. Event-level scoring (one TP per failure event, lead-time from
the first qualifying alert) matches the operational reality and
prevents alert-storms from gaming the precision metric.

This follows the convention used in NASA C-MAPSS prognostic metrics
(Saxena 2008) and in the broader PdM literature.

### Why walk-forward and not k-fold?

Time-series data has order. Random k-fold leaks future information
into training and overstates model performance. Walk-forward respects
the temporal ordering: train on past, evaluate on the immediate future,
slide the window. Standard practice for any time-series ML evaluation.

### Why per-machine thresholds?

Different lines have different operating regimes (line 1 runs at
2,000 RPM, line 12 at 800 RPM). A global threshold either over-alerts
on the noisy line or under-alerts on the quiet one. Per-machine
calibration at a quantile of the training-score distribution gives
each line a budget of alerts proportional to its own normal variability.

### Why all detectors share the same feature frame?

To make the ensemble auditable. If two detectors look at different
inputs, attribution becomes ambiguous. By forcing them to read the
same features, the only thing that differs is the *model*, which is
what we want to compare.

## Data flow contracts

All cross-module data is one of the Pydantic models in
``sentria_pm.data.schemas``. Specifically:

- **`TelemetryReading`**: one sensor reading at one timestamp. Long
  format. Source-of-truth.
- **`Incident`**: an incident logged by SENTRIA core. May or may not
  correspond to a failure.
- **`FailureLabel`**: a confirmed failure event. Ground truth for
  evaluation.
- **`AnomalyAlert`**: output of a single detector at a single timestamp.
  Score in [0, 1].
- **`EnsembleScore`**: fused output. Has both the score and the
  alert flag plus per-component contributions.
- **`EvaluationResult`**: aggregated metrics over an evaluation run.

These are **frozen** (immutable) by configuration: once produced, they
cannot be mutated. This makes pipeline stages safe to compose without
worrying about hidden side-effects.

## Where this connects to the SENTRIA core

The SENTRIA multi-agent system already running in Absara provides:

- A structured incident stream (one of the ingestion sources).
- The operator notification channel (WhatsApp via Twilio).
- The historical incident archive (DynamoDB, partitioned by machine).

The predictive maintenance layer is **additive**: it reads the same
incident stream as a feature, it pushes alerts through the same
operator channel. Operators do not learn a new tool. The model
either earns its place in their workflow or it gets ignored.

## What this repository is NOT

- **Not production-ready.** This is a Q3 2026 spec + working skeleton.
  No real client data has touched it yet.
- **Not a Lambda deployment.** AWS deployment manifests come during
  pilot execution.
- **Not a UI.** Operator-facing UI lives in SENTRIA core.
- **Not a labeling tool.** Ground-truth labels come from the SENTRIA
  incident database with operator confirmation.
