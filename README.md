# SENTRIA Predictive Maintenance

> Anomaly detection layer for industrial production lines. Read-only
> alerting (no closed-loop control). Ensemble of Isolation Forest, LSTM
> autoencoder, and ARIMA baseline over multivariate sensor telemetry.

[![CI](https://github.com/ilsantino/sentria-predictive-maintenance/actions/workflows/ci.yml/badge.svg)](https://github.com/ilsantino/sentria-predictive-maintenance/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

**Status:** Spec + working skeleton on synthetic data. Pilot execution Q3 2026.
**Cliente piloto:** Absara (extensión del despliegue SENTRIA core).
**Maintainer:** Santiago Álvarez · [iaGO Agency Group](https://github.com/ilsantino)

---

## What this is

The SENTRIA core multi-agent system is already in production on Absara's
plant (1,700 employees, 31 lines, 80+ users). It logs incidents in real
time and routes them via WhatsApp to the responsible operator.

This repository is the predictive-maintenance extension: a system that
detects anomalies on multivariate machine telemetry **before** an
incident is reported, with enough lead-time for the operator to act.

It operates in **read-only mode** by design — alerts only, no automated
shutdowns. Reasons documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Current state

This repository contains:

- **A working Python package** (`sentria_pm`) with three baselines, an
  ensemble, evaluation metrics, and a CLI. All 55 tests passing.
- **A synthetic telemetry generator** inspired by NASA C-MAPSS, used
  for testing the pipeline end-to-end before real data is available.
- **Full technical specification** for the Q3 2026 pilot in
  [`docs/`](docs/): architecture, methodology, evaluation plan, data
  schema, roadmap.

What this **is not**:

- Not trained on real client data (Q3 2026 pilot deliverable).
- Not deployed to AWS (pilot deliverable).
- Not a UI (operators consume alerts through SENTRIA core's existing
  WhatsApp flow).

## Quickstart

```bash
# Install
git clone https://github.com/ilsantino/sentria-predictive-maintenance.git
cd sentria-predictive-maintenance
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the full test suite
pytest

# Generate synthetic telemetry and run end-to-end evaluation
sentria-pm evaluate --n-machines 10 --trajectory-length 200
```

Example output:

```
       Evaluation result
+------------------------------+
| Metric              | Value  |
|---------------------+--------|
| Alerts              | 25     |
| True positives      | 2      |
| False positives     | 15     |
| Lead time p50 (min) | 22.5   |
| Lead time p95 (min) | 24.8   |
+------------------------------+
```

## Architecture overview

```
Sensors + incidents + env data
            │
            ▼
   ETL → S3 parquet (partitioned by machine + day)
            │
            ▼
   Feature engineering (rolling stats, lags, z-score)
            │
            ▼
   ┌────────────────────────────────────────┐
   │ Ensemble of three heterogeneous models │
   │                                        │
   │  Isolation Forest │ LSTM AE │ ARIMA    │
   │     (sklearn)    (PyTorch) (statsmodels)│
   │                                        │
   │  Fusion: mean / weighted / max         │
   │  Threshold: per-machine quantile       │
   └────────────────────────────────────────┘
            │
            ▼
   Read-only alert → WhatsApp via SENTRIA core
```

Full architecture in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Methodology highlights

- **Event-level evaluation** (not point-wise): one TP per failure event
  regardless of alert count in window. Prevents alert-storm gaming.
- **Walk-forward evaluation**: no future data leaks into training.
- **Per-machine threshold calibration** at a quantile of training-score
  distribution — each line gets a budget proportional to its own
  variability.
- **Heterogeneous ensemble**: density-based (Isolation Forest),
  reconstruction-based (LSTM AE), and forecast-residual (ARIMA).
  Designed to fail in different ways so consensus = high confidence.
- **Read-only by design** — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
  for the operational and liability reasoning.

Full methodology in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | High-level system, module map, design decisions |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Features, models, evaluation framework |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Hypotheses, metrics, ablation matrix, decision rule |
| [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) | Authoritative Pydantic contracts |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Q2 2026 (done) → Q3 (pilot) → Q4 (model card) → Q1 2027 |

## CLI

```bash
sentria-pm version              # print version
sentria-pm generate             # write a synthetic CSV
sentria-pm evaluate             # generate + fit + score + metrics table
sentria-pm score <input.csv>    # score a telemetry CSV
```

Full help: `sentria-pm --help`, `sentria-pm <subcommand> --help`.

## Project layout

```
sentria-predictive-maintenance/
├── src/sentria_pm/
│   ├── data/         # Pydantic schemas, synthetic generator, features
│   ├── models/       # Isolation Forest, ARIMA, LSTM AE, ensemble
│   ├── evaluation/   # Event-level metrics + walk-forward harness
│   ├── utils/        # Rich logging
│   ├── cli.py        # Typer CLI
│   └── __main__.py
├── tests/            # 55 tests across all modules
├── docs/             # Architecture, methodology, evaluation plan
├── .github/workflows/ci.yml   # Pytest on 3.10/3.11/3.12 + ruff + mypy
└── pyproject.toml
```

## Tech stack

- **Python 3.10+**, packaging via `hatchling`.
- **scikit-learn** for Isolation Forest.
- **statsmodels** for ARIMA.
- **PyTorch** for the LSTM autoencoder.
- **Pydantic v2** for data contracts.
- **typer + rich** for the CLI.
- **pytest + coverage** for testing.
- **ruff + mypy** for linting and type-checking.

## How this connects to SENTRIA core

The predictive-maintenance layer is **additive** to the existing
SENTRIA deployment:

- It reads the incident stream that SENTRIA core writes to DynamoDB.
- It pushes alerts through the same WhatsApp pipeline that operators
  already use.
- Operators do not learn a new tool. The PM layer either earns its
  place in their workflow or it gets ignored.

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Santiago Álvarez Chamlati** — Ingeniero de IA Aplicada
iaGO Agency Group · [LinkedIn](https://www.linkedin.com/in/sanalvarezch) · [github.com/ilsantino](https://github.com/ilsantino)
