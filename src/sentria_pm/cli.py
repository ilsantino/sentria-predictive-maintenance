"""Command-line interface for SENTRIA PM.

Subcommands
-----------
- ``generate``: produce a synthetic dataset and write it as CSV.
- ``train``: fit the ensemble on a feature frame.
- ``evaluate``: run end-to-end on the synthetic dataset and print metrics.
- ``score``: read a feature frame and emit per-row anomaly scores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from sentria_pm.data.features import FeatureConfig, RollingFeatureExtractor
from sentria_pm.data.schemas import SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator
from sentria_pm.evaluation.metrics import compute_metrics
from sentria_pm.models.ensemble import EnsembleDetector
from sentria_pm.models.isolation_forest import IsolationForestDetector

app = typer.Typer(
    help="SENTRIA Predictive Maintenance — CLI for synthetic generation, training and evaluation.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def generate(
    output: Path = typer.Option(Path("synthetic.csv"), help="Where to write the CSV."),
    n_machines: int = typer.Option(5),
    n_sensors: int = typer.Option(8),
    trajectory_length: int = typer.Option(200),
    seed: int = typer.Option(42),
) -> None:
    """Generate a synthetic telemetry dataset and write it to CSV."""
    cfg = SyntheticConfig(
        n_machines=n_machines,
        n_sensors=n_sensors,
        trajectory_length=trajectory_length,
        seed=seed,
    )
    gen = SyntheticTelemetryGenerator(cfg)
    df, labels = gen.generate()
    df.to_csv(output, index=False)
    console.print(
        f"[green]Wrote[/green] {len(df):,} rows for {n_machines} machines "
        f"to {output} ({len(labels)} failure labels)."
    )


@app.command()
def evaluate(
    n_machines: int = typer.Option(10),
    trajectory_length: int = typer.Option(200),
    threshold_quantile: float = typer.Option(0.95, min=0.5, max=0.9999),
    lookback_minutes: float = typer.Option(180.0),
    seed: int = typer.Option(42),
) -> None:
    """Generate synthetic data, fit Isolation Forest baseline, evaluate end-to-end."""
    cfg = SyntheticConfig(
        n_machines=n_machines,
        trajectory_length=trajectory_length,
        seed=seed,
    )
    console.print("[cyan]Generating synthetic telemetry...[/cyan]")
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, labels = gen.generate()
    wide_df = gen.to_wide(long_df)

    console.print("[cyan]Building rolling features...[/cyan]")
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5, 15)))
    features = extractor.fit_transform(wide_df)
    features = RollingFeatureExtractor.drop_warmup(features, warmup=15)
    features = RollingFeatureExtractor.fill_remaining(features)

    console.print("[cyan]Fitting Isolation Forest...[/cyan]")
    detector = IsolationForestDetector(n_estimators=100, random_state=seed)
    ensemble = EnsembleDetector([detector])
    ensemble.fit(features)
    ensemble.calibrate_threshold(features, target_false_alarm_quantile=threshold_quantile)

    console.print("[cyan]Scoring...[/cyan]")
    alerts = ensemble.alerts(features)
    result = compute_metrics(
        alerts=alerts,
        labels=labels,
        lookback_window_minutes=lookback_minutes,
        n_machines=n_machines,
        threshold=ensemble.default_threshold,
        model_name="isolation_forest",
    )

    table = Table(title="Evaluation result")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Alerts", str(result.n_alerts))
    table.add_row("True positives", str(result.n_true_positives))
    table.add_row("False positives", str(result.n_false_positives))
    table.add_row("False negatives", str(result.n_false_negatives))
    table.add_row("Precision", f"{result.precision:.3f}")
    table.add_row("Recall", f"{result.recall:.3f}")
    table.add_row("F1", f"{result.f1:.3f}")
    if result.lead_time_minutes_p50 is not None:
        table.add_row("Lead time p50 (min)", f"{result.lead_time_minutes_p50:.1f}")
    if result.lead_time_minutes_p95 is not None:
        table.add_row("Lead time p95 (min)", f"{result.lead_time_minutes_p95:.1f}")
    table.add_row(
        "False alarm rate (/machine-day)",
        f"{result.false_alarm_rate_per_machine_day:.3f}",
    )
    console.print(table)


@app.command()
def score(
    input_csv: Path = typer.Argument(..., help="CSV with long-format telemetry."),
    output_csv: Path = typer.Option(Path("scores.csv"), help="Where to write per-row scores."),
    seed: int = typer.Option(42),
) -> None:
    """Score a CSV of telemetry using a fresh-fit Isolation Forest."""
    long_df = pd.read_csv(input_csv, parse_dates=["timestamp"])
    gen = SyntheticTelemetryGenerator()
    wide_df = gen.to_wide(long_df)

    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5, 15)))
    features = extractor.fit_transform(wide_df)
    features = RollingFeatureExtractor.drop_warmup(features, warmup=15)
    features = RollingFeatureExtractor.fill_remaining(features)

    detector = IsolationForestDetector(n_estimators=100, random_state=seed)
    ensemble = EnsembleDetector([detector])
    ensemble.fit(features)
    ensemble.calibrate_threshold(features)

    alerts = ensemble.alerts(features)
    out = pd.DataFrame(
        [
            {
                "timestamp": a.timestamp,
                "machine_id": a.machine_id,
                "score": a.score,
                "threshold": a.threshold,
                "alert": a.alert,
            }
            for a in alerts
        ]
    )
    out.to_csv(output_csv, index=False)
    console.print(
        f"[green]Wrote[/green] {len(out):,} scored rows to {output_csv} "
        f"({sum(out['alert']):,} alerts)."
    )


@app.command()
def version() -> None:
    """Print package version."""
    from sentria_pm import __version__

    console.print(f"sentria-pm v{__version__}")


if __name__ == "__main__":
    app()
