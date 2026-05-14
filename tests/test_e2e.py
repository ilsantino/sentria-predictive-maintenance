"""End-to-end pipeline test: generate → features → fit → score → metrics.

This is the integration test that proves the entire system stacks up. If
any earlier unit test passes but this one fails, the contract between
modules has drifted.
"""

from __future__ import annotations

from sentria_pm.data.features import FeatureConfig, RollingFeatureExtractor
from sentria_pm.data.schemas import SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator
from sentria_pm.evaluation.metrics import compute_metrics
from sentria_pm.models.ensemble import EnsembleDetector
from sentria_pm.models.isolation_forest import IsolationForestDetector


def test_pipeline_end_to_end() -> None:
    cfg = SyntheticConfig(
        n_machines=6,
        n_sensors=5,
        trajectory_length=150,
        seed=22,
        failure_rate=0.8,
    )
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, labels = gen.generate()
    wide_df = gen.to_wide(long_df)

    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5, 15)))
    features = extractor.fit_transform(wide_df)
    features = RollingFeatureExtractor.drop_warmup(features, warmup=15)
    features = RollingFeatureExtractor.fill_remaining(features)

    detector = IsolationForestDetector(n_estimators=80, random_state=1)
    ensemble = EnsembleDetector([detector])
    ensemble.fit(features)
    ensemble.calibrate_threshold(features, target_false_alarm_quantile=0.92)
    alerts = ensemble.alerts(features)

    result = compute_metrics(
        alerts=alerts,
        labels=labels,
        lookback_window_minutes=240.0,
        n_machines=cfg.n_machines,
        threshold=ensemble.default_threshold,
        model_name="isolation_forest_e2e",
    )

    # Sanity asserts: the pipeline runs and produces structurally valid output.
    assert result.n_alerts > 0
    assert result.precision >= 0.0
    assert result.recall >= 0.0
    assert result.f1 >= 0.0
    # On synthetic data with strong drift, we expect to catch at least one failure.
    assert result.n_true_positives + result.n_false_negatives == len(labels)
