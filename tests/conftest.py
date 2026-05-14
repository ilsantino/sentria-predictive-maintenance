"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from sentria_pm.data.features import FeatureConfig, RollingFeatureExtractor
from sentria_pm.data.schemas import FailureLabel, SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator


@pytest.fixture(scope="module")
def small_synthetic() -> tuple[pd.DataFrame, list[FailureLabel]]:
    """A small reproducible synthetic dataset suitable for fast tests."""
    cfg = SyntheticConfig(
        n_machines=4,
        n_sensors=4,
        trajectory_length=120,
        seed=11,
        failure_rate=0.75,
    )
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, labels = gen.generate()
    return long_df, labels


@pytest.fixture(scope="module")
def wide_synthetic(small_synthetic) -> pd.DataFrame:
    long_df, _ = small_synthetic
    gen = SyntheticTelemetryGenerator()
    return gen.to_wide(long_df)


@pytest.fixture(scope="module")
def features_synthetic(wide_synthetic) -> pd.DataFrame:
    """Engineered feature frame ready for modeling."""
    extractor = RollingFeatureExtractor(FeatureConfig(rolling_windows=(5, 15)))
    features = extractor.fit_transform(wide_synthetic)
    features = RollingFeatureExtractor.drop_warmup(features, warmup=15)
    features = RollingFeatureExtractor.fill_remaining(features)
    return features
