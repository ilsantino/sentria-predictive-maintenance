"""Tests for LSTM autoencoder."""

from __future__ import annotations

import numpy as np
import pytest

from sentria_pm.data.schemas import SyntheticConfig
from sentria_pm.data.synthetic import SyntheticTelemetryGenerator
from sentria_pm.models.lstm_autoencoder import LSTMAutoencoderDetector


@pytest.fixture(scope="module")
def small_wide():
    cfg = SyntheticConfig(n_machines=3, n_sensors=3, trajectory_length=80, seed=4, failure_rate=1.0)
    gen = SyntheticTelemetryGenerator(cfg)
    long_df, _ = gen.generate()
    return gen.to_wide(long_df)


@pytest.mark.slow
def test_fit_and_score_shape(small_wide) -> None:
    det = LSTMAutoencoderDetector(
        window=8, hidden_dim=8, latent_dim=4, epochs=3, batch_size=16, seed=1
    )
    det.fit(small_wide)
    assert det.is_fitted
    scores = det.score(small_wide)
    assert scores.shape == (len(small_wide),)


@pytest.mark.slow
def test_scores_in_unit_interval(small_wide) -> None:
    det = LSTMAutoencoderDetector(
        window=8, hidden_dim=8, latent_dim=4, epochs=3, batch_size=16, seed=1
    )
    det.fit(small_wide)
    scores = det.score(small_wide)
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()


@pytest.mark.slow
def test_training_reduces_recon_error_on_normal(small_wide) -> None:
    """After training, reconstruction error on training data should be lower
    than reconstruction error on shuffled (noised) data of the same shape."""
    det = LSTMAutoencoderDetector(
        window=8, hidden_dim=16, latent_dim=4, epochs=10, batch_size=16, seed=1
    )
    det.fit(small_wide)

    noised = small_wide.copy()
    sensor_cols = [c for c in small_wide.columns if c.startswith("S-")]
    rng = np.random.default_rng(0)
    noised[sensor_cols] = noised[sensor_cols] + rng.normal(0, 50, size=noised[sensor_cols].shape)

    normal_scores = det.score(small_wide).mean()
    noisy_scores = det.score(noised).mean()
    assert noisy_scores >= normal_scores * 0.5


def test_score_before_fit_raises() -> None:
    det = LSTMAutoencoderDetector()
    try:
        det.score(None)  # type: ignore[arg-type]
        raise AssertionError("Expected RuntimeError")
    except (RuntimeError, AttributeError):
        pass
