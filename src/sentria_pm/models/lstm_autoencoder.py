"""LSTM Autoencoder anomaly detector.

Architecture
------------
- Encoder LSTM compresses a window of multivariate observations to a
  latent vector of size ``latent_dim``.
- Decoder LSTM reconstructs the window from the latent vector.
- Anomaly score = mean squared reconstruction error per window.

Training is done on rolling windows from the (assumed-mostly-normal)
training data. At score time, each input row is the *end* of a
window; the reconstruction error of that window is attributed to the row.

References
----------
- Malhotra, P. et al. *LSTM-based Encoder-Decoder for Multi-sensor
  Anomaly Detection.* ICML Anomaly Detection Workshop 2016.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from sentria_pm.models.base import AnomalyDetector


class _Seq2SeqLSTMAE(nn.Module):
    """Simple LSTM encoder/decoder. Decoder runs autoregressively from
    the latent state with a learned start-token."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int, window: int) -> None:
        super().__init__()
        self.window = window
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_output = nn.Linear(hidden_dim, n_features)
        self.start_token = nn.Parameter(torch.zeros(1, 1, n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window, features)
        batch = x.shape[0]
        _, (h, c) = self.encoder(x)
        latent = self.to_latent(h[-1])  # (batch, latent)
        h0 = self.from_latent(latent).unsqueeze(0).contiguous()  # (1, batch, hidden)
        c0 = torch.zeros_like(h0)

        # Teacher-forcing-free decoder: feed the previous output back in
        outputs = []
        prev = self.start_token.expand(batch, 1, self.n_features).contiguous()
        h, c = h0, c0
        for _ in range(self.window):
            out, (h, c) = self.decoder(prev, (h, c))
            y = self.to_output(out)
            outputs.append(y)
            prev = y
        recon = torch.cat(outputs, dim=1)
        return recon


class LSTMAutoencoderDetector(AnomalyDetector):
    """LSTM autoencoder over rolling windows of multivariate features."""

    name = "lstm_autoencoder"

    def __init__(
        self,
        window: int = 16,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.window = window
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self._model: _Seq2SeqLSTMAE | None = None
        self._sensor_columns: list[str] = []
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None
        self._train_recon_std: float = 1.0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, features: pd.DataFrame) -> "LSTMAutoencoderDetector":
        """Fit the autoencoder on rolling windows over raw sensor signals."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self._sensor_columns = [c for c in features.columns if c.startswith("S-") and "_" not in c]
        if not self._sensor_columns:
            raise ValueError(
                "LSTMAutoencoderDetector expects raw sensor columns (S-XX) in features."
            )

        windows = self._extract_windows(features)
        if windows.shape[0] < 2:
            raise ValueError("Not enough windows to train LSTM autoencoder.")

        # Normalize per-feature using training statistics
        flat = windows.reshape(-1, windows.shape[-1])
        self._feature_means = flat.mean(axis=0)
        self._feature_stds = np.where(flat.std(axis=0) < 1e-6, 1.0, flat.std(axis=0))

        normalized = (windows - self._feature_means) / self._feature_stds
        X = torch.tensor(normalized, dtype=torch.float32, device=self.device)

        n_features = len(self._sensor_columns)
        self._model = _Seq2SeqLSTMAE(
            n_features=n_features,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            window=self.window,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        n_samples = X.shape[0]
        self._model.train()
        for _ in range(self.epochs):
            idx = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                batch = X[batch_idx]
                recon = self._model(batch)
                loss = loss_fn(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            recon = self._model(X)
            errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
            self._train_recon_std = max(float(np.std(errors)), 1e-6)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Per-row anomaly score from windowed reconstruction error."""
        if self._model is None or self._feature_means is None or self._feature_stds is None:
            raise RuntimeError("LSTMAutoencoderDetector must be fit before scoring.")

        scores = np.zeros(len(features), dtype=np.float64)
        row_index = features.index.to_numpy()

        self._model.eval()
        with torch.no_grad():
            for machine_id, group in features.groupby("machine_id", sort=False):
                group = group.sort_values("timestamp")
                machine_positions = np.array([row_index.tolist().index(i) for i in group.index])
                values = group[self._sensor_columns].to_numpy(dtype=np.float64)

                if len(values) < self.window:
                    continue

                windows = np.lib.stride_tricks.sliding_window_view(
                    values, window_shape=(self.window, values.shape[1])
                )
                windows = windows.squeeze(axis=1)
                normalized = (windows - self._feature_means) / self._feature_stds
                X = torch.tensor(normalized, dtype=torch.float32, device=self.device)
                recon = self._model(X)
                errors = ((recon - X) ** 2).mean(dim=(1, 2)).cpu().numpy()
                # Attribute window error to the *end* of the window
                end_positions = machine_positions[self.window - 1 :]
                scores[end_positions] = errors

        return self._normalize_scores(scores)

    def _extract_windows(self, features: pd.DataFrame) -> np.ndarray:
        """Slice raw sensor matrix into windows per machine."""
        chunks = []
        for _, group in features.groupby("machine_id", sort=False):
            group = group.sort_values("timestamp")
            values = group[self._sensor_columns].to_numpy(dtype=np.float64)
            if len(values) < self.window:
                continue
            view = np.lib.stride_tricks.sliding_window_view(
                values, window_shape=(self.window, values.shape[1])
            )
            view = view.squeeze(axis=1)
            chunks.append(view.copy())
        if not chunks:
            return np.zeros((0, self.window, len(self._sensor_columns)))
        return np.concatenate(chunks, axis=0)
