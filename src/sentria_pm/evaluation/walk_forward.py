"""Walk-forward evaluation harness.

Splits the dataset into consecutive train/test blocks, fits the model
on the train block, scores on the test block, slides forward. This
mimics deployment: we never train on data from the future.

The harness operates on feature frames already produced by the feature
extractor + raw sensor columns (ARIMA and LSTM AE need raw signals;
Isolation Forest can take engineered features alongside).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardSplit:
    """One fold of the walk-forward schedule."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray


class WalkForwardEvaluator:
    """Generate walk-forward train/test splits per machine.

    Splits are computed per machine (each machine has its own time axis),
    then concatenated. This avoids leaking one machine's future into
    another machine's training.
    """

    def __init__(
        self,
        n_folds: int = 3,
        min_train_fraction: float = 0.5,
        test_fraction: float = 0.2,
    ) -> None:
        if n_folds < 1:
            raise ValueError("n_folds must be >= 1.")
        if not 0.0 < min_train_fraction < 1.0:
            raise ValueError("min_train_fraction must be in (0, 1).")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1).")
        if min_train_fraction + test_fraction > 1.0:
            raise ValueError("min_train_fraction + test_fraction must be <= 1.0.")
        self.n_folds = n_folds
        self.min_train_fraction = min_train_fraction
        self.test_fraction = test_fraction

    def splits(self, features: pd.DataFrame) -> list[WalkForwardSplit]:
        """Produce ``n_folds`` walk-forward splits."""
        features = features.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

        per_machine_indices: dict[str, np.ndarray] = {
            m: g.index.to_numpy()
            for m, g in features.groupby("machine_id", sort=False)
        }

        splits: list[WalkForwardSplit] = []
        for fold in range(self.n_folds):
            train_idx_chunks: list[np.ndarray] = []
            test_idx_chunks: list[np.ndarray] = []
            progress = self.min_train_fraction + fold * (
                (1 - self.min_train_fraction - self.test_fraction) / max(self.n_folds - 1, 1)
            )
            for indices in per_machine_indices.values():
                n = len(indices)
                if n < 10:
                    continue
                train_end = int(n * progress)
                test_end = int(n * (progress + self.test_fraction))
                train_idx_chunks.append(indices[:train_end])
                test_idx_chunks.append(indices[train_end:test_end])

            if train_idx_chunks and test_idx_chunks:
                splits.append(
                    WalkForwardSplit(
                        fold_idx=fold,
                        train_indices=np.concatenate(train_idx_chunks),
                        test_indices=np.concatenate(test_idx_chunks),
                    )
                )
        return splits
