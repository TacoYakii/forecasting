"""Coherence strategies for hierarchical reconciliation."""

from __future__ import annotations

import numpy as np


class CoherenceStrategy:
    """Base class for coherence reconstruction."""

    needs_setup: bool = False

    def setup(self, base_train_data, recon_val_data, model) -> None:
        """Optional data-dependent initialization."""

    def preprocess(self, forecast: np.ndarray) -> np.ndarray:
        """Preprocess forecast before combination."""
        return forecast

    def reconstruct(self, bottom: np.ndarray, model) -> np.ndarray:
        """Reconstruct full coherent hierarchy from bottom forecasts."""
        raise NotImplementedError


class RankedCoherence(CoherenceStrategy):
    """Comonotonic quantile alignment followed by S reconstruction."""

    def preprocess(self, forecast: np.ndarray) -> np.ndarray:
        if forecast.ndim == 3:
            return np.sort(forecast, axis=-1)
        return forecast

    def reconstruct(self, bottom: np.ndarray, model) -> np.ndarray:
        if bottom.ndim == 3:
            return np.einsum("nl,tlq->tnq", model.S, np.sort(bottom, axis=-1))
        return np.einsum("nl,tl->tn", model.S, bottom)


class EmpiricalCopulaCoherence(CoherenceStrategy):
    """Schaake shuffle coherence using empirical rank structure."""

    needs_setup = True

    def __init__(self):
        self.empirical_distribution = None
        self._T = 0
        self._num_nodes = 0
        self._num_sim = 0
        self._rng = np.random.default_rng(42)

    def setup(self, base_train_data, recon_val_data, model) -> None:
        data = base_train_data if base_train_data is not None else recon_val_data
        if data is None:
            return
        forecast = np.sort(data.forecast[:, -model.num_low :, :], axis=-1)
        observed = data.observed[:, -model.num_low :]
        self._fit_copula(forecast, observed)

    def _fit_copula(self, in_sample_hat: np.ndarray, in_sample_obs: np.ndarray) -> None:
        self._T, self._num_nodes, self._num_sim = in_sample_hat.shape
        self.empirical_distribution = (in_sample_hat <= in_sample_obs[..., None]).mean(axis=-1)

    def preprocess(self, forecast: np.ndarray) -> np.ndarray:
        if forecast.ndim == 3:
            return np.sort(forecast, axis=-1)
        return forecast

    def _sample_rank(self, batch_size: int) -> np.ndarray:
        if self.empirical_distribution is None:
            raise RuntimeError("EmpiricalCopulaCoherence requires setup() before use.")
        indices = self._rng.integers(0, self._T, size=(batch_size, self._num_sim))
        selected = self.empirical_distribution[indices]
        ranks = selected.argsort(axis=1).argsort(axis=1)
        return np.swapaxes(ranks, 1, 2)

    def reconstruct(self, bottom: np.ndarray, model) -> np.ndarray:
        sorted_bottom = np.sort(bottom, axis=-1)
        ranks = self._sample_rank(sorted_bottom.shape[0])
        shuffled = np.take_along_axis(sorted_bottom, ranks, axis=-1)
        return np.einsum("nl,tlq->tnq", model.S, shuffled)
