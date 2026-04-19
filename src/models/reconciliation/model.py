"""Unified hierarchical reconciliation model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .coherence import EmpiricalCopulaCoherence, RankedCoherence
from .combine import AngularCombine, LinearCombine, MeanShiftCombine, WeightedCombine
from .config import ReconciliationConfig
from .data import ReconciliationData
from .fit import AngularReconciliationFitter, WeightedReconciliationFitter
from .fit_config import ReconciliationFitConfig
from .projection import (
    BottomUpProjection,
    MinTProjection,
    SparseProjection,
    TopDownProjection,
)
from .utils import build_padded_adjacency, initialize_sparse_P


def _build_projection(config: ReconciliationConfig, model):
    if config.projection == "topdown":
        return TopDownProjection(model.num_low, model.num_node)
    if config.projection == "bottomup":
        return BottomUpProjection(model.num_low, model.num_node)
    if config.projection == "mint":
        return MinTProjection(config.mint_mode, model.num_node, model.num_low, model.S)
    if config.projection == "sparse":
        P_init = initialize_sparse_P(model.S, model.padded_parents, config.init_P)
        return SparseProjection(config.constraint_P, model.padded_parents, model.parent_mask, P_init)
    raise ValueError(f"Unknown projection type {config.projection!r}")


def _build_coherence(config: ReconciliationConfig):
    if config.coherence == "ranked":
        return RankedCoherence()
    if config.coherence == "empirical_copula":
        return EmpiricalCopulaCoherence()
    raise ValueError(f"Unknown coherence type {config.coherence!r}")


def _build_combine(config: ReconciliationConfig, model):
    if config.combine == "linear":
        return LinearCombine()
    if config.combine == "weighted":
        return WeightedCombine()
    if config.combine == "mean_shift":
        return MeanShiftCombine()
    if config.combine == "angular":
        return AngularCombine(
            num_low=model.num_low,
            n_quantiles=config.n_samples,
            mc_samples=config.mc_samples,
            q_start=config.q_start,
            q_end=config.q_end,
        )
    raise ValueError(f"Unknown combine type {config.combine!r}")


class HierarchicalReconciliation:
    """Facade for all reconciliation strategies."""

    def __init__(self, S: np.ndarray, config: ReconciliationConfig):
        config.validate()
        self.S = np.asarray(S, dtype=float)
        self.num_node = self.S.shape[0]
        self.num_low = self.S.shape[1]
        self.max_parents, self.padded_parents, self.parent_mask = build_padded_adjacency(self.S)
        self.config = config
        self.projection = _build_projection(config, self)
        self.coherence = _build_coherence(config)
        self.combine = _build_combine(config, self)
        self.q_val = getattr(
            self.combine,
            "q_val",
            np.linspace(config.q_start, config.q_end, config.n_samples),
        )

    def fit(
        self,
        base_train_data: ReconciliationData | None = None,
        recon_val_data: ReconciliationData | None = None,
        fit_config: ReconciliationFitConfig | None = None,
    ) -> None:
        """Fit reconciliation parameters using the appropriate split."""
        if fit_config is None:
            fit_config = ReconciliationFitConfig()

        self.coherence.setup(base_train_data, recon_val_data, self)
        self.projection.fit(base_train_data, recon_val_data, self)

        if self.config.projection == "sparse" and self.config.combine == "weighted":
            if recon_val_data is None:
                raise ValueError("Weighted sparse fitting requires recon_val_data.")
            WeightedReconciliationFitter(self, fit_config).fit(recon_val_data)
        elif self.config.projection == "sparse" and self.config.combine == "angular":
            if recon_val_data is None:
                raise ValueError("Angular sparse fitting requires recon_val_data.")
            AngularReconciliationFitter(self, fit_config).fit(recon_val_data)

    def reconcile(self, data: ReconciliationData) -> np.ndarray:
        """Return coherent forecasts for all nodes."""
        forecast = self.coherence.preprocess(np.asarray(data.forecast, dtype=float))
        P = self.projection.get_P()
        bottom = self.combine.combine(forecast, P, self)
        return self.coherence.reconstruct(bottom, self)

    def evaluate(self, data: ReconciliationData) -> np.ndarray:
        """Return node-wise CRPS on a probabilistic test split."""
        from src.utils.metrics import crps_quantile

        coherent = self.reconcile(data)
        if coherent.ndim != 3:
            raise ValueError(
                "evaluate() currently expects probabilistic forecasts with shape (T, N, Q)."
            )
        q_val = self.q_val
        if coherent.shape[2] != len(q_val):
            q_val = np.linspace(float(q_val[0]), float(q_val[-1]), coherent.shape[2])

        scores = np.zeros((coherent.shape[1],), dtype=float)
        for n in range(coherent.shape[1]):
            scores[n] = crps_quantile(
                q_val,
                coherent[:, n, :],
                data.observed[:, n],
                reduction="mean",
            )
        return scores

    def save_pretrained(self, path) -> None:
        """Save config and optimized parameters."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.config.save(path / "config.yaml")
        state = {
            "S": self.S,
            "projection_P_raw": getattr(self.projection, "P_raw", None),
            "combine_angle": getattr(self.combine, "angle", None),
        }
        np.savez(path / "state.npz", **state)

    @classmethod
    def from_pretrained(cls, S, path):
        """Load model from config and saved numpy state."""
        path = Path(path)
        config = ReconciliationConfig.load(path / "config.yaml")
        model = cls(S, config)
        state = np.load(path / "state.npz", allow_pickle=True)
        if hasattr(model.projection, "P_raw") and state["projection_P_raw"].dtype != object:
            model.projection.P_raw = state["projection_P_raw"]
        if hasattr(model.combine, "angle") and state["combine_angle"].dtype != object:
            model.combine.angle = state["combine_angle"]
        return model
