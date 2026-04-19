"""Fitting utilities for reconciliation models."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize

from src.utils.metrics import crps_quantile

from .combine import (
    angular_combine_core,
    apply_P_constraint_np,
    pack_sparse_params,
    sample_angular_random_numbers,
    unpack_sparse_params,
    weighted_combine_core,
)


def make_crn(
    num_low: int,
    T: int,
    mc_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create common random numbers for deterministic objectives."""
    return sample_angular_random_numbers(num_low, T, mc_samples, seed)


def _mean_quantile_crps(
    coherent: np.ndarray,
    observed: np.ndarray,
    q_val: np.ndarray,
) -> float:
    if coherent.ndim != 3:
        raise ValueError("CRPS scoring expects a probabilistic array with shape (T, N, Q).")
    if coherent.shape[2] != len(q_val):
        q_val = np.linspace(float(q_val[0]), float(q_val[-1]), coherent.shape[2])
    total = 0.0
    for n in range(coherent.shape[1]):
        total += crps_quantile(q_val, coherent[:, n, :], observed[:, n], reduction="mean")
    return total / coherent.shape[1]


def weighted_objective(x, model, recon_val_data, fit_config) -> float:
    """Objective for sparse weighted reconciliation."""
    P_raw, _ = unpack_sparse_params(x, model.parent_mask, with_angles=False)
    P = apply_P_constraint_np(P_raw, model.parent_mask, model.config.constraint_P)
    bottom = weighted_combine_core(
        P,
        np.sort(recon_val_data.forecast, axis=-1),
        model.padded_parents,
        model.parent_mask,
    )
    coherent = model.coherence.reconstruct(bottom, model)
    loss = _mean_quantile_crps(coherent, recon_val_data.observed, model.q_val)
    if fit_config.reg_lambda > 0:
        for i in range(model.num_low):
            valid = model.parent_mask[i]
            count = int(valid.sum())
            if count > 0:
                loss += fit_config.reg_lambda * np.sum((P[i, valid] - 1.0 / count) ** 2)
    return float(loss)


def angular_objective(x, model, recon_val_data, fit_config, fixed_u, fixed_v) -> float:
    """Objective for sparse angular reconciliation."""
    P_raw, angles = unpack_sparse_params(x, model.parent_mask, with_angles=True)
    if angles is None:
        raise RuntimeError("Angles must be present for angular optimization.")
    P = apply_P_constraint_np(P_raw, model.parent_mask, model.config.constraint_P)
    bottom = angular_combine_core(
        P,
        angles,
        model.q_val,
        np.sort(recon_val_data.forecast, axis=-1),
        model.padded_parents,
        model.parent_mask,
        fixed_u,
        fixed_v,
    )
    coherent = model.coherence.reconstruct(bottom, model)
    loss = _mean_quantile_crps(coherent, recon_val_data.observed, model.q_val)
    if fit_config.reg_lambda > 0:
        for i in range(model.num_low):
            valid = model.parent_mask[i]
            count = int(valid.sum())
            if count > 0:
                loss += fit_config.reg_lambda * np.sum((P[i, valid] - 1.0 / count) ** 2)
    return float(loss)


class WeightedReconciliationFitter:
    """SciPy fitter for sparse weighted reconciliation."""

    def __init__(self, model, fit_config):
        self.model = model
        self.fit_config = fit_config

    def fit(self, recon_val_data) -> None:
        x0 = pack_sparse_params(self.model.projection.P_raw, self.model.parent_mask)
        result = minimize(
            weighted_objective,
            x0,
            args=(self.model, recon_val_data, self.fit_config),
            method="SLSQP",
            options={"maxiter": self.fit_config.maxiter},
        )
        if not result.success:
            warnings.warn(
                f"Weighted fitting did not fully converge: {result.message}",
                stacklevel=2,
            )
        P_raw, _ = unpack_sparse_params(result.x, self.model.parent_mask, with_angles=False)
        self.model.projection.P_raw = P_raw


class AngularReconciliationFitter:
    """SciPy fitter for sparse angular reconciliation."""

    def __init__(self, model, fit_config):
        self.model = model
        self.fit_config = fit_config

    def fit(self, recon_val_data) -> None:
        fixed_u, fixed_v = make_crn(
            self.model.num_low,
            recon_val_data.forecast.shape[0],
            self.model.combine.mc_samples,
            self.fit_config.crn_seed,
        )
        x0 = pack_sparse_params(
            self.model.projection.P_raw,
            self.model.parent_mask,
            self.model.combine.get_angle_vector(),
        )
        n_p = len(x0) - self.model.num_low
        constraints = []
        for i in range(self.model.num_low):
            idx = n_p + i
            constraints.append({"type": "ineq", "fun": lambda x, j=idx: x[j] - 1.0})
            constraints.append({"type": "ineq", "fun": lambda x, j=idx: 90.0 - x[j]})
        result = minimize(
            angular_objective,
            x0,
            args=(self.model, recon_val_data, self.fit_config, fixed_u, fixed_v),
            method=self.fit_config.scipy_method,
            constraints=constraints,
            options={"maxiter": self.fit_config.maxiter, "rhobeg": self.fit_config.rhobeg},
        )
        if not result.success:
            warnings.warn(
                f"Angular fitting did not fully converge: {result.message}",
                stacklevel=2,
            )
        P_raw, angles = unpack_sparse_params(result.x, self.model.parent_mask, with_angles=True)
        self.model.projection.P_raw = P_raw
        self.model.combine.set_angle(np.clip(angles, 1.0, 90.0))
