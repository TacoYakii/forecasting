"""Combine strategies for hierarchical reconciliation."""

from __future__ import annotations

import numpy as np

from src.models.combining.vertical import (
    _sampling_combine_crn,
)


def apply_P_constraint_np(
    P_raw: np.ndarray,
    mask: np.ndarray,
    constraint: str,
) -> np.ndarray:
    """Apply sparse projection constraints row-wise."""
    P_raw = np.asarray(P_raw, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    P = np.zeros_like(P_raw, dtype=float)
    for i in range(P_raw.shape[0]):
        valid = mask[i]
        if not valid.any():
            continue
        row = P_raw[i, valid]
        if constraint == "linear":
            row = row - row.max()
            weights = np.exp(row)
            P[i, valid] = weights / weights.sum()
        elif constraint == "sum":
            weights = np.log1p(np.exp(row))
            P[i, valid] = weights / (weights.sum() + 1e-9)
        elif constraint == "unconstrained":
            P[i, valid] = np.log1p(np.exp(row))
        else:
            raise ValueError(f"Unknown constraint_P {constraint!r}.")
    return P


def pack_sparse_params(
    P_raw: np.ndarray,
    mask: np.ndarray,
    angles: np.ndarray | None = None,
) -> np.ndarray:
    """Pack valid sparse P entries and optional angles into 1-D vector."""
    flat_rows = [P_raw[i, mask[i]] for i in range(P_raw.shape[0])]
    packed = np.concatenate(flat_rows) if flat_rows else np.empty(0, dtype=float)
    if angles is None:
        return packed
    return np.concatenate([packed, np.asarray(angles, dtype=float)])


def unpack_sparse_params(
    x: np.ndarray,
    mask: np.ndarray,
    with_angles: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Unpack flat optimization vector into sparse padded arrays."""
    mask = np.asarray(mask, dtype=bool)
    num_low, max_parents = mask.shape
    P_raw = np.zeros((num_low, max_parents), dtype=float)
    cursor = 0
    for i in range(num_low):
        count = int(mask[i].sum())
        P_raw[i, mask[i]] = x[cursor : cursor + count]
        cursor += count
    if not with_angles:
        return P_raw, None
    angles = np.asarray(x[cursor : cursor + num_low], dtype=float)
    return P_raw, angles


def sample_angular_random_numbers(
    num_low: int,
    T: int,
    mc_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random numbers for angular mixture sampling."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=(num_low, T, mc_samples))
    v = rng.uniform(0.0, 1.0, size=(num_low, T, mc_samples))
    return u, v


def weighted_combine_core(
    P: np.ndarray,
    forecast: np.ndarray,
    parents: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Sparse weighted quantile average."""
    forecast = np.asarray(forecast, dtype=float)
    T, _, Q = forecast.shape
    num_low = P.shape[0]
    bottom = np.zeros((T, num_low, Q), dtype=float)
    for i in range(num_low):
        valid = mask[i]
        parent_idx = parents[i, valid]
        weights = P[i, valid]
        weights = weights / (weights.sum() + 1e-12)
        local = forecast[:, parent_idx, :]
        bottom[:, i, :] = np.tensordot(local, weights, axes=([1], [0]))
    return np.sort(bottom, axis=-1)


def angular_combine_core(
    P: np.ndarray,
    angles: np.ndarray,
    q_val: np.ndarray,
    forecast: np.ndarray,
    parents: np.ndarray,
    mask: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """CRN-based angular combining shared by fit and inference."""
    forecast = np.asarray(forecast, dtype=float)
    T, _, Q = forecast.shape
    num_low = len(angles)
    bottom = np.zeros((T, num_low, Q), dtype=float)
    for i in range(num_low):
        valid = mask[i]
        parent_idx = parents[i, valid]
        weights = np.maximum(P[i, valid], 0.0)
        if weights.sum() < 1e-12:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()
        qa = [forecast[:, p, :] for p in parent_idx]
        deg = float(np.clip(angles[i], 1.0, 90.0))
        bottom[:, i, :] = _sampling_combine_crn(
            weights, q_val, qa, deg, u[i], v[i]
        )
    return bottom


class CombineStrategy:
    """Base class for combine strategies."""

    def combine(self, forecast: np.ndarray, P: np.ndarray, model) -> np.ndarray:
        raise NotImplementedError


class LinearCombine(CombineStrategy):
    """Direct matrix multiplication for full P projections."""

    def combine(self, forecast: np.ndarray, P: np.ndarray, model) -> np.ndarray:
        if forecast.ndim == 2:
            return np.einsum("mn,tn->tm", P, forecast)
        if P.ndim == 3:
            return np.einsum("nms,tms->tns", P, forecast)
        return np.einsum("mn,tnq->tmq", P, forecast)


class WeightedCombine(CombineStrategy):
    """Sparse weighted reconciliation."""

    def combine(self, forecast: np.ndarray, P: np.ndarray, model) -> np.ndarray:
        return weighted_combine_core(P, forecast, model.padded_parents, model.parent_mask)


class MeanShiftCombine(CombineStrategy):
    """Reconcile means, then shift bottom distributions."""

    def combine(self, forecast: np.ndarray, P: np.ndarray, model) -> np.ndarray:
        if forecast.ndim != 3:
            raise ValueError("MeanShiftCombine requires probabilistic forecasts.")

        mean_forecast = forecast.mean(axis=-1, keepdims=True)
        if model.config.projection == "sparse":
            reconciled_mean = weighted_combine_core(
                P,
                mean_forecast,
                model.padded_parents,
                model.parent_mask,
            )
        else:
            reconciled_mean = LinearCombine().combine(mean_forecast, P, model)

        bottom = forecast[:, -model.num_low :, :]
        bottom_mean = bottom.mean(axis=-1, keepdims=True)
        return bottom + (reconciled_mean - bottom_mean)


class AngularCombine(CombineStrategy):
    """Sampling-based angular combine with per-node angles."""

    def __init__(
        self,
        num_low: int,
        n_quantiles: int = 1000,
        mc_samples: int = 5000,
        q_start: float = 0.001,
        q_end: float = 0.999,
        rng_seed: int = 42,
    ):
        self.angle = np.full((num_low,), 45.0, dtype=float)
        self.q_val = np.linspace(q_start, q_end, n_quantiles, dtype=float)
        self.mc_samples = mc_samples
        self.rng_seed = rng_seed

    def get_angle_vector(self) -> np.ndarray:
        return np.clip(self.angle, 1.0, 90.0)

    def set_angle(self, angle: np.ndarray) -> None:
        self.angle = np.asarray(angle, dtype=float).copy()

    def combine(self, forecast: np.ndarray, P: np.ndarray, model) -> np.ndarray:
        u, v = sample_angular_random_numbers(
            model.num_low,
            forecast.shape[0],
            self.mc_samples,
            seed=self.rng_seed,
        )
        return angular_combine_core(
            P,
            self.get_angle_vector(),
            self.q_val,
            forecast,
            model.padded_parents,
            model.parent_mask,
            u,
            v,
        )
