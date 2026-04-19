"""Projection strategies for hierarchical reconciliation."""

from __future__ import annotations

import numpy as np

from .utils import get_sample_covariance, get_shrinkage_estimator


class ProjectionStrategy:
    """Base class for projection strategies."""

    needs_base_train: bool = False

    def fit(self, base_train_data, recon_val_data, model) -> None:
        """Optional data-dependent fitting."""

    def get_P(self) -> np.ndarray:
        raise NotImplementedError


class TopDownProjection(ProjectionStrategy):
    """Top-down projection: distribute top-level forecast equally.

    P = [q | 0], where q = ones(num_low, 1).
    Each bottom node receives the full top-level forecast value.
    """

    def __init__(self, num_low: int, num_node: int):
        q = np.ones((num_low, 1), dtype=float)
        zero_matrix = np.zeros((num_low, num_node - 1), dtype=float)
        self.P = np.hstack([q, zero_matrix])

    def get_P(self) -> np.ndarray:
        return self.P


class BottomUpProjection(ProjectionStrategy):
    """Bottom-up projection: use only bottom-level base forecasts.

    P = [0 | I], where I = eye(num_low).
    Ignores all upper-level base forecasts entirely.
    """

    def __init__(self, num_low: int, num_node: int):
        zero_matrix = np.zeros((num_low, num_node - num_low), dtype=float)
        identity = np.eye(num_low, dtype=float)
        self.P = np.hstack([zero_matrix, identity])

    def get_P(self) -> np.ndarray:
        return self.P


class MinTProjection(ProjectionStrategy):
    """Minimum Trace projection (Wickramasuriya et al. 2019).

    Computes P from the base forecast error covariance matrix W:
        P = J - J W U (U' W U)^{-1} U'

    Supports four covariance estimators via ``mode``:
    - OLS: identity (uncorrelated, equivariant)
    - WLS: diagonal of sample covariance
    - MINT_SAMPLE: full sample covariance
    - MINT_SHRINK: Schaefer-Strimmer shrinkage estimator

    Args:
        mode: Covariance estimator variant.
        num_node: Total number of hierarchy nodes.
        num_low: Number of bottom-level nodes.
        S: Summation matrix as a numpy array.

    Example:
        >>> S = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        >>> proj = MinTProjection("MINT_SHRINK", 3, 2, S)
        >>> proj.get_P()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        RuntimeError: MinTProjection requires fit() before get_P().
    """

    needs_base_train = True

    def __init__(
        self,
        mode: str,
        num_node: int,
        num_low: int,
        S: np.ndarray,
    ):
        self.mode = mode.upper()
        self.num_node = num_node
        self.num_low = num_low
        self.num_top = num_node - num_low
        C = (S.T[:, : self.num_top]).T
        self.J = np.concatenate(
            [
                np.zeros((num_low, self.num_top), dtype=float),
                np.eye(num_low, dtype=float),
            ],
            axis=1,
        )
        self.U_transposed = np.concatenate(
            [np.eye(self.num_top, dtype=float), -C],
            axis=1,
        )
        self.U = self.U_transposed.T
        self.P = None

    def fit(self, base_train_data, recon_val_data, model) -> None:
        if base_train_data is None:
            raise ValueError("MinTProjection.fit requires base_train_data.")
        train_forecast = np.asarray(base_train_data.forecast, dtype=float)
        train_observed = np.asarray(base_train_data.observed, dtype=float)
        if train_forecast.ndim == 3 and train_forecast.shape[2] == 1:
            residuals = (train_forecast[..., 0] - train_observed).T
        elif train_forecast.ndim == 3:
            residuals = train_forecast - train_observed[..., None]
        else:
            residuals = (train_forecast - train_observed).T
        self.P = self._compute_P(residuals)

    def _compute_P(self, residuals: np.ndarray) -> np.ndarray:
        is_3d = residuals.ndim == 3

        if self.mode == "OLS":
            if is_3d:
                num_samples = residuals.shape[2]
                W = np.repeat(np.eye(self.num_node, dtype=float)[..., None], num_samples, axis=2)
            else:
                W = np.eye(self.num_node, dtype=float)
        elif self.mode == "WLS":
            W_sample = get_sample_covariance(residuals)
            if is_3d:
                diag_W = np.diagonal(W_sample, axis1=0, axis2=1)
                W = np.zeros((self.num_node, self.num_node, residuals.shape[2]), dtype=float)
                indices = np.arange(self.num_node)
                W[indices, indices, :] = diag_W.T
            else:
                W = np.diag(np.diag(W_sample))
        elif self.mode == "MINT_SAMPLE":
            W = get_sample_covariance(residuals)
        elif self.mode == "MINT_SHRINK":
            W = get_shrinkage_estimator(residuals)
        else:
            raise ValueError(f"Unknown MinT mode {self.mode!r}. Available: OLS, WLS, MINT_SAMPLE, MINT_SHRINK")

        if is_3d:
            P = np.zeros((self.num_low, self.num_node, residuals.shape[2]), dtype=float)
            for s in range(residuals.shape[2]):
                middle = np.linalg.pinv(self.U_transposed @ W[:, :, s] @ self.U)
                P[:, :, s] = self.J - self.J @ W[:, :, s] @ self.U @ middle @ self.U_transposed
            return P
        return self.J - self.J @ W @ self.U @ np.linalg.pinv(self.U_transposed @ W @ self.U) @ self.U_transposed

    def get_P(self) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("MinTProjection requires fit() before get_P().")
        return self.P


class SparseProjection(ProjectionStrategy):
    """Learned sparse projection (Jeon et al. 2019).

    Each bottom node combines only from its ancestors in the hierarchy
    tree, with weights optimized via SciPy.  The sparsity pattern is
    determined by ``padded_parents`` from the S matrix.

    The P weights can be constrained via ``constraint_P``:
    - linear: softmax (sum-to-one, non-negative)
    - sum: softplus + normalize (sum-to-one, positive)
    - unconstrained: raw values masked by parent_mask
    """

    def __init__(
        self,
        constraint_P: str,
        padded_parents: np.ndarray,
        parent_mask: np.ndarray,
        P_init: np.ndarray,
    ):
        self.constraint_P = constraint_P
        self.padded_parents = np.asarray(padded_parents, dtype=int)
        self.parent_mask = np.asarray(parent_mask, dtype=bool)
        self.P_raw = np.asarray(P_init, dtype=float).copy()

    def get_P(self) -> np.ndarray:
        from .combine import apply_P_constraint_np

        return apply_P_constraint_np(self.P_raw, self.parent_mask, self.constraint_P)
