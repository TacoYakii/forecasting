"""Utility functions for hierarchical reconciliation."""

from __future__ import annotations

import numpy as np


def get_sample_covariance(residuals: np.ndarray) -> np.ndarray:
    """Compute sample covariance matrix of base forecast residuals.

    Supports both 2D (m, T) and 3D (T, m, S) residual tensors.
    For 3D, computes per-sample covariance using chunked einsum
    to avoid memory issues with large sample counts.

    Args:
        residuals: Residual tensor, shape (m, T) or (T, m, S).

    Returns:
        Covariance matrix, shape (m, m) or (m, m, S).

    Example:
        >>> residuals = np.random.randn(5, 100)
        >>> W = get_sample_covariance(residuals)
        >>> W.shape
        (5, 5)
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim == 3:
        T, m, S = residuals.shape
        means = residuals.mean(axis=0, keepdims=True)
        X = residuals - means
        factor = 1.0 / (T - 1)
        return np.einsum("tms,tns->mns", X, X) * factor

    n_samples = residuals.shape[1]
    means = residuals.mean(axis=1, keepdims=True)
    X = residuals - means
    factor = 1.0 / (n_samples - 1)
    return X @ X.T * factor


def get_shrinkage_estimator(residuals: np.ndarray) -> np.ndarray:
    r"""Compute Schaefer & Strimmer (2005) shrinkage covariance estimator.

    Shrinks the sample covariance toward its diagonal to ensure
    positive semi-definiteness, especially when m > T.

    Formula:
        W_shrunk = \lambda^* W_diag + (1 - \lambda^*) W_sample

    Supports both 2D (m, T) and 3D (T, m, S) residual tensors.

    Args:
        residuals: Residual tensor, shape (m, T) or (T, m, S).

    Returns:
        Shrinkage-estimated covariance, shape (m, m) or (m, m, S).

    Example:
        >>> residuals = np.random.randn(5, 100)
        >>> W = get_shrinkage_estimator(residuals)
        >>> W.shape
        (5, 5)
    """
    residuals = np.asarray(residuals, dtype=float)
    epsilon = 2e-8
    W = get_sample_covariance(residuals)

    if residuals.ndim == 3:
        T, m, S = residuals.shape
        means = residuals.mean(axis=0, keepdims=True)
        X = residuals - means
        stds = residuals.std(axis=0, ddof=0, keepdims=True)
        Xs = X / (stds + epsilon)
        R_biased = np.einsum("tms,tns->mns", Xs, Xs) / T
        R_sq = R_biased**2
        diag_R_sq = np.diagonal(R_sq, axis1=0, axis2=1).sum(axis=0)
        sum_sq_emp_corr = R_sq.sum(axis=(0, 1)) - diag_R_sq
        Xs_sq = Xs**2
        sum_w_sq = np.einsum("tms,tns->mns", Xs_sq, Xs_sq)
        var_matrix = sum_w_sq - T * R_sq
        diag_var = np.diagonal(var_matrix, axis1=0, axis2=1).sum(axis=0)
        sum_var_emp_corr = var_matrix.sum(axis=(0, 1)) - diag_var
        factor_shrinkage = 1.0 / (T * (T - 1))
        lambda_star = (factor_shrinkage * sum_var_emp_corr) / (sum_sq_emp_corr + epsilon)
        lambda_star = np.clip(lambda_star, 0.0, 1.0)
        shrinkage = 1.0 - lambda_star
        W_shrunk = W * shrinkage[None, None, :]
        diag = np.arange(m)
        W_shrunk[diag, diag, :] = W[diag, diag, :]
        return W_shrunk

    T = residuals.shape[1]
    means = residuals.mean(axis=1, keepdims=True)
    X = residuals - means
    factor_shrinkage = 1.0 / (T * (T - 1))
    stds = residuals.std(axis=1, ddof=0, keepdims=True)
    Xs = X / (stds + epsilon)
    Xs_centered = Xs - Xs.mean(axis=1, keepdims=True)
    R_biased = (Xs_centered @ Xs_centered.T) / T
    R_sq = R_biased**2
    sum_sq_emp_corr = R_sq.sum() - np.trace(R_sq)
    Xs_sq = Xs_centered**2
    sum_w_sq = Xs_sq @ Xs_sq.T
    var_matrix = sum_w_sq - T * R_sq
    sum_var_emp_corr = var_matrix.sum() - np.trace(var_matrix)
    lambda_star = (factor_shrinkage * sum_var_emp_corr) / (sum_sq_emp_corr + epsilon)
    lambda_star = float(np.clip(lambda_star, 0.0, 1.0))
    shrinkage = 1.0 - lambda_star
    diag_W = np.diag(W).copy()
    W = W * shrinkage
    np.fill_diagonal(W, diag_W)
    return W


def extract_tree_from_S(S: np.ndarray) -> dict:
    """Extract parent-child relations from the summation matrix.

    Finds direct parent-child relationships by testing subset
    inclusion of support sets (nonzero columns per row).

    Args:
        S: Summation matrix, shape (n, m).

    Returns:
        Dictionary mapping parent index to list of (child_index, weight)
        tuples.

    Example:
        >>> S = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        >>> tree = extract_tree_from_S(S)
        >>> tree[0]  # node 0 is parent of nodes 1, 2
        [(1, 1.0), (2, 1.0)]
    """
    n = S.shape[0]
    S = np.asarray(S, dtype=float)
    supports = [set(np.where(row > 0)[0].tolist()) for row in S]
    children_dict = {}
    for i in range(n):
        child_list = []
        for j in range(n):
            if i == j:
                continue
            if supports[j].issubset(supports[i]) and supports[j] != supports[i]:
                is_direct_child = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (
                        supports[j].issubset(supports[k])
                        and supports[k].issubset(supports[i])
                        and supports[k] != supports[j]
                        and supports[k] != supports[i]
                    ):
                        is_direct_child = False
                        break
                if is_direct_child:
                    c = next(iter(supports[j]))
                    weight = S[i, c] / S[j, c]
                    child_list.append((j, float(weight)))
        if child_list:
            children_dict[i] = child_list
    return children_dict


def build_padded_adjacency(S: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    """Create sparse ancestor map from summation matrix."""
    S = np.asarray(S, dtype=float)
    active_row, active_col = np.where(S.T > 0)
    adj_list = [active_col[active_row == i] for i in range(S.shape[1])]
    max_parents = max((len(v) for v in adj_list), default=0)
    padded = np.zeros((S.shape[1], max_parents), dtype=int)
    mask = np.zeros((S.shape[1], max_parents), dtype=bool)
    for i, parents in enumerate(adj_list):
        padded[i, : len(parents)] = parents
        mask[i, : len(parents)] = True
    return max_parents, padded, mask


def initialize_sparse_P(S: np.ndarray, padded_parents: np.ndarray, init_P: str) -> np.ndarray:
    """Initialize sparse projection parameters."""
    S = np.asarray(S, dtype=float)
    if init_P == "ols":
        full = np.linalg.pinv(S.T @ S) @ S.T
        return np.take_along_axis(full, padded_parents, axis=1)
    if init_P == "random":
        rng = np.random.default_rng(42)
        return rng.normal(scale=0.01, size=padded_parents.shape)
    if init_P == "uniform":
        return np.ones_like(padded_parents, dtype=float)
    raise ValueError(f"Unknown init_P {init_P!r}.")
