"""Abstract base class for forecast combiners.

Provides the common pipeline: validate → align → convert to quantiles →
fit/combine per horizon → assemble QuantileForecastResult.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.core.forecast_results import (
    BaseForecastResult,
    QuantileForecastResult,
    SampleForecastResult,
)
from src.utils.metrics import crps_quantile


class BaseCombiner(ABC):
    """Abstract base for combining multiple models' ForecastResults.

    Handles validation, index alignment, horizon-wise distribution
    extraction, and final SampleForecastResult assembly.  Subclasses
    implement only ``_fit_horizon`` and ``_combine_distributions``.

    Attributes:
        n_quantiles (int): Number of quantile levels for combining.
        quantile_levels (np.ndarray): Evenly spaced levels in (0, 1).
        weights_ (Dict[int, np.ndarray]): Learned per-horizon weights,
            keyed by horizon (1-indexed). Each value has shape (M,).
        train_scores_ (Dict[int, float]): Per-horizon CRPS on training set.
        val_scores_ (Dict[int, float]): Per-horizon CRPS on validation set.
            Empty when ``val_ratio=0``.
        model_names_ (List[str]): Model names from fit(), for display only.

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).
        val_ratio: Fraction of data to hold out as temporal validation
            set (default 0.0, no split). Last ``val_ratio`` fraction is
            used for validation; weights are learned on the rest.

    Example:
        >>> combiner = EqualWeightCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(
        self,
        n_quantiles: int = 99,
        n_jobs: int = 1,
        val_ratio: float = 0.0,
    ):
        if n_quantiles < 2:
            raise ValueError(
                f"n_quantiles must be >= 2, got {n_quantiles}."
            )
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError(
                f"val_ratio must be in [0, 1), got {val_ratio}."
            )
        self.n_quantiles = n_quantiles
        self.n_jobs = n_jobs
        self.val_ratio = val_ratio
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]
        self.is_fitted_ = False
        self.weights_: Dict[int, np.ndarray] = {}
        self.train_scores_: Dict[int, float] = {}
        self.val_scores_: Dict[int, float] = {}
        self._horizon: Optional[int] = None
        self._n_models: Optional[int] = None
        self.model_names_: Optional[List[str]] = None

    # ── Validation & Alignment ──

    @staticmethod
    def _validate_weights(
        weights: np.ndarray, n_models: int
    ) -> np.ndarray:
        """Validate combining weights.

        Checks 1-D shape, correct length, finiteness, non-negativity,
        and unit sum.  Reusable by all subclasses for both user-supplied
        and learned weights.

        Args:
            weights: Candidate weight array.
            n_models: Expected number of models.

        Returns:
            The validated array, unchanged.

        Raises:
            ValueError: If any check fails.

        Example:
            >>> BaseCombiner._validate_weights(np.array([0.5, 0.5]), 2)
            array([0.5, 0.5])
        """
        if weights.ndim != 1:
            raise ValueError(
                "Weights must be a 1D array with one value per model."
            )
        if len(weights) != n_models:
            raise ValueError(
                f"Expected {n_models} weights, got {len(weights)}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights must be finite.")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative.")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError("Weights must sum to 1.")
        return weights

    @staticmethod
    def _inverse_crps_weights(
        tau: np.ndarray,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse-CRPS weights: w_i = (1/CRPS_i) / sum(1/CRPS_j).

        Args:
            tau: Quantile levels, shape (Q,).
            quantile_arrays: M arrays of shape (N, Q).
            observed: Observed values, shape (N,).

        Returns:
            np.ndarray: Normalized weights, shape (M,).

        Example:
            >>> weights = BaseCombiner._inverse_crps_weights(tau, qarrays, obs)
            >>> weights.sum()
            1.0
        """
        crps_scores = np.array([
            crps_quantile(tau, q, observed, reduction="mean")
            for q in quantile_arrays
        ])
        crps_scores = np.maximum(crps_scores, 1e-12)
        inv_crps = 1.0 / crps_scores
        return inv_crps / inv_crps.sum()

    def _validate_results(self, results: List[BaseForecastResult]) -> None:
        """Validate compatibility of ForecastResult list.

        Checks that at least 2 models are provided and all share
        the same horizon H.

        Args:
            results: List of ForecastResult objects.

        Raises:
            ValueError: If fewer than 2 models or horizons differ.

        Example:
            >>> combiner._validate_results([res_a, res_b])
        """
        if len(results) < 2:
            raise ValueError(
                f"At least 2 models required, got {len(results)}."
            )

        H = results[0].horizon
        for i, r in enumerate(results[1:], 1):
            if r.horizon != H:
                raise ValueError(
                    f"Horizon mismatch: results[0].horizon={H}, "
                    f"results[{i}].horizon={r.horizon}"
                )

    def _align_results(
        self, results: List[BaseForecastResult]
    ) -> List[BaseForecastResult]:
        """Align all results to their common basis_index intersection.

        Args:
            results: M ForecastResult objects with potentially
                different basis_index values.

        Returns:
            List of reindexed results sharing the same basis_index.

        Raises:
            ValueError: If the common index is empty.

        Example:
            >>> aligned = combiner._align_results([res_a, res_b])
            >>> aligned[0].basis_index.equals(aligned[1].basis_index)
            True
        """
        common_idx = results[0].basis_index
        for r in results[1:]:
            common_idx = common_idx.intersection(r.basis_index)

        if len(common_idx) == 0:
            raise ValueError(
                "Common basis_index is empty. "
                "The models' forecast periods do not overlap."
            )

        all_equal = all(
            r.basis_index.equals(common_idx) for r in results
        )
        if all_equal:
            return results

        return [r.reindex(common_idx) for r in results]

    # ── Conversion ──

    def _convert_to_quantile(
        self, result: BaseForecastResult
    ) -> QuantileForecastResult:
        """Convert any ForecastResult to QuantileForecastResult.

        Normalizes all inputs to the combiner's ``quantile_levels``,
        avoiding expensive sort operations on large sample arrays
        during per-horizon distribution extraction.

        Args:
            result: Any ForecastResult subclass.

        Returns:
            QuantileForecastResult with combiner's quantile levels.

        Example:
            >>> qr = combiner._convert_to_quantile(sample_result)
            >>> qr.quantile_levels == combiner.quantile_levels
        """
        ql = self.quantile_levels
        H = result.horizon

        # Already a QuantileForecastResult with matching levels
        if isinstance(result, QuantileForecastResult):
            if np.array_equal(
                np.array(result.quantile_levels), ql
            ):
                return result

        quantiles_data = {}
        for h in range(1, H + 1):
            if isinstance(result, SampleForecastResult):
                # Direct percentile on raw samples — no sort needed
                samples_h = result.samples[:, :, h - 1]  # (N, n_samples)
                q_vals = np.percentile(
                    samples_h, ql * 100, axis=1
                ).T  # (N, Q)
            else:
                # Parametric or mismatched Quantile — use ppf
                dist = result.to_distribution(h)
                q_vals = dist.ppf(ql)  # (N, Q)

            for i, level in enumerate(ql):
                if level not in quantiles_data:
                    quantiles_data[level] = np.empty(
                        (len(result), H), dtype=float
                    )
                quantiles_data[level][:, h - 1] = q_vals[:, i]

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=result.basis_index,
            model_name=result.model_name,
        )

    def _convert_results(
        self, results: List[BaseForecastResult]
    ) -> List[QuantileForecastResult]:
        """Convert all results to QuantileForecastResult.

        Args:
            results: M ForecastResult objects.

        Returns:
            List of M QuantileForecastResult with combiner's quantile levels.

        Example:
            >>> converted = combiner._convert_results(results)
        """
        return [self._convert_to_quantile(r) for r in results]

    # ── Quantile Extraction ──

    def _extract_quantile_arrays(
        self, results: List[QuantileForecastResult], h: int
    ) -> List[np.ndarray]:
        """Extract quantile value arrays for a specific horizon.

        Directly reads from QuantileForecastResult.quantiles_data,
        bypassing QuantileDistribution to preserve exact values.

        Args:
            results: M QuantileForecastResult objects (post-conversion).
            h: Forecast horizon (1-indexed).

        Returns:
            List of M arrays, each shape (N, Q).

        Example:
            >>> arrays = combiner._extract_quantile_arrays(results, h=1)
            >>> arrays[0].shape  # (N, 99)
        """
        ql = self.quantile_levels
        return [
            np.column_stack(
                [r.quantiles_data[level][:, h - 1] for level in ql]
            )
            for r in results
        ]

    # ── Fit ──

    def fit(
        self,
        results: List[BaseForecastResult],
        observed: pd.DataFrame,
    ) -> "BaseCombiner":
        """Learn combining weights from training-period forecasts.

        For each horizon h, extracts distributions and calls
        ``_fit_horizon`` to obtain per-model weights.

        When ``val_ratio > 0``, the last fraction of data is held out
        as a temporal validation set. Weights are learned on the
        training portion only. Per-horizon CRPS scores for both splits
        are stored in ``train_scores_`` and ``val_scores_``.

        Args:
            results: M training-period ForecastResult objects.
            observed: Observed values, shape (N_total, H).
                Index must contain the common basis times.

        Returns:
            Self for method chaining.

        Example:
            >>> combiner.fit(train_results, observed)
            >>> combiner.weights_[1]   # np.array([0.5, 0.5])
            >>> combiner.val_scores_   # {1: 0.32, 2: 0.35, 3: 0.41}
        """
        # Reset fitted state so a failed refit doesn't leave stale weights
        self.is_fitted_ = False
        self.weights_ = {}
        self.train_scores_ = {}
        self.val_scores_ = {}
        self._horizon = None
        self._n_models = None
        self.model_names_ = None

        self._validate_results(results)

        # Reject duplicate or empty model names
        names = [r.model_name for r in results]
        for i, name in enumerate(names):
            if not name:
                raise ValueError(
                    f"results[{i}] has empty model_name. "
                    "All models must have a non-empty model_name."
                )
        if len(set(names)) != len(names):
            raise ValueError(
                f"Duplicate model_name values: {names}. "
                "Each model must have a unique model_name."
            )

        results = self._align_results(results)
        results = self._convert_results(results)

        H = results[0].horizon
        self._horizon = H
        self._n_models = len(results)
        self.model_names_ = names
        common_idx = results[0].basis_index

        # Intersect with observed index — forecast results may contain
        # origins without future actuals (e.g. last H origins)
        fit_idx = common_idx.intersection(observed.index)
        if len(fit_idx) == 0:
            raise ValueError(
                "No overlap between forecast results basis_index and "
                "observed index."
            )

        n_trimmed = len(common_idx) - len(fit_idx)
        if n_trimmed > 0:
            print(
                f"[BaseCombiner.fit] Trimmed {n_trimmed} origins without "
                f"observed data: {len(common_idx)} → {len(fit_idx)} "
                f"({len(fit_idx)} used for fitting)"
            )
            results = [r.reindex(fit_idx) for r in results]

        # Validate observed shape and finiteness
        observed_common = observed.loc[fit_idx]
        if observed_common.shape[1] != H:
            raise ValueError(
                f"observed must have {H} columns (one per horizon), "
                f"got {observed_common.shape[1]}."
            )
        observed_aligned = observed_common.values  # (N_common, H)
        if not np.all(np.isfinite(observed_aligned)):
            raise ValueError(
                "observed contains NaN or Inf values. "
                "All observations must be finite."
            )

        # Pre-extract quantile arrays for all horizons
        qarrays_per_h = {
            h: self._extract_quantile_arrays(results, h)
            for h in range(1, H + 1)
        }

        # Temporal train/validation split
        N = observed_aligned.shape[0]
        if self.val_ratio > 0:
            n_val = max(1, int(N * self.val_ratio))
            n_train = N - n_val
            if n_train < 2:
                raise ValueError(
                    f"val_ratio={self.val_ratio} leaves only {n_train} "
                    f"training samples (need >= 2). "
                    f"Reduce val_ratio or provide more data."
                )
        else:
            n_train = N

        def _split_train(qarrays_h, observed_h):
            """Split quantile arrays and observed into train portion."""
            train_q = [q[:n_train] for q in qarrays_h]
            train_o = observed_h[:n_train]
            return train_q, train_o

        def _fit_single(h, qarrays_h, observed_h):
            train_q, train_o = _split_train(qarrays_h, observed_h)
            return h, self._fit_horizon(h, train_q, train_o)

        if self.n_jobs == 1:
            fit_results = [
                _fit_single(h, qarrays_per_h[h], observed_aligned[:, h - 1])
                for h in range(1, H + 1)
            ]
        else:
            fit_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single)(
                    h, qarrays_per_h[h], observed_aligned[:, h - 1]
                )
                for h in range(1, H + 1)
            )

        for h, weight_array in fit_results:
            self._validate_weights(weight_array, self._n_models)
            self.weights_[h] = weight_array

        # Compute CRPS scores on train (and validation if split)
        # Uses _combine_distributions so scoring reflects the actual
        # combining method (horizontal vs vertical).
        tau = self.quantile_levels
        for h in range(1, H + 1):
            qarrays_h = qarrays_per_h[h]
            observed_h = observed_aligned[:, h - 1]

            train_q = [q[:n_train] for q in qarrays_h]
            q_train = self._combine_distributions(h, train_q)
            self.train_scores_[h] = crps_quantile(
                tau, q_train, observed_h[:n_train], reduction="mean"
            )

            if self.val_ratio > 0:
                val_q = [q[n_train:] for q in qarrays_h]
                q_val = self._combine_distributions(h, val_q)
                self.val_scores_[h] = crps_quantile(
                    tau, q_val, observed_h[n_train:], reduction="mean"
                )

        self.is_fitted_ = True
        return self

    # ── Combine ──

    def combine(
        self,
        results: List[BaseForecastResult],
        model_name: Optional[str] = None,
    ) -> QuantileForecastResult:
        """Combine test-period forecasts into a single QuantileForecastResult.

        Args:
            results: M test-period ForecastResult objects
                (same model count and order as fit).
            model_name: Name for the combined result. Defaults to
                the combiner class name (e.g. "HorizontalCombiner").

        Returns:
            QuantileForecastResult with Q quantile levels and H horizons.

        Raises:
            RuntimeError: If fit() has not been called.

        Example:
            >>> combined = combiner.combine(test_results, model_name="Ensemble_v1")
            >>> combined.model_name  # 'Ensemble_v1'
        """
        if not self.is_fitted_:
            raise RuntimeError("fit() must be called before combine().")

        self._validate_results(results)
        results = self._align_results(results)
        results = self._convert_results(results)

        H = results[0].horizon
        if H != self._horizon:
            raise ValueError(
                f"Expected horizon {self._horizon} from fit(), got {H}."
            )

        if len(results) != self._n_models:
            raise ValueError(
                f"Expected {self._n_models} models from fit(), "
                f"got {len(results)}."
            )

        combine_names = [r.model_name for r in results]
        if combine_names != self.model_names_:
            raise ValueError(
                "Model names/order mismatch between fit() and combine(). "
                f"Expected {self.model_names_}, got {combine_names}."
            )

        basis_index = results[0].basis_index

        qarrays_per_h = {
            h: self._extract_quantile_arrays(results, h)
            for h in range(1, H + 1)
        }

        def _combine_single(h, qarrays_h):
            return h, self._combine_distributions(h, qarrays_h)

        if self.n_jobs == 1:
            combine_results = [
                _combine_single(h, qarrays_per_h[h])
                for h in range(1, H + 1)
            ]
        else:
            combine_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_combine_single)(h, qarrays_per_h[h])
                for h in range(1, H + 1)
            )

        combine_results.sort(key=lambda x: x[0])
        combined_per_h = [arr for _, arr in combine_results]

        # List of (N, Q) → quantiles_data: {level: (N, H)}
        stacked = np.stack(combined_per_h, axis=2)  # (N, Q, H)
        quantiles_data = {
            level: stacked[:, i, :]
            for i, level in enumerate(self.quantile_levels)
        }

        return QuantileForecastResult(
            quantiles_data=quantiles_data,
            basis_index=basis_index,
            model_name=model_name or type(self).__name__,
        )

    # ── Abstract Methods ──

    @abstractmethod
    def _fit_horizon(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
        observed: np.ndarray,
    ) -> np.ndarray:
        """Learn per-model weights for a single horizon.

        Args:
            h: Forecast horizon (1-indexed).
            quantile_arrays: M arrays of shape (N_train, Q), one per model.
            observed: Observed values at this horizon, shape (N_train,).

        Returns:
            np.ndarray: Model weights, shape (M,), summing to 1.

        Example:
            >>> weights = combiner._fit_horizon(1, qarrays, obs)
            >>> weights.sum()
            1.0
        """
        ...

    @abstractmethod
    def _combine_distributions(
        self,
        h: int,
        quantile_arrays: List[np.ndarray],
    ) -> np.ndarray:
        """Combine quantile arrays for a single horizon using learned weights.

        Args:
            h: Forecast horizon (1-indexed).
            quantile_arrays: M arrays of shape (N_test, Q), one per model.

        Returns:
            np.ndarray: Combined quantile values, shape (N, Q).

        Example:
            >>> combined = combiner._combine_distributions(1, qarrays)
            >>> combined.shape
            (100, 99)
        """
        ...
