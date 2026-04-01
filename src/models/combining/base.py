"""Abstract base class for forecast combiners.

Provides the common pipeline: validate → align → convert to quantiles →
fit/combine per horizon → assemble QuantileForecastResult.

TODO: MQLoss/IQLoss 기반 모델은 각 분위수를 독립적으로 예측하므로
    quantile crossing (Q(tau_i) > Q(tau_j) for tau_i < tau_j)이
    발생할 수 있다. Crossing quantiles가 VerticalCombiner의 CDF
    보간을 왜곡하므로, 입력 검증 또는 isotonic regression 기반
    monotone repair가 필요하다.
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
        model_names_ (List[str]): Model names from fit(), for display only.

    Args:
        n_quantiles: Number of quantile levels (default 99).
        n_jobs: Parallel workers for horizon loop (default 1).

    Example:
        >>> combiner = EqualWeightCombiner(n_quantiles=99)
        >>> combiner.fit(train_results, observed)
        >>> combined = combiner.combine(test_results)
    """

    def __init__(self, n_quantiles: int = 99, n_jobs: int = 1):
        if n_quantiles < 2:
            raise ValueError(
                f"n_quantiles must be >= 2, got {n_quantiles}."
            )
        self.n_quantiles = n_quantiles
        self.n_jobs = n_jobs
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 2)[1:-1]
        self.is_fitted_ = False
        self.weights_: Dict[int, np.ndarray] = {}
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
        bypassing EmpiricalDistribution to preserve exact values.

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

        Args:
            results: M training-period ForecastResult objects.
            observed: Observed values, shape (N_total, H).
                Index must contain the common basis times.

        Returns:
            Self for method chaining.

        Example:
            >>> combiner.fit(train_results, observed)
            >>> combiner.weights_[1]   # np.array([0.5, 0.5])
        """
        # Reset fitted state so a failed refit doesn't leave stale weights
        self.is_fitted_ = False
        self.weights_ = {}
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

        # Validate observed shape and finiteness
        observed_common = observed.loc[common_idx]
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

        def _fit_single(h, qarrays_h, observed_h):
            return h, self._fit_horizon(h, qarrays_h, observed_h)

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
