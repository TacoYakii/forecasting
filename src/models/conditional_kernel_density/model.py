"""Conditional Kernel Density (CKD) estimation model.

Implements CKD as a plain Python class with NumPy/SciPy operations:
- Config: CKDConfig(BaseConfig)
- Model: ConditionalKernelDensity with build()/apply()

Bandwidth and grid resolution are determined adaptively from training
data statistics.  Hyperparameter optimization is handled externally
(Optuna via CKDOptunaTrainer).
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Optional, Tuple

from .ckd_types import CKDConfig, resolve_to_samples
from src.core.forecast_distribution import GridDistribution


class ConditionalKernelDensity:
    """Conditional Kernel Density Estimation post-processor for wind power forecasting.

    Builds a conditional density estimate using Gaussian kernels with
    exponential time-decay weighting.  Takes base-model probabilistic
    forecasts as input and produces a GridDistribution of the target
    variable (power).

    Bandwidth and grid resolution are determined adaptively from
    training data statistics at ``build()`` time.  When used with
    ``CKDOptunaTrainer``, absolute bandwidths are injected via
    ``set_bandwidths()`` before ``build()``.

    Workflow:
        1. model = ConditionalKernelDensity(config)
        2. model.build(x_train, y_train, x_columns)
        3. grid_dist = model.apply(samples)
        4. crps = grid_crps(grid_dist, y_true)

    Args:
        config: CKDConfig dataclass with adaptive control parameters.

    Example:
        >>> config = CKDConfig(n_x_vars=1)
        >>> model = ConditionalKernelDensity(config)
        >>> model.build(x_train, y_train, ["wind_speed"])
        >>> gd = model.apply([wind_speed_samples])
        >>> gd.mean()
    """

    def __init__(self, config: CKDConfig):
        self.config: CKDConfig = config
        self.is_fitted: bool = False

        self.x_columns: List[str] = []
        self.n_x: int = config.n_x_vars
        self.time_decay_factor: float = config.time_decay_factor

        # Set by build() or set_bandwidths()
        self.x_bandwidths: List[float] = []
        self.y_bandwidth: float = 0.0
        self._bandwidths_injected: bool = False

        # Populated by build()
        self.density: Optional[np.ndarray] = None
        self.y_basis: Optional[np.ndarray] = None
        self.x_basis_list: List[np.ndarray] = []

        # Data statistics — populated by build()
        self.x_stats_: List[dict] = []
        self.y_stats_: dict = {}
        self.x_search_ranges_: List[Tuple[float, float]] = []
        self.y_search_range_: Tuple[float, float] = (0.0, 0.0)

    # ------------------------------------------------------------------
    # Data-adaptive helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_search_range(values: np.ndarray) -> Tuple[float, float]:
        """Compute Optuna search range from data.

        Lower bound: median spacing between sorted values.
        Upper bound: data range / 10.

        Args:
            values: 1-D array of observed values.

        Returns:
            (h_min, h_max) tuple of absolute bandwidth bounds.

        Example:
            >>> vals = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
            >>> ConditionalKernelDensity._compute_search_range(vals)
            (1.0, 0.7)
        """
        sorted_vals = np.sort(values)
        diffs = np.diff(sorted_vals)
        # Filter out zero-diffs (duplicate values)
        nonzero_diffs = diffs[diffs > 0]
        if len(nonzero_diffs) == 0:
            raise ValueError("All values are identical. Cannot compute search range.")
        h_min = float(np.median(nonzero_diffs))
        h_max = float((sorted_vals[-1] - sorted_vals[0]) / 10.0)
        # Ensure h_min < h_max
        if h_min >= h_max:
            h_max = h_min * 10.0
        return (h_min, h_max)

    @staticmethod
    def _reference_bandwidth(values: np.ndarray) -> float:
        """Silverman's robust rule of thumb for reference bandwidth.

        Uses min(σ, IQR/1.34) to handle multimodal distributions
        (e.g., wind power concentrated near 0 and rated capacity).

        Args:
            values: 1-D array of observed values.

        Returns:
            Reference bandwidth (absolute).

        Example:
            >>> vals = np.random.normal(0, 1, 1000)
            >>> h = ConditionalKernelDensity._reference_bandwidth(vals)
        """
        n = len(values)
        std = float(np.std(values))
        iqr = float(np.subtract(*np.percentile(values, [75, 25])))
        # Robust spread estimate
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        if spread <= 0:
            raise ValueError("Data has zero spread. Cannot compute bandwidth.")
        return 0.9 * spread * n ** (-1.0 / 5.0)

    def _compute_basis_points(self, data_range: float, bandwidth: float) -> int:
        """Compute grid resolution from bandwidth.

        Args:
            data_range: max - min of the variable.
            bandwidth: Absolute bandwidth.

        Returns:
            Number of grid points, clamped to [min_basis_points, max_basis_points].

        Example:
            >>> model._compute_basis_points(30.0, 1.0)
            121
        """
        bin_width = bandwidth / self.config.bins_per_bandwidth
        raw = int(data_range / bin_width) + 1
        return max(self.config.min_basis_points, min(raw, self.config.max_basis_points))

    # ------------------------------------------------------------------
    # Bandwidth injection (for Optuna)
    # ------------------------------------------------------------------

    def set_bandwidths(
        self,
        x_bandwidths: List[float],
        y_bandwidth: float,
    ) -> "ConditionalKernelDensity":
        """Inject absolute bandwidths (used by CKDOptunaTrainer).

        When set, ``build()`` skips the reference-bandwidth fallback
        and uses these values directly.

        Args:
            x_bandwidths: Per-variable absolute bandwidths.
            y_bandwidth: Response variable absolute bandwidth.

        Returns:
            self (for method chaining).

        Example:
            >>> model.set_bandwidths([0.8], 0.2).build(x, y, cols)
        """
        self.x_bandwidths = list(x_bandwidths)
        self.y_bandwidth = y_bandwidth
        self._bandwidths_injected = True
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_columns: List[str],
    ) -> "ConditionalKernelDensity":
        """Build the CKD density tensor from training data.

        Bandwidth and grid resolution are determined adaptively:

        1. Compute data statistics and search ranges per variable.
        2. If bandwidths were not injected via ``set_bandwidths()``,
           use Silverman's robust rule as the default bandwidth.
        3. Compute basis points from the selected bandwidth.
        4. Build the conditional density tensor.

        Args:
            x: Training explanatory variables, shape (T, n_x_vars).
            y: Training response variable, shape (T,).
            x_columns: Names of the explanatory variables.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: On shape mismatch, non-finite inputs, or degenerate range.

        Example:
            >>> model.build(x_train, y_train, ["wind_speed", "wind_dir"])
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # --- input validation ---
        if not np.all(np.isfinite(x)):
            raise ValueError("x contains non-finite values (NaN or Inf).")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains non-finite values (NaN or Inf).")

        self.x_columns = x_columns
        self.n_x = len(x_columns)

        T = x.shape[0]
        if T != y.shape[0]:
            raise ValueError(
                f"Length mismatch: x has {T} rows but y has {y.shape[0]}."
            )

        # --- Step 1: data statistics & search ranges ---
        self.x_stats_ = []
        self.x_search_ranges_ = []
        for i in range(self.n_x):
            col_vals = x[:, i]
            col_min, col_max = col_vals.min(), col_vals.max()
            if col_min == col_max:
                raise ValueError(
                    f"Explanatory variable '{x_columns[i]}' has zero range "
                    f"(min == max == {col_min}). Cannot build grid."
                )
            self.x_stats_.append({
                "min": float(col_min),
                "max": float(col_max),
                "range": float(col_max - col_min),
            })
            self.x_search_ranges_.append(self._compute_search_range(col_vals))

        y_min, y_max = float(y.min()), float(y.max())
        if y_min == y_max:
            raise ValueError(
                f"Response variable has zero range "
                f"(min == max == {y_min}). Cannot build grid."
            )
        self.y_stats_ = {
            "min": y_min,
            "max": y_max,
            "range": y_max - y_min,
        }
        self.y_search_range_ = self._compute_search_range(y)

        # --- Step 2: bandwidth (injected or reference) ---
        if not self._bandwidths_injected:
            self.x_bandwidths = [
                self._reference_bandwidth(x[:, i]) for i in range(self.n_x)
            ]
            self.y_bandwidth = self._reference_bandwidth(y)

        # --- Step 3: basis points from bandwidth ---
        # Use the search-range lower bound to determine grid resolution.
        # This ensures the grid is fine enough for the narrowest bandwidth
        # that could be tried (important for per-horizon Optuna where
        # different horizons may have different bandwidths but must share
        # the same grid).
        self.x_basis_list = []
        for i in range(self.n_x):
            bw_for_grid = min(self.x_bandwidths[i], self.x_search_ranges_[i][0])
            n_pts = self._compute_basis_points(
                self.x_stats_[i]["range"], bw_for_grid,
            )
            basis = np.linspace(
                self.x_stats_[i]["min"], self.x_stats_[i]["max"], n_pts,
            )
            self.x_basis_list.append(basis)

        bw_for_y_grid = min(self.y_bandwidth, self.y_search_range_[0])
        y_pts = self._compute_basis_points(
            self.y_stats_["range"], bw_for_y_grid,
        )
        self.y_basis = np.linspace(y_min, y_max, y_pts)

        # --- Step 4: density tensor ---
        self.density = self._build_density(x, y)
        self.is_fitted = True
        return self

    def _build_density(
        self,
        x_vals: np.ndarray,
        y_vals: np.ndarray,
    ) -> np.ndarray:
        """Vectorized CKD construction using np.einsum.

        Computes:
            density[b1,b2,...,y] = Σ_t w_t · K_x(x_t, b) · K_y(y_t, y) / Σ_t w_t · K_x(x_t, b)
        where K is a Gaussian kernel and w_t = decay^(T-1-t).

        Args:
            x_vals: shape (T, n_x_vars).
            y_vals: shape (T,).

        Returns:
            Density tensor of shape (B1, B2, ..., Y_size).

        Example:
            >>> density = model._build_density(x_vals, y_vals)
        """
        T = x_vals.shape[0]

        # Time decay weights: (T,)
        decay = self.time_decay_factor
        time_weights = decay ** np.arange(T - 1, -1, -1, dtype=float)

        # X kernel PDFs: each (T, B_i) with per-variable bandwidth
        x_pdfs = []
        for i, basis in enumerate(self.x_basis_list):
            diff = x_vals[:, i:i + 1] - basis  # (T, B_i)
            bw = self.x_bandwidths[i]
            x_pdfs.append(norm.pdf(diff, scale=bw))

        # Y kernel PDF: (T, Y_size)
        y_diff = y_vals[:, None] - self.y_basis  # (T, Y_size)
        y_pdf = norm.pdf(y_diff, scale=self.y_bandwidth)

        # Joint X density via broadcasting: (T, B1, B2, ...)
        joint_x = x_pdfs[0]  # (T, B0)
        for j, pdf in enumerate(x_pdfs[1:], start=1):
            # joint_x has shape (T, B0, B1, ..., B_{j-1})
            # pdf has shape (T, B_j)
            # Reshape pdf to (T, 1, 1, ..., 1, B_j) with j intermediate dims
            new_shape = (T,) + (1,) * j + (pdf.shape[1],)
            joint_x = joint_x[..., np.newaxis] * pdf.reshape(new_shape)

        # Apply time weights
        n_basis_dims = joint_x.ndim - 1
        w_shape = (T,) + (1,) * n_basis_dims
        weighted_x = time_weights.reshape(w_shape) * joint_x

        # Einsum for conditional density
        basis_letters = "".join(chr(ord("a") + i) for i in range(n_basis_dims))
        ein_str = f"t{basis_letters},ty->{basis_letters}y"
        numerator = np.einsum(ein_str, weighted_x, y_pdf)
        denominator = weighted_x.sum(axis=0)

        density = numerator / np.clip(denominator[..., np.newaxis], 1e-30, None)
        density = density / np.clip(density.sum(axis=-1, keepdims=True), 1e-30, None)

        return density

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def _resolve_inputs(
        self,
        inputs,
        time_index: Optional[pd.Index],
        horizon: Optional[int],
        n_samples: Optional[int],
        seed: Optional[int],
    ) -> Tuple[List[np.ndarray], Optional[pd.Index]]:
        """Normalize flexible inputs to (List[np.ndarray], time_index).

        Handles:
            - List[np.ndarray]: returned as-is
            - Single np.ndarray: wrapped in list (for n_x_vars=1)
            - Single Result object: resolved and wrapped in list
            - List of Result objects: each resolved independently

        Example:
            >>> samples, idx = model._resolve_inputs([arr], None, None, 1000, None)
        """
        _n_samples = n_samples if n_samples is not None else self.config.n_samples

        # Already a list of ndarrays → return as-is
        if (
            isinstance(inputs, list)
            and len(inputs) > 0
            and isinstance(inputs[0], np.ndarray)
        ):
            return inputs, time_index

        # Single ndarray → wrap
        if isinstance(inputs, np.ndarray):
            return [inputs], time_index

        # Single non-list object → wrap in list
        if not isinstance(inputs, list):
            inputs = [inputs]

        # List of Result objects → resolve each
        samples_list: List[np.ndarray] = []
        resolved_time_index = time_index
        for obj in inputs:
            s, ti = resolve_to_samples(
                obj,
                n_samples=_n_samples,
                horizon=horizon,
                seed=seed,
            )
            samples_list.append(s)
            if resolved_time_index is None and ti is not None:
                resolved_time_index = ti

        return samples_list, resolved_time_index

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def apply(
        self,
        inputs,
        time_index: Optional[pd.Index] = None,
        *,
        horizon: Optional[int] = None,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GridDistribution:
        """Apply the fitted CKD to forecast inputs.

        Accepts flexible input formats — raw arrays or any forecast result
        object from ``src.core``.

        Args:
            inputs: Forecast data in any supported format.
            time_index: Optional time index. Extracted from Result objects
                if not provided. Falls back to RangeIndex for raw arrays.
            horizon: Forecast horizon (1-indexed). Required for multi-horizon types.
            n_samples: Samples to draw from distribution-based types.
            seed: Random seed for reproducibility of sampling.

        Returns:
            GridDistribution with (T, Y_size) probability values.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: On input shape/index mismatch or non-finite samples.

        Example:
            >>> gd = model.apply([wind_speed_samples], horizon=1)
            >>> gd.mean()
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not built. Call .build(x, y, x_columns) first."
            )

        samples, resolved_time_index = self._resolve_inputs(
            inputs, time_index, horizon, n_samples, seed,
        )

        if len(samples) != self.n_x:
            raise ValueError(
                f"Expected {self.n_x} input variables, got {len(samples)}."
            )

        # --- Input validation ---
        # Check shape consistency
        ref_shape = samples[0].shape
        for i, s in enumerate(samples):
            if s.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch: input[0] has shape {ref_shape} "
                    f"but input[{i}] has shape {s.shape}."
                )
            if not np.all(np.isfinite(s)):
                raise ValueError(
                    f"Input[{i}] contains non-finite values (NaN or Inf)."
                )

        T = ref_shape[0]

        # Build time index fallback
        if resolved_time_index is None:
            resolved_time_index = pd.RangeIndex(T)

        # Map samples to grid indices using edges
        indices = []
        for i, s in enumerate(samples):
            basis = self.x_basis_list[i]
            bw = basis[1] - basis[0]
            edges = np.linspace(
                basis[0] - bw / 2, basis[-1] + bw / 2, len(basis) + 1
            )
            idx = np.searchsorted(edges, s, side="right") - 1
            idx = np.clip(idx, 0, len(basis) - 1)
            indices.append(idx)

        # Fancy indexing: density[idx_0, idx_1, ...] → (T, n_samples, Y_size)
        looked_up = self.density[tuple(indices)]

        # Average over simulations → (T, Y_size)
        power_density = looked_up.mean(axis=1)

        # Handle zero-mass rows (unseen conditioning regions)
        row_sums = power_density.sum(axis=1)
        zero_mask = row_sums < 1e-30
        if zero_mask.any():
            n_zero = zero_mask.sum()
            warnings.warn(
                f"{n_zero} time step(s) have near-zero density "
                f"(unseen conditioning region). Applying uniform fallback.",
                stacklevel=2,
            )
            power_density[zero_mask] = 1.0 / power_density.shape[1]
            row_sums[zero_mask] = 1.0

        # Re-normalize
        power_density = power_density / row_sums[:, None]

        return GridDistribution(
            index=resolved_time_index,
            grid=self.y_basis,
            prob=power_density,
        )

    def get_hyperparameters(self) -> dict:
        """Return current hyperparameter values as a dict.

        Includes absolute bandwidths, search ranges, and auto-computed
        grid resolution for full transparency.

        Example:
            >>> model.get_hyperparameters()
            {'x_bandwidth': {'wind_speed': 0.8}, 'y_bandwidth': 0.2, ...}
        """
        result: dict = {
            "x_bandwidth": {
                col: bw for col, bw in zip(self.x_columns, self.x_bandwidths)
            } if self.x_columns else self.x_bandwidths,
            "y_bandwidth": self.y_bandwidth,
            "time_decay_factor": self.time_decay_factor,
        }
        if self.x_basis_list:
            result["x_basis_points"] = [len(b) for b in self.x_basis_list]
        if self.y_basis is not None:
            result["y_basis_points"] = len(self.y_basis)
        if self.x_search_ranges_:
            result["x_search_ranges"] = {
                col: rng for col, rng in zip(self.x_columns, self.x_search_ranges_)
            } if self.x_columns else self.x_search_ranges_
        if self.y_search_range_ != (0.0, 0.0):
            result["y_search_range"] = self.y_search_range_
        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Save the fitted model state to a pickle file.

        Persists the density tensor, grid bases, bandwidths, and all
        metadata needed to restore the model for ``apply()`` without
        re-fitting on training data.

        Args:
            path: Destination file path. ``.pkl`` suffix is added
                automatically if not present.

        Returns:
            The resolved file path (with ``.pkl`` suffix).

        Raises:
            RuntimeError: If the model has not been fitted.

        Example:
            >>> model.build(x_train, y_train, ["wind_speed"])
            >>> saved = model.save(Path("ckd_model"))
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "config": self.config.to_dict(),
            "x_columns": self.x_columns,
            "n_x": self.n_x,
            "time_decay_factor": self.time_decay_factor,
            "x_bandwidths": self.x_bandwidths,
            "y_bandwidth": self.y_bandwidth,
            "_bandwidths_injected": self._bandwidths_injected,
            "density": self.density,
            "y_basis": self.y_basis,
            "x_basis_list": self.x_basis_list,
            "x_stats_": self.x_stats_,
            "y_stats_": self.y_stats_,
            "x_search_ranges_": self.x_search_ranges_,
            "y_search_range_": self.y_search_range_,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return path

    @classmethod
    def load(cls, path: Path) -> "ConditionalKernelDensity":
        """Load a fitted model from a pickle file.

        Restores the full model state so that ``apply()`` can be called
        immediately without re-fitting.

        Args:
            path: Source file path (``.pkl`` suffix added if missing).

        Returns:
            A fitted ConditionalKernelDensity instance.

        Example:
            >>> model = ConditionalKernelDensity.load(Path("ckd_model.pkl"))
            >>> gd = model.apply([wind_speed_samples])
        """
        path = Path(path).with_suffix(".pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = CKDConfig(**state["config"])
        model = cls(config)

        model.x_columns = state["x_columns"]
        model.n_x = state["n_x"]
        model.time_decay_factor = state["time_decay_factor"]
        model.x_bandwidths = state["x_bandwidths"]
        model.y_bandwidth = state["y_bandwidth"]
        model._bandwidths_injected = state["_bandwidths_injected"]
        model.density = state["density"]
        model.y_basis = state["y_basis"]
        model.x_basis_list = state["x_basis_list"]
        model.x_stats_ = state["x_stats_"]
        model.y_stats_ = state["y_stats_"]
        model.x_search_ranges_ = state["x_search_ranges_"]
        model.y_search_range_ = state["y_search_range_"]
        model.is_fitted = True

        return model
