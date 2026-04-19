"""CKDRunner: Per-horizon CKD orchestrator.

Applies Conditional Kernel Density estimation per forecast horizon,
optionally optimizing bandwidth/decay via Optuna for each horizon.
Returns a GridForecastResult (N, G, H).

Pipeline per horizon h:
    1. Extract validation samples from base model ForecastResult at horizon h
    2. (Optional) Optimize bandwidth/time_decay via Optuna using val CRPS
    3. Fit CKD on observed (X, Y) with best hyperparameters
    4. Apply CKD to test ForecastResult at horizon h → GridDistribution
    5. Collect prob slices → stack into GridForecastResult (N, G, H)

All horizons are independent and can run in parallel via n_jobs.
"""

import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .forecast_results import BaseForecastResult, GridForecastResult
from src.models.conditional_kernel_density import (
    CKDConfig,
    ConditionalKernelDensity,
    resolve_to_samples,
)
from src.trainers.ckd import CKDOptunaConfig, CKDOptunaTrainer


class CKDRunner:
    """Per-horizon CKD orchestrator with optional Optuna optimization.

    Fits CKD models (one per horizon) on observed explanatory/target
    variable pairs, optionally searching for optimal bandwidth and
    time-decay per horizon.  Produces a GridForecastResult containing
    histogram densities for all horizons.

    Args:
        x_obs: Observed explanatory variables, shape (T_train, n_x_vars).
        y_obs: Observed target variable, shape (T_train,).
        x_columns: Names of the explanatory variables.
        base_config: Default CKDConfig (used when Optuna is disabled).
        n_jobs: Number of parallel workers (1=sequential, -1=all cores).
        model_name: Display name for the output GridForecastResult.
        optuna_config: Optional Optuna search configuration.
            If None, uses base_config directly (no optimization).

    Example:
        >>> runner = CKDRunner(
        ...     x_obs=x_train, y_obs=y_train, x_columns=["wind_speed"],
        ...     base_config=CKDConfig(n_x_vars=1, x_bandwidth=1.0, y_bandwidth=500.0),
        ... )
        >>> runner.fit_hyperparameters(val_results=[wind_speed_result], val_observed=val_y)
        >>> result = runner.run(test_results=[wind_speed_result_test])
        >>> result.to_distribution(h=1).mean()
    """

    def __init__(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        x_columns: List[str],
        base_config: CKDConfig,
        n_jobs: int = 1,
        model_name: str = "CKD",
        optuna_config: Optional[CKDOptunaConfig] = None,
    ):
        self.x_obs = np.asarray(x_obs, dtype=float)
        self.y_obs = np.asarray(y_obs, dtype=float).ravel()
        self.x_columns = x_columns
        self.base_config = base_config
        self.n_jobs = n_jobs
        self.model_name = model_name
        if isinstance(optuna_config, dict):
            optuna_config = CKDOptunaConfig(**optuna_config)
        self.optuna_config = optuna_config

        self.models_: Dict[int, ConditionalKernelDensity] = {}
        self.is_fitted_ = False

    def fit_hyperparameters(
        self,
        val_results: Optional[List[BaseForecastResult]] = None,
        val_observed: Optional[np.ndarray] = None,
    ) -> "CKDRunner":
        """Optimize hyperparameters and build CKD models, one per horizon.

        If val_results and val_observed are provided (and optuna_config
        is set), delegates to CKDOptunaTrainer per horizon to find
        the best bandwidth and time_decay.  Otherwise, builds a single
        CKD model with base_config and shares it across all horizons.

        Args:
            val_results: Validation-period ForecastResult objects
                (one per explanatory variable).  Required for Optuna.
            val_observed: Observed target values for validation period,
                shape (N_val,) or (N_val, H).  If 2-D, column h-1 is
                used for horizon h.  Required for Optuna.

        Returns:
            Self for method chaining.

        Example:
            >>> runner.fit_hyperparameters()  # fixed bandwidth
            >>> runner.fit_hyperparameters(val_results=[ws_result], val_observed=val_y)
        """
        use_optuna = (
            self.optuna_config is not None
            and val_results is not None
            and val_observed is not None
        )

        if use_optuna:
            H = val_results[0].horizon
            val_observed = np.asarray(val_observed, dtype=float)

            if self.n_jobs == 1:
                results = [
                    self._fit_horizon_optuna(h, val_results, val_observed)
                    for h in range(1, H + 1)
                ]
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._fit_horizon_optuna)(h, val_results, val_observed)
                    for h in range(1, H + 1)
                )

            for h, model in results:
                self.models_[h] = model
        else:
            # Fixed config: build once, share across all horizons
            model = ConditionalKernelDensity(self.base_config)
            model.build(self.x_obs, self.y_obs, self.x_columns)
            self._shared_model = model

        self.is_fitted_ = True
        return self

    def run(
        self,
        test_results: List[BaseForecastResult],
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GridForecastResult:
        """Apply fitted CKD models across all horizons.

        Args:
            test_results: Test-period ForecastResult objects
                (one per explanatory variable).
            n_samples: Monte Carlo samples for apply().
                Defaults to base_config.n_samples.
            seed: Random seed for reproducibility.

        Returns:
            GridForecastResult with shape (N, G, H).

        Raises:
            RuntimeError: If fit() has not been called.

        Example:
            >>> result = runner.run(test_results=[ws_result_test])
            >>> result.prob.shape  # (N, G, H)
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before run().")

        H = test_results[0].horizon

        if self.n_jobs == 1:
            horizon_results = [
                self._apply_horizon(h, test_results, n_samples, seed)
                for h in range(1, H + 1)
            ]
        else:
            horizon_results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._apply_horizon)(h, test_results, n_samples, seed)
                for h in range(1, H + 1)
            )

        # Sort by horizon and assemble
        horizon_results.sort(key=lambda x: x[0])

        grid = horizon_results[0][1].grid
        basis_index = horizon_results[0][1].index

        # Stack prob: (N, G, H)
        prob_slices = [gd.prob for _, gd in horizon_results]
        prob = np.stack(prob_slices, axis=-1)

        return GridForecastResult(
            grid=grid,
            prob=prob,
            basis_index=basis_index,
            model_name=self.model_name,
        )

    # ------------------------------------------------------------------
    # Private: per-horizon Optuna optimization
    # ------------------------------------------------------------------

    def _fit_horizon_optuna(
        self,
        h: int,
        val_results: List[BaseForecastResult],
        val_observed: np.ndarray,
    ) -> Tuple[int, ConditionalKernelDensity]:
        """Optimize and fit CKD for a single horizon via CKDOptunaTrainer.

        Example:
            >>> h, model = runner._fit_horizon_optuna(1, val_results, val_y)
        """
        # Extract validation samples for this horizon
        val_samples = []
        for res in val_results:
            s, _ = resolve_to_samples(
                res, n_samples=self.base_config.n_samples, horizon=h,
            )
            val_samples.append(s)

        # Observed for this horizon
        if val_observed.ndim == 2:
            val_y_h = val_observed[:, h - 1]
        else:
            val_y_h = val_observed

        trainer = CKDOptunaTrainer(
            base_config=self.base_config,
            optuna_config=self.optuna_config,
            train_x=self.x_obs,
            train_y=self.y_obs,
            x_columns=self.x_columns,
            val_samples=val_samples,
            val_observed=val_y_h,
        )
        return h, trainer.fit()

    # ------------------------------------------------------------------
    # Private: per-horizon apply
    # ------------------------------------------------------------------

    def _apply_horizon(
        self,
        h: int,
        test_results: List[BaseForecastResult],
        n_samples: Optional[int],
        seed: Optional[int],
    ) -> Tuple[int, "GridDistribution"]:
        """Apply CKD for a single horizon.

        Example:
            >>> h, gd = runner._apply_horizon(1, test_results, 1000, 42)
        """
        model = self.models_.get(h, getattr(self, "_shared_model", None))
        if model is None:
            raise RuntimeError(f"No fitted model for horizon {h}.")

        gd = model.apply(
            test_results,
            horizon=h,
            n_samples=n_samples,
            seed=seed,
        )
        return h, gd

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Save fitted CKDRunner state (per-horizon models) to a pickle file.

        Persists all fitted CKD models and runner metadata so that
        ``run()`` can be called immediately after ``load()`` without
        re-fitting.

        Args:
            path: Destination file path. ``.pkl`` suffix is added
                automatically if not present.

        Returns:
            The resolved file path (with ``.pkl`` suffix).

        Raises:
            RuntimeError: If the runner has not been fitted.

        Example:
            >>> runner.fit_hyperparameters(val_results, val_y)
            >>> saved = runner.save(Path("ckd_runner"))
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot save an unfitted runner.")

        path = Path(path).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize each CKD model via its own state dict
        model_states: Dict[str, object] = {}
        if self.models_:
            for h, model in self.models_.items():
                model_states[h] = model
        elif hasattr(self, "_shared_model"):
            model_states["shared"] = self._shared_model

        state = {
            "base_config": self.base_config.to_dict(),
            "x_columns": self.x_columns,
            "model_name": self.model_name,
            "models": model_states,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return path

    @classmethod
    def load(cls, path: Path) -> "CKDRunner":
        """Load a fitted CKDRunner from a pickle file.

        Restores all per-horizon CKD models so that ``run()`` can be
        called immediately without re-fitting or re-providing training
        data.

        Args:
            path: Source file path (``.pkl`` suffix added if missing).

        Returns:
            A fitted CKDRunner instance ready for ``run()``.

        Example:
            >>> runner = CKDRunner.load(Path("ckd_runner.pkl"))
            >>> result = runner.run(test_results=[ws_test])
        """
        path = Path(path).with_suffix(".pkl")
        with open(path, "rb") as f:
            state = pickle.load(f)

        config = CKDConfig(**state["base_config"])

        # Create runner with dummy training data (not needed for apply)
        runner = cls(
            x_obs=np.empty((0, config.n_x_vars)),
            y_obs=np.empty(0),
            x_columns=state["x_columns"],
            base_config=config,
            model_name=state["model_name"],
        )

        model_states = state["models"]
        if "shared" in model_states:
            runner._shared_model = model_states["shared"]
        else:
            runner.models_ = model_states

        runner.is_fitted_ = True
        return runner
