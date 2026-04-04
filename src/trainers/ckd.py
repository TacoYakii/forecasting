"""Trainers for Conditional Kernel Density (CKD) models.

Provides two trainer variants:

- CKDClosedFormTrainer: Builds CKD with fixed bandwidths, evaluates on validation.
  Use when bandwidths are known or pre-selected.

- CKDOptunaTrainer: Optimizes bandwidth/decay hyperparameters via Optuna TPE.
  Suitable for 3~4 hyperparameters where gradient-free search is natural.
"""

import numpy as np
import optuna
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
from pathlib import Path

from src.core.config import BaseConfig
from src.models.conditional_kernel_density.ckd_types import CKDConfig
from src.utils.metrics.crps import grid_crps
from src.models.conditional_kernel_density.model import ConditionalKernelDensity
from src.trainers.config import BaseTrainerConfig


# ---------------------------------------------------------------------------
# Optuna search space config
# ---------------------------------------------------------------------------

@dataclass
class CKDOptunaConfig(BaseConfig):
    """Configuration for CKD hyperparameter optimization via Optuna.

    Search ranges for bandwidth are determined adaptively from
    training data (median spacing → lower bound, range/10 → upper
    bound).  Only ``decay_range`` is specified here since time decay
    is not data-scale-dependent.

    Attributes:
        n_trials: Number of Optuna trials.
        decay_range: (min, max) search range for time decay factor.
        n_samples_eval: Monte Carlo samples for evaluation.
        save_root: Path to save results.

    Example:
        >>> config = CKDOptunaConfig(n_trials=50)
    """

    n_trials: int = 100
    decay_range: Tuple[float, float] = (0.98, 1.0)
    n_samples_eval: int = 1000
    save_root: Path = Path("outputs/ckd_optuna")


# ---------------------------------------------------------------------------
# Closed-form trainer (fixed bandwidths)
# ---------------------------------------------------------------------------

class CKDClosedFormTrainer:
    """Closed-form CKD trainer: builds density with fixed bandwidths.

    All evaluation is numpy-native using grid_crps().

    Example:
        >>> model = ConditionalKernelDensity(config)
        >>> trainer = CKDClosedFormTrainer(model, config, train_x, train_y, ...)
        >>> trainer.run()
    """

    def __init__(
        self,
        model: ConditionalKernelDensity,
        config: BaseTrainerConfig,
        train_x: np.ndarray,
        train_y: np.ndarray,
        x_columns: List[str],
        test_samples: Optional[List[np.ndarray]] = None,
        test_observed: Optional[np.ndarray] = None,
    ):
        self.model = model
        self.config = config

        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)
        self.x_columns = x_columns
        self.test_samples = test_samples
        self.test_observed = (
            np.asarray(test_observed, dtype=float)
            if test_observed is not None else None
        )

        self.save_root = Path(config.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

    def fit(self) -> ConditionalKernelDensity:
        """Build CKD from training data (no optimization loop).

        Example:
            >>> trainer.fit()
        """
        self.model.build(self.train_x, self.train_y, self.x_columns)
        return self.model

    def test(self) -> np.ndarray:
        """Evaluate on test set using grid-exact CRPS.

        Returns:
            Per-observation CRPS values, shape (N,).

        Example:
            >>> crps_values = trainer.test()
            >>> crps_values.mean()
        """
        if self.test_samples is None or self.test_observed is None:
            raise ValueError("Test samples and observations required.")

        grid_dist = self.model.apply(self.test_samples)
        return grid_crps(grid_dist,self.test_observed)

    def run(self) -> Optional[np.ndarray]:
        """Full pipeline: fit → test.

        Example:
            >>> result = trainer.run()
        """
        self.fit()
        if self.test_samples is not None:
            return self.test()
        return None


# ---------------------------------------------------------------------------
# Optuna-based trainer
# ---------------------------------------------------------------------------

class CKDOptunaTrainer:
    """Hyperparameter optimization for CKD via Optuna TPE.

    Searches over x_bandwidth (per variable), y_bandwidth, and
    time_decay_factor.  Bandwidth search ranges are computed
    adaptively from the training data at ``__init__`` time using
    ``ConditionalKernelDensity._compute_search_range()``.

    Example:
        >>> trainer = CKDOptunaTrainer(
        ...     base_config=ckd_config,
        ...     optuna_config=optuna_config,
        ...     train_x=train_x, train_y=train_y, x_columns=x_columns,
        ...     val_samples=val_samples, val_observed=val_observed,
        ... )
        >>> best_model = trainer.fit()
    """

    def __init__(
        self,
        base_config: CKDConfig,
        optuna_config: CKDOptunaConfig,
        train_x: np.ndarray,
        train_y: np.ndarray,
        x_columns: List[str],
        val_samples: List[np.ndarray],
        val_observed: np.ndarray,
        test_samples: Optional[List[np.ndarray]] = None,
        test_observed: Optional[np.ndarray] = None,
    ):
        self.base_config = base_config
        self.optuna_config = optuna_config

        self.train_x = np.asarray(train_x, dtype=float)
        self.train_y = np.asarray(train_y, dtype=float)
        self.x_columns = x_columns
        self.n_x = len(x_columns)
        self.val_samples = val_samples
        self.val_observed = np.asarray(val_observed, dtype=float)
        self.test_samples = test_samples
        self.test_observed = (
            np.asarray(test_observed, dtype=float)
            if test_observed is not None else None
        )

        self.save_root = Path(optuna_config.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.best_model: Optional[ConditionalKernelDensity] = None

        # Compute data-driven search ranges once
        self._x_search_ranges = [
            ConditionalKernelDensity._compute_search_range(self.train_x[:, i])
            for i in range(self.n_x)
        ]
        self._y_search_range = ConditionalKernelDensity._compute_search_range(
            self.train_y,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: build CKD with candidate params, return val CRPS.

        Example:
            >>> # Called internally by Optuna
        """
        oc = self.optuna_config

        # Suggest bandwidths from data-driven ranges (log-scale)
        x_bws = [
            trial.suggest_float(
                f"x_bandwidth_{self.x_columns[i]}",
                self._x_search_ranges[i][0],
                self._x_search_ranges[i][1],
                log=True,
            )
            for i in range(self.n_x)
        ]
        y_bw = trial.suggest_float(
            "y_bandwidth",
            self._y_search_range[0],
            self._y_search_range[1],
            log=True,
        )
        decay = trial.suggest_float(
            "time_decay_factor",
            oc.decay_range[0],
            oc.decay_range[1],
        )

        # Build model with candidate absolute bandwidths
        config = CKDConfig(
            n_x_vars=self.n_x,
            time_decay_factor=decay,
            bins_per_bandwidth=self.base_config.bins_per_bandwidth,
            min_basis_points=self.base_config.min_basis_points,
            max_basis_points=self.base_config.max_basis_points,
            n_samples=oc.n_samples_eval,
        )
        model = ConditionalKernelDensity(config)
        model.set_bandwidths(x_bws, y_bw)
        model.build(self.train_x, self.train_y, self.x_columns)

        # Evaluate on validation set — grid-exact CRPS
        grid_dist = model.apply(self.val_samples)
        crps_values = grid_crps(grid_dist, self.val_observed)

        return float(crps_values.mean())

    def fit(self) -> ConditionalKernelDensity:
        """Run Optuna study and return the best model.

        Example:
            >>> best_model = trainer.fit()
        """
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.optuna_config.n_trials)

        # Rebuild best model
        best = study.best_params
        x_bws = [best[f"x_bandwidth_{col}"] for col in self.x_columns]

        best_config = CKDConfig(
            n_x_vars=self.n_x,
            time_decay_factor=best["time_decay_factor"],
            bins_per_bandwidth=self.base_config.bins_per_bandwidth,
            min_basis_points=self.base_config.min_basis_points,
            max_basis_points=self.base_config.max_basis_points,
            n_samples=self.base_config.n_samples,
        )
        self.best_model = ConditionalKernelDensity(best_config)
        self.best_model.set_bandwidths(x_bws, best["y_bandwidth"])
        self.best_model.build(self.train_x, self.train_y, self.x_columns)

        print(f"Best CRPS: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")

        # Save best config
        best_config.save(self.save_root / "best_model_config.yaml")
        self.study = study

        return self.best_model

    def test(self) -> np.ndarray:
        """Evaluate the best model on test set.

        Returns:
            Per-observation CRPS values, shape (N,).

        Example:
            >>> crps_values = trainer.test()
        """
        if self.test_samples is None or self.test_observed is None:
            raise ValueError("Test samples and observations required.")
        if self.best_model is None:
            raise RuntimeError("Call fit() first.")

        grid_dist = self.best_model.apply(self.test_samples)
        return grid_crps(grid_dist, self.test_observed)

    def run(self) -> Optional[np.ndarray]:
        """Full pipeline: optimize → test.

        Example:
            >>> result = trainer.run()
        """
        self.fit()
        if self.test_samples is not None:
            return self.test()
        return None
