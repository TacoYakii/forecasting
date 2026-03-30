"""
Trainers for Conditional Kernel Density (CKD) models.

Provides two trainer variants:

- CKDClosedFormTrainer: Builds CKD with fixed bandwidths, evaluates on validation.
  Use when bandwidths are known or pre-selected.

- CKDOptunaTrainer: Optimizes bandwidth/decay hyperparameters via Optuna TPE.
  Suitable for 3~4 hyperparameters where gradient-free search is natural.
"""

import torch
import numpy as np
import optuna
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict, Tuple
from pathlib import Path

from src.core.config import BaseConfig
from src.models.conditional_kernel_density.ckd_types import CKDConfig, PowerDensity
from src.models.conditional_kernel_density.model import ConditionalKernelDensity
from src.utils.loss import get_loss_from_config
from src.trainers.config import BaseTrainerConfig


# ---------------------------------------------------------------------------
# Optuna search space config
# ---------------------------------------------------------------------------

@dataclass
class CKDOptunaConfig(BaseConfig):
    """
    Configuration for CKD hyperparameter optimization via Optuna.

    Attributes:
        n_trials: Number of Optuna trials.
        x_bandwidth_range: (min, max) search range for each x bandwidth.
        y_bandwidth_range: (min, max) search range for y bandwidth.
        decay_range: (min, max) search range for time decay factor.
        n_samples_eval: Monte Carlo samples for CRPS evaluation.
        loss_func: Loss function name for evaluation.
        device: Torch device string.
        save_root: Path to save results.
    """
    n_trials: int = 100
    x_bandwidth_range: Tuple[float, float] = (0.01, 5.0)
    y_bandwidth_range: Tuple[float, float] = (50.0, 5000.0)
    decay_range: Tuple[float, float] = (0.98, 1.0)
    n_samples_eval: int = 1000
    loss_func: str = "RandomCRPSLoss"
    device: str = "cpu"
    save_root: Path = Path("outputs/ckd_optuna")


# ---------------------------------------------------------------------------
# Closed-form trainer (fixed bandwidths)
# ---------------------------------------------------------------------------

class CKDClosedFormTrainer:
    """
    Closed-form CKD trainer: builds density with fixed bandwidths.

    Usage:
        model = ConditionalKernelDensity(config, device)
        trainer = CKDClosedFormTrainer(model, config, train_x, train_y, ...)
        trainer.run()
    """

    def __init__(
        self,
        model: ConditionalKernelDensity,
        config: BaseTrainerConfig,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        x_columns: List[str],
        val_samples: Optional[List[torch.Tensor]] = None,
        val_observed: Optional[torch.Tensor] = None,
        test_samples: Optional[List[torch.Tensor]] = None,
        test_observed: Optional[torch.Tensor] = None,
        loss_kwargs: dict = {},
    ):
        self.model = model
        self.config = config
        self.device = config.device

        self.train_x = train_x
        self.train_y = train_y
        self.x_columns = x_columns
        self.val_samples = val_samples
        self.val_observed = val_observed
        self.test_samples = test_samples
        self.test_observed = test_observed

        self.loss_function = get_loss_from_config(config, **loss_kwargs)

        self.save_root = Path(config.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

    def fit(self) -> ConditionalKernelDensity:
        """Build CKD from training data (no optimization loop)."""
        self.model.fit(self.train_x, self.train_y, self.x_columns)
        return self.model

    def test(self, custom_loss_function=None) -> Any:
        """Evaluate on test set using CRPS."""
        if self.test_samples is None or self.test_observed is None:
            raise ValueError("Test samples and observations required.")

        loss_func = custom_loss_function or self.loss_function
        if hasattr(loss_func, "reduction"):
            loss_func.reduction = "none"

        with torch.no_grad():
            power_density = self.model.predict(self.test_samples)
            samples = power_density.to_samples(n=self.model.config.n_samples)
            y_true = self.test_observed.to(self.device)
            loss = loss_func(samples.float(), y_true.float())

        return loss.detach().cpu().numpy()

    def run(self, custom_loss_function=None) -> Any:
        self.fit()
        if self.test_samples is not None:
            return self.test(custom_loss_function)
        return None


# ---------------------------------------------------------------------------
# Optuna-based trainer
# ---------------------------------------------------------------------------

class CKDOptunaTrainer:
    """
    Hyperparameter optimization for CKD via Optuna TPE.

    Searches over x_bandwidth (per variable), y_bandwidth, and time_decay_factor.
    Each trial builds a CKD with candidate hyperparameters and evaluates CRPS
    on the validation set.

    Usage:
        trainer = CKDOptunaTrainer(
            base_config=ckd_config,
            optuna_config=optuna_config,
            train_x=train_x, train_y=train_y, x_columns=x_columns,
            val_samples=val_samples, val_observed=val_observed,
        )
        best_model = trainer.run()
    """

    def __init__(
        self,
        base_config: CKDConfig,
        optuna_config: CKDOptunaConfig,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        x_columns: List[str],
        val_samples: List[torch.Tensor],
        val_observed: torch.Tensor,
        test_samples: Optional[List[torch.Tensor]] = None,
        test_observed: Optional[torch.Tensor] = None,
        loss_kwargs: dict = {},
    ):
        self.base_config = base_config
        self.optuna_config = optuna_config
        self.device = optuna_config.device

        self.train_x = train_x
        self.train_y = train_y
        self.x_columns = x_columns
        self.n_x = len(x_columns)
        self.val_samples = val_samples
        self.val_observed = val_observed
        self.test_samples = test_samples
        self.test_observed = test_observed

        self.loss_function = get_loss_from_config(optuna_config, **loss_kwargs)

        self.save_root = Path(optuna_config.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.best_model: Optional[ConditionalKernelDensity] = None

    def _objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: build CKD with candidate params, return val CRPS."""
        oc = self.optuna_config

        # Suggest hyperparameters
        x_bws = [
            trial.suggest_float(
                f"x_bandwidth_{self.x_columns[i]}",
                oc.x_bandwidth_range[0],
                oc.x_bandwidth_range[1],
                log=True,
            )
            for i in range(self.n_x)
        ]
        y_bw = trial.suggest_float(
            "y_bandwidth",
            oc.y_bandwidth_range[0],
            oc.y_bandwidth_range[1],
            log=True,
        )
        decay = trial.suggest_float(
            "time_decay_factor",
            oc.decay_range[0],
            oc.decay_range[1],
        )

        # Build model with candidate hyperparameters
        config = CKDConfig(
            n_x_vars=self.n_x,
            x_bandwidth=x_bws,
            y_bandwidth=y_bw,
            time_decay_factor=decay,
            x_basis_points=self.base_config.x_basis_points,
            y_basis_points=self.base_config.y_basis_points,
            n_samples=oc.n_samples_eval,
        )
        model = ConditionalKernelDensity(config, device=self.device)
        model.fit(self.train_x, self.train_y, self.x_columns)

        # Evaluate on validation set
        with torch.no_grad():
            power_density = model.predict(self.val_samples)
            samples = power_density.to_samples(n=oc.n_samples_eval)
            y_true = self.val_observed.to(self.device)
            loss = self.loss_function(samples.float(), y_true.float())

        return loss.item()

    def fit(self) -> ConditionalKernelDensity:
        """Run Optuna study and return the best model."""
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.optuna_config.n_trials)

        # Rebuild best model
        best = study.best_params
        x_bws = [best[f"x_bandwidth_{col}"] for col in self.x_columns]

        best_config = CKDConfig(
            n_x_vars=self.n_x,
            x_bandwidth=x_bws,
            y_bandwidth=best["y_bandwidth"],
            time_decay_factor=best["time_decay_factor"],
            x_basis_points=self.base_config.x_basis_points,
            y_basis_points=self.base_config.y_basis_points,
            n_samples=self.base_config.n_samples,
        )
        self.best_model = ConditionalKernelDensity(best_config, device=self.device)
        self.best_model.fit(self.train_x, self.train_y, self.x_columns)

        print(f"Best CRPS: {study.best_value:.6f}")
        print(f"Best params: {study.best_params}")

        # Save best config
        best_config.save(self.save_root / "best_model_config.json")
        self.study = study

        return self.best_model

    def test(self, custom_loss_function=None) -> Any:
        """Evaluate the best model on test set."""
        if self.test_samples is None or self.test_observed is None:
            raise ValueError("Test samples and observations required.")
        if self.best_model is None:
            raise RuntimeError("Call fit() first.")

        loss_func = custom_loss_function or self.loss_function
        if hasattr(loss_func, "reduction"):
            loss_func.reduction = "none"

        with torch.no_grad():
            power_density = self.best_model.predict(self.test_samples)
            samples = power_density.to_samples(n=self.best_model.config.n_samples)
            y_true = self.test_observed.to(self.device)
            loss = loss_func(samples.float(), y_true.float())

        return loss.detach().cpu().numpy()

    def run(self, custom_loss_function=None) -> Any:
        """Full pipeline: optimize → test."""
        self.fit()
        if self.test_samples is not None:
            return self.test(custom_loss_function)
        return None
