import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
from copy import deepcopy
from typing import Optional, Any, Union
import numpy as np

# Forward declaration for type hinting

from src.core.loss import LossBase

from src.trainers.config import BaseTrainerConfig
from src.utils.loss import get_loss_from_config


class BaseTrainer:
    """
    Standard BaseTrainer abstraction.

    Contains shared DataLoader connections, model registrations, and test executions.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_kwargs = {}
    ):
        self.config = config
        self.device = torch.device(getattr(self.config, "device", "cuda"))

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        base_save_root = Path(self.config.save_root)
        if base_save_root.exists() and any(base_save_root.iterdir()):
            counter = 0
            while (base_save_root / str(counter)).exists() and any((base_save_root / str(counter)).iterdir()):
                counter += 1
            self.save_root = base_save_root / str(counter)
        else:
            self.save_root = base_save_root
            
        self.save_root.mkdir(parents=True, exist_ok=True)
        
        # Save configuration files if .save() method is implemented (e.g., config inherits from BaseConfig)
        if hasattr(self.config, "save") and callable(self.config.save):
            self.config.save(self.save_root / "trainer_config.json")
            
        if hasattr(self.model, "config") and hasattr(self.model.config, "save") and callable(self.model.config.save):
            self.model.config.save(self.save_root / "model_config.json")  # ty:ignore[too-many-positional-arguments]

        self.loss_kwargs = loss_kwargs
        self.loss_function = get_loss_from_config(self.config, **self.loss_kwargs)
        self.sv_forecast = getattr(self.config, "sv_forecast", True)

    def fit(self) -> Optional[dict]:
        raise NotImplementedError("Subclasses must implement the fit logic.")

    def test(self, custom_loss_function: Optional[Union[nn.Module, LossBase]] = None) -> Any:
        if not self.test_loader:
            raise ValueError("Test loader not provided.")

        self.model.eval()
        loss_list = []
        forecast_list = []

        loss_func = (
            custom_loss_function
            if custom_loss_function
            else getattr(self, "loss_function", None)
        )
            
        if not loss_func:
            raise ValueError("Testing requires a valid loss function.")

        if hasattr(loss_func, "reduction"):
            loss_func.reduction = "none"  # type: ignore

        with torch.no_grad():
            for batch in self.test_loader:
                raw_forecast, target = batch
                raw_forecast = raw_forecast.to(self.device)
                target = target.to(self.device)

                output = self.model(y_hat=raw_forecast)
                target_loss = loss_func(output, target)

                loss_list.append(target_loss.cpu().numpy()) 
                if self.sv_forecast:
                    forecast_list.append(output.cpu().numpy())

        if self.sv_forecast:
            with open(self.save_root / "test_forecast.pkl", "wb") as f:
                pickle.dump(np.concatenate(forecast_list, axis=0), f)

        return np.concatenate(loss_list, axis=0).mean(axis=0).reshape(-1, 1)

    def run(self, custom_loss_function: Optional[Union[nn.Module, LossBase]] = None) -> Any:
        if hasattr(self.model, "setup") and callable(self.model.setup):
            self.model.setup(self.train_loader)

        best_model_state = self.fit()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            with open(self.save_root / "best_model.pt", "wb") as f:
                torch.save(self.model.state_dict(), f)

        if self.test_loader:
            test_loss = self.test(custom_loss_function=custom_loss_function)

            with open(self.save_root / "test_loss.pkl", "wb") as f:
                pickle.dump(test_loss, f)
        else:
            test_loss = None

        return test_loss


class OptimTrainer(BaseTrainer):
    """
    Standard OptimTrainer for PyTorch models requiring gradient descent.

    Subclasses should override:
    - _setup_optimizer()
    - train_step() (if custom forward or loss calculation is needed)
    """

    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_kwargs={},
    ):
        super().__init__(model, config, train_loader, val_loader, test_loader, loss_kwargs)

        self.optimizer = self._setup_optimizer()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError("Subclasses must implement _setup_optimizer()")

    def train_step(self, batch: Any) -> torch.Tensor:
        raw_forecast, target = batch
        raw_forecast, target = raw_forecast.to(self.device), target.to(self.device)

        output = self.model(y_hat=raw_forecast)
        loss = self.loss_function(output, target)
        return loss

    def val_step(self, batch: Any) -> torch.Tensor:
        raw_forecast, target = batch
        raw_forecast, target = raw_forecast.to(self.device), target.to(self.device)

        output = self.model(y_hat=raw_forecast)
        loss = self.loss_function(output, target)
        return loss

    def train_epoch(self) -> float:
        if not self.train_loader or not self.optimizer:
            raise ValueError(
                "Train loader and optimizer must be configured for training."
            )

        self.model.train()
        train_loss = 0.0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self.train_step(batch)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(self.train_loader)

    def validate(self) -> float:
        if not self.val_loader:
            raise ValueError("Validation loader must be configured for validation.")

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                loss = self.val_step(batch)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def fit(self) -> Optional[dict]:
        if not self.train_loader or not self.val_loader:
            print("Skipping fit: Train or Validation loader not provided.")
            return None

        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        epochs = getattr(self.config, "epoch", 100)
        patience = getattr(self.config, "patience", 10)

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            status_message = f"Epoch {epoch} | Train Loss {train_loss:.6f} | Validation Loss {val_loss:.6f}"
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = deepcopy(self.model.state_dict())
                status_message += " (New best model)"
            else:
                status_message += " (No improvement)"
                patience_counter += 1

            print(status_message)

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        return best_model_state


class ClosedFormTrainer(BaseTrainer):
    """
    Closed Form Trainer strictly for models defining exact analytical solutions internally
    via a `.fit(train_y_hat, train_y_obs)` method iteratively feeding over batches.
    Requires no gradients operations.
    """

    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        loss_kwargs={},
    ):
        super().__init__(model, config, train_loader, val_loader, test_loader, loss_kwargs)

    def fit(self) -> Optional[dict]:
        if not self.train_loader:
            print(
                "Skipping fit: Train loader not provided. Assuming model is parameter-free (e.g., BottomUp, TopDown)."
            )
            return deepcopy(self.model.state_dict())

        self.model.eval()  # Gradients unneeded.

        y_hat_list, target_list = [], []
        for batch in self.train_loader:
            raw_forecast, target = batch
            y_hat_list.append(raw_forecast.to(self.device))
            target_list.append(target.to(self.device))

        y_hat = torch.cat(y_hat_list, dim=0)
        target = torch.cat(target_list, dim=0)
        
        del y_hat_list, target_list
        import gc
        gc.collect()

        # Implementations dynamically vary on `.fit()` behavior explicitly parsing batched forms logically.
        if hasattr(self.model, "fit") and callable(self.model.fit):
            self.model.fit(y_hat, target)
        else:
            print(
                "Model does not implement a .fit() method. Skipping explicit fit mapping."
            )
            
        del y_hat, target
        gc.collect()

        return deepcopy(self.model.state_dict())
