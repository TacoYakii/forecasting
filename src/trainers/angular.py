import torch
import torch.optim as optim
from src.core.trainer import OptimTrainer
from src.trainers.config import OptimTrainerConfig
from src.datasets import HierarchyDataModule, get_data

import numpy as np 
from typing import Optional, Callable, Any 


class AngularTrainer(OptimTrainer):
    """
    Specific trainer implementation for Angular / Horizontal / Vertical Reconciliation Models.
    Handles specialized learning rate allocations for P and angle parameters.
    """
    def __init__(
        self, 
        model,
        trainer_config: OptimTrainerConfig,
        loss_kwargs: dict = {}
    ):
        test_data, train_validation_data, observed_data, is_valid_data = get_data(trainer_config.model_input_root, trainer_config.observed_root)

        train_data_module = HierarchyDataModule(
            forecast=train_validation_data,
            observed=observed_data,
            is_valid=is_valid_data,
            config=trainer_config
        )
        train_data_module.setup()

        test_data_module = HierarchyDataModule(
            forecast=test_data,
            observed=observed_data,
            is_valid=is_valid_data,
            config=trainer_config
        )
        test_data_module.setup()

        train_loader = train_data_module.train_dataloader()
        val_loader = train_data_module.val_dataloader()
        test_loader = test_data_module.test_dataloader()
        
        super().__init__(model, trainer_config, train_loader, val_loader, test_loader, loss_kwargs=loss_kwargs)
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        from typing import cast
        import torch.nn as nn

        config = self.config
        model = self.model

        # Ensure dynamic parameters are explicitly typed as nn.Parameter for the IDE
        p_valid = cast(nn.Parameter, getattr(model, "P_valid"))
        angle_raw = cast(nn.Parameter, getattr(model, "angle_raw"))

        # mode == 'angular' requires optimizing both P_valid and angle_raw using different LRs
        lr_P = getattr(config, "lr", {}).get("P", 1e-2)
        
        # Angle usually needs a larger learning rate because gradients can shrink after activation (like sigmoid)
        lr_angle = getattr(config, "lr", {}).get("angle", 5e-2)

        if getattr(config, "mode", "angular") == "angular":
            angle_raw.requires_grad = True
            return optim.Adam(
                [
                    {"params": p_valid, "lr": lr_P},
                    {"params": angle_raw, "lr": lr_angle},
                ]
            )
        else:
            # Non-angular modes only optimize P_valid parameter
            angle_raw.requires_grad = False
            return optim.Adam([p_valid], lr=lr_P)

    def test(self, custom_loss_function: Optional[Callable] = None) -> Any:
        if not self.test_loader:
            raise ValueError("Test loader not provided.")

        self.model.eval()
        loss_list = []
        loss_func = custom_loss_function if custom_loss_function else self.loss_function
        loss_func.reduction = "none" # type: ignore

        with torch.no_grad():
            for batch in self.test_loader:
                raw_forecast, target = batch
                raw_forecast = raw_forecast.to(self.device)
                target = target.to(self.device)

                output = self.model(y_hat=raw_forecast)
                target_loss = loss_func(output, target)

                loss_list.append(target_loss.cpu().numpy())

        return np.concatenate(loss_list, axis=0).mean(axis=0).reshape(-1, 1)