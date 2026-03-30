import torch
from src.core.trainer import ClosedFormTrainer
from src.trainers.config import ClosedFormTrainerConfig
from src.datasets import HierarchyDataModule, get_data
from typing import Optional


class MinTTrainer(ClosedFormTrainer):
    """
    Trainer suitable for MinT reconciliation models which implement a closed-form solution.
    Utilizes ClosedFormTrainer mapping parameters directly using explicit training subsets.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trainer_config: ClosedFormTrainerConfig,
        train_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        loss_kwargs: dict = {}
    ):

        if train_loader is None or test_loader is None:
            test_data, train_validation_data, observed_data, is_valid_data = get_data(trainer_config.model_input_root, trainer_config.observed_root)

            # MinT utilizes training dataset natively to reconstruct error-dependent structural matrices via ClosedFormTrainer.fit()
            train_data_module = HierarchyDataModule(
                forecast=train_validation_data,
                observed=observed_data,
                is_valid=is_valid_data,
                config=trainer_config
            )
            train_data_module.setup()

            # Build testing dataset for final coherent quantile testing natively.
            test_data_module = HierarchyDataModule(
                forecast=test_data,
                observed=observed_data,
                is_valid=is_valid_data,
                config=trainer_config
            )
            test_data_module.setup()

            # Uses train_dataloader essentially for scaling projection metrics organically over batches when required.
            train_loader = train_data_module.train_dataloader()
            test_loader = test_data_module.test_dataloader()

        super().__init__(model, trainer_config, train_loader=train_loader, val_loader=None, test_loader=test_loader, loss_kwargs=loss_kwargs)
