from typing import Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from src.trainers.config import BaseTrainerConfig


class HierarchyDataset(Dataset):
    def __init__(
        self,
        forecast_dict: dict,
        observed_dict: dict,
        keys: list
    ):
        """
        Args:
            forecast_dict: Dictionary containing Forecast data
            observed_dict: Dictionary containing Observed data
            keys: List of valid sorted keys
        """
        self.forecast = forecast_dict
        self.observed = observed_dict
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.keys[index]
        forecast_tensor = torch.from_numpy(self.forecast[key]).float()
        observed_tensor = torch.from_numpy(self.observed[key]).float()

        return forecast_tensor, observed_tensor


class HierarchyDataModule:
    """
    Data module for managing hierarchical time series datasets.
    Handles data preparation, splitting, and DataLoader creation for reconciliation models.
    """

    def __init__(
        self,
        forecast: Dict[Union[pd.Timestamp, str], np.ndarray],
        observed: Dict[Union[pd.Timestamp, str], np.ndarray],
        is_valid: Dict[Union[pd.Timestamp, str], np.ndarray],
        config: BaseTrainerConfig,
    ):
        self.forecast_dict = forecast
        self.observed_dict = observed
        self.is_valid_dict = is_valid

        self.config = config

        self.full_dataset: Optional[HierarchyDataset] = None
        self.train_dataset: Optional[HierarchyDataset] = None
        self.val_dataset: Optional[HierarchyDataset] = None

    def _prepare_data(self) -> list:
        """
        Find common keys among forecast, observed, and is_valid dictionaries.
        Filter out invalid data based on is_valid.
        Returns the sorted valid keys. 
        """
        common_keys = (
            set(self.forecast_dict.keys())
            & set(self.observed_dict.keys())
            & set(self.is_valid_dict.keys())
        )

        if not common_keys:
            raise ValueError("No common timestamps found")

        valid_keys = [k for k in common_keys if np.all(self.is_valid_dict[k])]

        if not valid_keys:
            raise ValueError("No valid data found")

        sorted_keys = sorted(valid_keys)
        
        first_obs = self.observed_dict[sorted_keys[0]]
        first_for = self.forecast_dict[sorted_keys[0]]
        
        if first_obs.ndim == 1: 
            for k in sorted_keys: 
                self.observed_dict[k] = self.observed_dict[k][:, None]
        elif first_obs.ndim != 2: 
            raise ValueError(f"observed_items must be 1D or 2D, got {first_obs.ndim}D")

        first_obs_up = self.observed_dict[sorted_keys[0]]
        if first_for.shape[:1] != first_obs_up.shape[:1]:
            raise ValueError("Shape mismatch between forecast and observed")

        return sorted_keys

    def setup(self):
        """
        Prepare datasets and handle train/val splitting logic.
        """
        sorted_keys = self._prepare_data()

        self.full_dataset = HierarchyDataset(
            forecast_dict=self.forecast_dict,
            observed_dict=self.observed_dict,
            keys=sorted_keys
        )

        if self.config.val_ratio is not None:
            if not (0.0 < self.config.val_ratio < 1.0):
                raise ValueError(
                    f"val_ratio must be between 0 and 1, got {self.config.val_ratio}"
                )

            val_size = int(len(self.full_dataset) * self.config.val_ratio)
            train_size = len(self.full_dataset) - val_size

            generator = torch.Generator().manual_seed(self.config.random_seed)
            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset, [train_size, val_size], generator=generator
            )

    def _create_dataloader(
        self, dataset: Optional[HierarchyDataset], shuffle: bool, error_msg: str
    ) -> DataLoader:
        if dataset is None:
            raise ValueError(error_msg)

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.train_dataset,
            shuffle=self.config.shuffle_train,
            error_msg="Setup must be called before requesting dataloaders, and val_ratio must be provided.",
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.val_dataset,
            shuffle=False,
            error_msg="Setup must be called before requesting dataloaders, and val_ratio must be provided.",
        )

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.full_dataset,
            shuffle=False,
            error_msg="Setup must be called before requesting dataloaders.",
        )


# Kept for backward compatibility if needed, but DataModule usage is preferred.
def create_hierarchy_dataloaders(
    forecast: Dict[Union[pd.Timestamp, str], np.ndarray],
    observed: Dict[Union[pd.Timestamp, str], np.ndarray],
    is_valid: Dict[Union[pd.Timestamp, str], np.ndarray],
    config: BaseTrainerConfig,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Factory function leveraging the new DataModule design.
    """
    datamodule = HierarchyDataModule(forecast, observed, is_valid, config)
    datamodule.setup()

    if config.val_ratio is not None:
        return datamodule.train_dataloader(), datamodule.val_dataloader()
    else:
        return datamodule.test_dataloader()
