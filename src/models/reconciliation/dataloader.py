from typing import Dict, Optional, Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split


class HierarchyDataset(Dataset):
    """
    PyTorch Dataset for hierarchical time series data.

    forecast와 observed 데이터를 쌍으로 관리하는 Dataset
    """

    def __init__(
        self,
        forecast_stack: np.ndarray,
        observed_stack: np.ndarray,
        indices: List[pd.Timestamp],
    ):
        """
        Args:
            forecast_stack: Forecast data array (N, forecast_dim)
            observed_stack: Observed data array (N, observed_dim)
            indices: Timestamp indices
        """
        if len(forecast_stack) != len(observed_stack):
            raise ValueError(
                f"Forecast and observed data must have same length. "
                f"Got {len(forecast_stack)} vs {len(observed_stack)}"
            )

        self.forecast = forecast_stack
        self.observed = observed_stack
        self.indices = indices

    def __len__(self) -> int:
        return len(self.forecast)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        forecast = torch.from_numpy(self.forecast[index]).float()
        observed = torch.from_numpy(self.observed[index]).float()

        return forecast, observed

    def get_timestamp(self, idx: int) -> pd.Timestamp:
        return self.indices[idx]

    def get_all_timestamps(self) -> List[pd.Timestamp]:
        return self.indices.copy()


class HierarchyDataLoader(DataLoader):
    """
    DataLoader for HierarchyDataset

    DataLoader를 상속하여 커스텀 메서드 추가
    """

    def __init__(self, dataset: Union[HierarchyDataset, Subset], **kwargs):
        """
        Args:
            dataset: HierarchyDataset instance or subset(HierarchyDataset)
            **kwargs: DataLoader의 다른 파라미터들
                        (batch_size, shuffle, num_workers 등)
        """

        super().__init__(dataset, **kwargs)
        self._base_dataset = self._find_base_dataset(
            dataset
        )  # random_split = wrapper class (never copies the whole original instance, it saves the indices of it to split the dataset)

    @staticmethod
    def _find_base_dataset(dataset) -> HierarchyDataset:
        """Subset으로 감싸진 경우 원본 HierarchyDataset 찾기"""
        current = dataset
        while hasattr(current, "dataset"):
            current = current.dataset

        if not isinstance(current, HierarchyDataset):
            raise TypeError(
                f"Base dataset must be HierarchyDataset, got {type(current)}"
                f"Make sure you're using HierarchyDataset or Subset(HierarchyDataset)."
            )

        return current

    def get_timestamp(self, idx: int) -> pd.Timestamp:
        """
        DataLoader의 인덱스로 타임스탬프 반환

        주의: random_split 사용 시 idx는 split된 데이터셋 기준
        """

        if isinstance(self.dataset, Subset):
            actual_idx = self.dataset.indices[idx]
        else:
            actual_idx = idx

        return self._base_dataset.get_timestamp(actual_idx)

    def get_all_timestamps(self) -> List[pd.Timestamp]:
        """현재 DataLoader가 사용하는 모든 타임스탬프 반환"""
        if isinstance(self.dataset, Subset):
            return [
                self._base_dataset.get_timestamp(idx) for idx in self.dataset.indices
            ]
        else:
            return self._base_dataset.get_all_timestamps()

    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """데이터의 시작일과 종료일 반환"""
        timestamps = self.get_all_timestamps()
        return min(timestamps), max(timestamps)

    @property
    def base_dataset(self) -> HierarchyDataset:
        """원본 HierarchyDataset 접근"""
        return self._base_dataset


class HierarchyDataProcessor:
    """
    데이터 전처리를 담당하는 클래스

    forecast, observed, is_valid의 교집합 key를 찾고
    valid한 데이터만 필터링
    """

    @staticmethod
    def find_common_valid_indices(
        forecast: Dict[pd.Timestamp, np.ndarray],
        observed: Dict[pd.Timestamp, np.ndarray],
        is_valid: Dict[pd.Timestamp, np.ndarray],
    ) -> List[pd.Timestamp]:
        """
        Find common + valid indices

        Args:
            forecast: Forecast data dictionary
            observed: Observed data dictionary
            is_valid: Validation mask dictionary

        Returns:
            List of valid timestamps (sorted)
        """
        common_keys = set(forecast.keys()) & set(observed.keys()) & set(is_valid.keys())

        if len(common_keys) == 0:
            raise ValueError(
                "No common timestamps found among forecast, observed, and is_valid"
            )

        print(f"Common timestamps found: {len(common_keys)}")

        valid_keys = [key for key in common_keys if np.all(is_valid[key])]

        if len(valid_keys) == 0:
            raise ValueError(
                "No valid data found (all timestamps have at least one False in is_valid)"
            )

        print(f"Valid timestamps after filtering: {len(valid_keys)}")

        return sorted(valid_keys)

    @staticmethod
    def validate_shapes(
        forecast: Dict[pd.Timestamp, np.ndarray],
        observed: Dict[pd.Timestamp, np.ndarray],
        indices: List[pd.Timestamp],
    ) -> None:
        """
        Validate the shape of samples

        Args:
            forecast: Forecast data dictionary
            observed: Observed data dictionary
            indices: Timestamps to validate
        """
        if len(indices) == 0:
            return

        first_key = indices[0]  # 첫 번째 샘플의 shape을 기준으로 사용
        sample_shape = forecast[first_key].shape[0]

        for key in indices:
            if forecast[key].shape[0] != sample_shape:
                raise ValueError(
                    f"Inconsistent forecast shape at {key}: "
                    f"expected {sample_shape}, got {forecast[key].shape}"
                )
            if observed[key].shape[0] != sample_shape:
                raise ValueError(
                    f"Inconsistent observed shape at {key}: "
                    f"expected {sample_shape}, got {observed[key].shape}"
                )

        print(f"Shape validation passed: sample_shape={sample_shape}")

    @staticmethod
    def prepare_data(
        forecast: Dict[Union[pd.Timestamp, str], np.ndarray],
        observed: Dict[Union[pd.Timestamp, str], np.ndarray],
        is_valid: Dict[Union[pd.Timestamp, str], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Data preprocessing pipeline

        Args:
            forecast: Forecast data dictionary
            observed: Observed data dictionary
            is_valid: Validation mask dictionary

        Returns:
            forecast_stack: Stacked forecast data (N, forecast_dim)
            observed_stack: Stacked observed data (N, observed_dim)
            indices: List of timestamps (length N)
        """
        # 1. Convert keys to pandas Timestamp
        forecast_ts = {pd.Timestamp(k): v for k, v in forecast.items()}
        observed_ts = {pd.Timestamp(k): v for k, v in observed.items()}
        is_valid_ts = {pd.Timestamp(k): v for k, v in is_valid.items()}

        print(
            f"Initial lengths - forecast: {len(forecast_ts)} \n \
            observed: {len(observed_ts)}, is_valid: {len(is_valid_ts)}"
        )

        # 2. Find common indices & valid samples
        valid_indices = HierarchyDataProcessor.find_common_valid_indices(
            forecast_ts, observed_ts, is_valid_ts
        )

        # 3. Validate shape
        HierarchyDataProcessor.validate_shapes(forecast_ts, observed_ts, valid_indices)

        # 4. Stack arrays
        forecast_stack = np.stack([forecast_ts[key] for key in valid_indices])
        observed_stack = np.stack([observed_ts[key] for key in valid_indices])

        #
        observed_stack = observed_stack[:, :, None]

        print(f"Final dataset size: {len(valid_indices)} samples")

        return forecast_stack, observed_stack, valid_indices


def create_hierarchy_dataloaders(
    forecast: Dict[Union[pd.Timestamp, str], np.ndarray],
    observed: Dict[Union[pd.Timestamp, str], np.ndarray],
    is_valid: Dict[Union[pd.Timestamp, str], np.ndarray],
    val_ratio: Optional[float] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    random_seed: int = 42,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Factory function for creating DataLoader(s)

    Args:
        forecast: Forecast data dictionary {timestamp: array}
        observed: Observed data dictionary {timestamp: array}
        is_valid: Validation mask dictionary {timestamp: bool_array}
        val_ratio: Validation split ratio (0.0~1.0), None for no split
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to use pinned memory (faster GPU transfer)
        random_seed: Random seed for reproducibility

    Returns:
        If val_ratio is None: single DataLoader
        If val_ratio is provided: (train_loader, val_loader)

    Example:
        >>> # Training (train/val split)
        >>> train_loader, val_loader = create_hierarchy_dataloaders(
        ...     forecast=forecast_dict,
        ...     observed=observed_dict,
        ...     is_valid=is_valid_dict,
        ...     val_ratio=0.2,
        ...     batch_size=32
        ... )
        >>>
        >>> # Validating (split 없음)
        >>> full_loader = create_hierarchy_dataloaders(
        ...     forecast=forecast_dict,
        ...     observed=observed_dict,
        ...     is_valid=is_valid_dict,
        ...     val_ratio=None,
        ...     shuffle=False
        ... )
    """
    # 1. Preprocessing
    forecast_stack, observed_stack, indices = HierarchyDataProcessor.prepare_data(
        forecast, observed, is_valid
    )

    # 2. Load Dataset
    full_dataset = HierarchyDataset(
        forecast_stack=forecast_stack, observed_stack=observed_stack, indices=indices
    )

    # 3. Train/Val split
    if val_ratio is not None:
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        val_size = int(len(full_dataset) * val_ratio)
        train_size = len(full_dataset) - val_size

        generator = torch.Generator().manual_seed(random_seed)
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        print(f"Dataset split - train: {train_size}, val: {val_size}")

        # Create Dataloader
        train_loader = HierarchyDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        val_loader = HierarchyDataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Didn't shuffled the validation dataset while training
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        return train_loader, val_loader

    else:
        return HierarchyDataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
