from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from src.core.config import BaseConfig

@dataclass
class BaseTrainerConfig(BaseConfig):
    """
    Common base configuration for all trainer models.
    """
    model_input_root: Path # ! Should contain train, test input dirs. validation set will be split from train set.
    observed_root: Path
    save_root: Path
    
    device: str = "cuda"
    random_seed: int = 123
    
    batch_size: int = 1024
    val_ratio: Optional[float] = None
    num_workers: int = 8
    pin_memory: bool = True
    shuffle_train: bool = True
    val_ratio: Optional[float] = 0.3

    loss_func: str = field(default="RandomCRPSLoss")
    sv_forecast: bool = True 


@dataclass
class OptimTrainerConfig(BaseTrainerConfig):
    """
    Configuration for hierarchy reconciliation models and data loading.

    Attributes:
        model_input_root: Path to root directory containing train and test input data.
        observed_root: Path to root directory for observed data.
        save_root: Path to root directory for saving training outputs.
        lr: Dictionary of learning rates for different parameters.
        loss_func: Name of the loss function to use.
        device: Device to use for training (e.g., 'cuda', 'cpu').
        random_seed: Random seed for dataset splitting and reproducibility.
        batch_size: DataLoader batch size.
        val_ratio: Validation split ratio.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        shuffle_train: Whether to shuffle training data.
        epoch: Maximum training epochs.
        patience: Early stopping patience.
    """
    lr: dict = field(default_factory=dict)
    epoch: int = 400
    patience: int = 20
    
    

@dataclass
class ClosedFormTrainerConfig(BaseTrainerConfig):
    """
    Configuration for hierarchy reconciliation models and data loading.

    Attributes:
        model_input_root: Path to root directory containing train and test input data.
        observed_root: Path to root directory for observed data.
        save_root: Path to root directory for saving training outputs.
        loss_func: Name of the loss function to use.
        device: Device to use for training (e.g., 'cuda', 'cpu').
        random_seed: Random seed for dataset splitting and reproducibility.
        batch_size: DataLoader batch size.
        val_ratio: Validation split ratio.
        num_workers: DataLoader num_workers.
        pin_memory: DataLoader pin_memory.
        shuffle_train: Whether to shuffle training data.
    """

