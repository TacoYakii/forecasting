"""Optimization config for reconciliation fitting."""

from dataclasses import dataclass

from src.core.config import BaseConfig


@dataclass
class ReconciliationFitConfig(BaseConfig):
    """Configuration for full-data reconciliation fitting."""

    method: str = "scipy"
    scipy_method: str = "COBYLA"
    maxiter: int = 2000
    rhobeg: float = 0.1
    reg_lambda: float = 0.0
    crn_seed: int = 42
    use_crn: bool = True
