"""
Training data builder package.

Combines preprocessed NWP CSVs and SCADA data into final training datasets.
Supports two output formats:
    - Per-horizon: one CSV per forecast horizon (horizon_0.csv ~ horizon_48.csv)
    - Continuous: single time-indexed CSV for time-series models
"""

from .config import (
    ScadaConfig,
    NWPSourceConfig,
    TurbineInfo,
    TrainingDataConfig,
)
from .pipeline import TrainingDataPipeline

__all__ = [
    "ScadaConfig",
    "NWPSourceConfig",
    "TurbineInfo",
    "TrainingDataConfig",
    "TrainingDataPipeline",
]
