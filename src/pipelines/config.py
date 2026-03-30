"""
Pipeline configuration classes.
"""

from pathlib import Path
from typing import Dict, List, Any, Union
import json
from dataclasses import dataclass, field, asdict
from src.core.config import BaseConfig

@dataclass
class HierarchyForecastConfig(BaseConfig):
    """Configuration for hierarchy forecasting.
    
    Attributes:
        data_root: Path to training data directory
        result_root: Path to results directory
        aggregation_levels: List of aggregation levels to process (e.g., ["3", "4", "12", "48"])
        y_col: Target column name
        forecast_idx_col: Forecast index column name
        model_distributions: Distribution for each model (e.g., {"catboost": "normal", "ngboost": "laplace"})
        period_setting: Start & End datetime of each period (e.g., {"training": [("2024-01-01", "2025-01-01") ("2025-01-01", "2026-01-01")], "validation": [("2025-01-01", "2026-01-01"), ("2026-01-01", "2027-01-01")]})
        exogeneous_columns: Exogeneous columns from JSON
        models: List of models to run
        periods: List of periods to process
        output_folder: Output folder name
    """
    
    # Paths
    data_root: Path
    result_root: Path
    
    # Aggregation levels to process
    aggregation_levels: List[str] = field(default_factory=list)
    
    # Model settings
    y_col: str = "forecast_time_observed_KPX_pwr"
    forecast_idx_col: str = "forecast_time"
    
    # Distribution for each model (user-specified)
    model_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict) # {model_nm: model_hyperparameter}
    
    # Period setting 
    period_setting: Dict[str, Any] = field(default_factory=dict)
    exogeneous_columns: List[str] = field(default_factory=list)
    
    
    # Output
    output_folder: str = "Hierarchy_base_forecast"
    
    @classmethod
    def from_json_files(
        cls,
        data_root: Path,
        result_root: Path,
        aggregation_levels: List[str],
        period_setting_path: Path,
        exogeneous_columns_path: Path,
    ) -> "HierarchyForecastConfig":
        """Create config from JSON files."""
        with open(period_setting_path, 'r') as f:
            period_setting = json.load(f)
        
        with open(exogeneous_columns_path, 'r') as f:
            exogeneous_columns = json.load(f)
        
        return cls(
            data_root=data_root,
            result_root=result_root,
            aggregation_levels=aggregation_levels,
            period_setting=period_setting,
            exogeneous_columns=exogeneous_columns,
        )

def create_config(
    data_root: Path,
    result_root: Path,
    aggregation_levels: List[str],
    period_setting_path: Path,
    exogeneous_columns_path: Path,
) -> HierarchyForecastConfig:
    """
    Create HierarchyForecastConfig from JSON files.
    
    Convenience function for common use case.
    """
    return HierarchyForecastConfig.from_json_files(
        data_root=data_root,
        result_root=result_root,
        aggregation_levels=aggregation_levels,
        period_setting_path=period_setting_path,
        exogeneous_columns_path=exogeneous_columns_path,
    )

@dataclass 
class HierarchyForecastCoordinatorConfig(BaseConfig):
    """
    Configuration for hierarchy forecast coordinator.
    
    Attributes:
        evaluation_source: Path to evaluation.json.
        target_period: The period to use for comparison (e.g., "validation" or "test").
        save_dir: Path to save the results.
    """
    evaluation_source: Union[str, Path] = "" 
    target_period: str = "validation"
    gathering_period: Union[str, List[str]] = "" 
    save_dir: Union[str, Path] = "" 

    best_models: Dict[str, Dict[str, str]] = field(default_factory=dict)


@dataclass
class HierarchyForecastSamplingConfig(BaseConfig):
    """
    Configuration for sampling from the gathered best models to create a dataset.
    
    Attributes:
        base_dir: Path to the directory where best models were gathered (e.g., Output of coordinator).
        output_dir: Path to save the sampled dataset.
        n_samples: Number of samples to draw per forecast distribution.
        sampling_method: 'quantile' or 'random'.
        quantiles: List of quantiles to sample if sampling_method is 'quantile' (e.g., [0.1, 0.5, 0.9]). If not provided, it will generate evenly spaced quantiles based on n_samples.
        target_period: Which periods to load and sample from (e.g., ["validation", "test"]).
        target_combinations: List of combinations to load in the format `"{aggregation_level}/horizon_{X}"`.
        index_col: The common datetime index column name to align everything on.
    """
    base_dir: Union[str, Path] = ""
    output_dir: Union[str, Path] = ""
    
    target_combinations: List[str] = field(default_factory=list)
    target_period: Union[str, List[str]] = field(default_factory=lambda: ["validation", "test"])
    quantile_start: float = 0.001
    quantile_end: float = 0.999
    n_samples: int = 1000
    sampling_method: str = "random" # 'quantile' or 'random'
    index_col: str = "basis_time"

