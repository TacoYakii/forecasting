import json
from pathlib import Path
from typing import Dict, Union, Any
import shutil
from src.pipelines.config import HierarchyForecastCoordinatorConfig

def get_best_models(
    evaluation_source: Union[str, Path, Dict[str, Any]], 
    target_period: str = "validation"
) -> Dict[str, Dict[str, str]]:
    """
    Finds the best performing model (lowest CRPS) for each aggregation level and horizon 
    based on the evaluation results of a specific period.
    
    Args:
        evaluation_source: Path to evaluation.json or the already loaded dictionary.
        target_period: The period to use for comparison (e.g., "validation" or "test").
        
    Returns:
        A dictionary mapping aggregation level -> horizon file -> best model name.
        Example: {'1': {'horizon_1.csv': 'ngboost', 'horizon_2.csv': 'catboost'}, '2': {...}}
    """
    target_period = "validation"
    # 1. Load data
    if isinstance(evaluation_source, (str, Path)):
        with open(evaluation_source, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
    else:
        eval_data = evaluation_source
        
    if target_period not in eval_data:
        raise ValueError(f"Period '{target_period}' not found in the evaluation data.")
        
    period_data = eval_data[target_period]
    # 2. Reorganize data for comparison
    # Structure: {agg_level: {horizon: {model_name: crps}}}
    comparison: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    for model_name, agg_levels in period_data.items():
        for agg_level, horizons in agg_levels.items():
            if agg_level not in comparison:
                comparison[agg_level] = {}
                
            for horizon, crps in horizons.items():
                if horizon not in comparison[agg_level]:
                    comparison[agg_level][horizon] = {}
                    
                comparison[agg_level][horizon][model_name] = crps
                
    # 3. Find the best model for each aggregation level and horizon
    best_models: Dict[str, Dict[str, str]] = {}
    
    for agg_level, horizons in comparison.items():
        best_models[agg_level] = {}
        for horizon, model_scores in horizons.items():
            best_model = min(model_scores.items(), key=lambda x: x[1])[0]
            best_models[agg_level][horizon] = best_model
            
    return best_models

def gather_best_models(
    config: HierarchyForecastCoordinatorConfig
) -> None:
    """
    Gathers the best models for each aggregation level and horizon based on the evaluation results.
    Copies the corresponding *_forecast.pkl files to the specified save_dir grouped by gathering_period.
    
    Args:
        config: Configuration for the hierarchy forecast coordinator.
        
    Returns:
        None
    """
    best_models = get_best_models(config.evaluation_source, config.target_period) 
    
    if not config.save_dir:
        print("save_dir is not provided. Skipping file copy.")
        return 
        
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify the base experiment directory from evaluation_source
    eval_path = Path(config.evaluation_source)
    exp_dir = eval_path.parent
        
    gathering_periods = config.gathering_period
    if isinstance(gathering_periods, str):
        gathering_periods = [gathering_periods] if gathering_periods else []
        
    for period in gathering_periods:
        period_save_dir = save_dir / period
        
        for agg_level, horizons in best_models.items():
            level_save_dir = period_save_dir / agg_level
            level_save_dir.mkdir(parents=True, exist_ok=True)
            
            for horizon_csv, best_model_name in horizons.items():
                # Derive the forecast.pkl filename from the horizon csv name
                # e.g horizon_1.csv -> horizon_1_forecast.pkl
                horizon_stem = Path(horizon_csv).stem
                pkl_filename = f"{horizon_stem}_forecast.pkl"
                
                # Source path: {exp_dir}/{period}/{model}/{level}/{pkl_filename}
                src_pkl = exp_dir / period / best_model_name / agg_level / pkl_filename
                
                # Destination path
                dst_pkl = level_save_dir / pkl_filename
                
                if src_pkl.exists():
                    shutil.copy2(src_pkl, dst_pkl)
                    print(f"Copied {src_pkl.relative_to(exp_dir)} to {dst_pkl.relative_to(save_dir)}")
                else:
                    print(f"Warning: Source file not found: {src_pkl}")
    
    config.best_models = best_models 
    
    config.save(save_dir / "config.json") 
    print(f"Best models saved to {save_dir}")
