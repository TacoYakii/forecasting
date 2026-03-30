from pathlib import Path
from typing import Dict, Any
import pandas as pd
import json
import pickle
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.utils.metrics import crps_numerical


class BaseEvaluator:
    """
    Evaluation using:
    - Metadata stored with each forecast
    - crps_numerical for CRPS calculation
    """
    
    def __init__(self, n_samples: int = 1000):
        """
        Initialize evaluator.
        
        Args:
            n_samples: Number of samples for numerical CRPS
        """
        self.n_samples = n_samples
    
    def evaluate_experiment(self, experiment_dir: Path, parallel: bool = True, n_workers: int = 0) -> Dict[str, Any]:
        """
        Evaluate all forecasts in an experiment, grouped by period, model, and aggregation level.
        
        Args:
            experiment_dir: Path to experiment directory
            parallel: Whether to use multiprocessing
            n_workers: Number of workers for parallel processing
            
        Returns:
            Dict with evaluation results
        """
        results = {}
        
        # Load info.json to get default column names
        info_json_path = experiment_dir / "info.json"
        
        default_y_col = "forecast_time_observed_KPX_pwr"
        default_idx_col = "forecast_time"

        if info_json_path.exists():
            with open(info_json_path) as f:
                info_data = json.load(f)
                default_y_col = info_data.get("y_col", default_y_col)
                default_idx_col = info_data.get("forecast_idx_col", default_idx_col)
                
        print(f"Evaluating experiments in {experiment_dir}...")
        
        # Find all metadata files
        metadata_files = [f for f in sorted(experiment_dir.rglob("*_metadata.json")) if f.name != "evaluation.json"]
        
        if parallel:
            if n_workers == 0:
                n_workers = min(cpu_count(), len(metadata_files))
            if n_workers == 0:
                n_workers = 1
                
            print(f"Using {n_workers} processes for evaluation")
            args_list = [(f, self.n_samples, default_y_col, default_idx_col) for f in metadata_files]
            
            with Pool(processes=n_workers) as pool:
                eval_results = list(tqdm(
                    pool.imap_unordered(self._evaluate_single_wrapper, args_list),
                    total=len(args_list),
                    desc="Evaluating forecasts"
                ))
        else:
            eval_results = []
            for metadata_file in tqdm(metadata_files, desc="Evaluating forecasts"):
                eval_results.append(self._evaluate_single(metadata_file, self.n_samples, default_y_col, default_idx_col))
                
        # Aggregate results
        for res in eval_results:
            if res is None:
                continue
                
            period_name, model_name_from_path, agg_level, csv_filename, crps = res
            
            # Make sure the period exists in results
            if period_name not in results:
                results[period_name] = {}
                
            # Make sure the model_name exists in results under period
            if model_name_from_path not in results[period_name]:
                results[period_name][model_name_from_path] = {}
            
            # Make sure the agg_level exists within the model
            if agg_level not in results[period_name][model_name_from_path]:
                results[period_name][model_name_from_path][agg_level] = {}
            
            results[period_name][model_name_from_path][agg_level][csv_filename] = crps
        
        # Save evaluation.json
        eval_path = experiment_dir / "evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✓ Saved evaluation to {eval_path}")
        
        return results

    def _evaluate_single(self, metadata_file: Path, n_samples: int, default_y_col: str, default_idx_col: str):
        """Process a single metadata file for evaluation."""
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        # Load PKL forecast distribution
        forecast_pkl = metadata_file.with_name(
            metadata_file.stem.replace("_metadata", "") + "_forecast.pkl"
        )
        
        if not forecast_pkl.exists():
            return None
        
        with open(forecast_pkl, 'rb') as f:
            forecast_dist = pickle.load(f)
        
        # Load original data
        original_data_path = metadata.get("original_data_path", "")
        if not original_data_path:
            return None
        
        obs_file = Path(original_data_path)
        if not obs_file.exists():
            return None
            
        # Get target column and index column
        y_col = metadata.get("y_col", default_y_col)
        idx_col = metadata.get("forecast_idx_col", default_idx_col)
        
        obs_df = pd.read_csv(obs_file, index_col=idx_col)
        
        # Filter valid observations
        if "is_valid" in obs_df.columns:
            obs_df = obs_df[obs_df["is_valid"] == True]
        
        # Get common index
        common_idx = forecast_dist.index.intersection(obs_df.index)
        if len(common_idx) == 0:
            return None
        
        # Get integer locations of common_idx within forecast_dist.index
        forecast_locs = forecast_dist.index.get_indexer(common_idx)
        
        # Get values and align them
        samples = forecast_dist.sample(n=n_samples)  # original shape: (T, n_samples)
        samples = samples[forecast_locs, :]  # shape: (n_common, n_samples)
        obs_values = obs_df.loc[common_idx, y_col].values
        
        # Calculate CRPS using numerical method
        crps = float(crps_numerical(obs_values, samples, need_sorting=True))
        
        # Extract actual model name, aggregation level, and period from the path
        agg_level = metadata_file.parent.name
        model_name_from_path = metadata_file.parent.parent.name
        period_name = metadata_file.parent.parent.parent.name
        csv_filename = obs_file.name
        
        return period_name, model_name_from_path, agg_level, csv_filename, crps

    def _evaluate_single_wrapper(self, args):
        """Wrapper for imap_unordered multiprocessing."""
        return self._evaluate_single(*args)


__all__ = [
    "BaseEvaluator",
]