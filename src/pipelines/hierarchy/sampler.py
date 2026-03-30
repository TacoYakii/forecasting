import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

from src.pipelines.config import HierarchyForecastSamplingConfig

class BaseForecastSampler:
    """
    Samples distributions from gathered best models to create a dataset.
    Reads `[{agg_level}/{horizon_X}_forecast.pkl]` and samples them either using quantiles or random sampling.
    """
    
    def __init__(self, config: HierarchyForecastSamplingConfig):
        self.config = config
        
    def run(self):
        """
        Executes the sampling process based on the configuration.
        """
        base_dir = Path(self.config.base_dir)
        output_dir = Path(self.config.output_dir)
        
        target_periods = self.config.target_period
        if isinstance(target_periods, str):
            target_periods = [target_periods]
            
        for period in target_periods:
            period_base_dir = base_dir / period
            period_out_dir = output_dir / period
            
            if not period_base_dir.exists():
                raise FileNotFoundError(f"Period directory {period_base_dir} does not exist.")
                
            period_out_dir.mkdir(parents=True, exist_ok=True)
                
            # Dictionary to collect dataframes for each combination
            # Key: combo, Value: DataFrame(index, columns=samples)
            combo_dfs = {}
            
            for combo in tqdm(self.config.target_combinations, desc=f"Loading {period} samples"):
                # combo example: "4/horizon_1", "12/horizon_3"
                parts = combo.split("/")
                if len(parts) != 2:
                    raise ValueError(f"Invalid combination format '{combo}'. Expected 'agg_level/horizon_X'.")
                    
                agg_level, horizon_name = parts
                
                pkl_name = f"{horizon_name}_forecast.pkl"
                pkl_path = period_base_dir / agg_level / pkl_name
                
                if not pkl_path.exists():
                    raise FileNotFoundError(f"Forecast file {pkl_path} does not exist.")
                    
                # 1. Load the ParametricDistribution
                with open(pkl_path, 'rb') as f:
                    dist = pickle.load(f)
                
                # Assume dist.index exists and maps to index_col in original data
                # Typically, target index is dist.index
                
                # 2. Sample
                if self.config.sampling_method == "quantile":
                    if self.config.quantile_start is None or self.config.quantile_end is None:
                        raise ValueError("quantile_start and quantile_end must be specified for quantile sampling.")
                    quantiles = np.linspace(self.config.quantile_start, self.config.quantile_end, self.config.n_samples).tolist()
                    sampled_array = dist.ppf(q=quantiles) 
                    
                elif self.config.sampling_method == "random":
                    sampled_array = dist.sample(n=self.config.n_samples)
                    
                else:
                    raise ValueError(f"Unknown sampling method: {self.config.sampling_method}")
                
                # Build DataFrame to align indices later
                # Shape: (num_time_steps, num_samples)
                df = pd.DataFrame(sampled_array, index=dist.base_idx)
                combo_dfs[combo] = df
                
            if not combo_dfs:
                raise ValueError(f"No valid combinations found for period {period}.")
                
            print(f"Aligning and building dictionary for {period}...")
            
            # 3. Find exactly matching indices across all valid combinations
            # We take the intersection to ensure no NaNs in the final arrays
            common_idx = None
            for combo in self.config.target_combinations:
                idx = combo_dfs[combo].index
                if common_idx is None:
                    common_idx = idx
                else:
                    common_idx = common_idx.intersection(idx)
                    
            if common_idx is None or len(common_idx) == 0:
                raise ValueError(f"No overlapping indices found across combinations for period {period}.")
                
            # 4. Build the final dictionary
            # Key: index_col value (e.g. basis_time)
            # Value: np.ndarray of shape (len(target_combinations), n_samples)
            sampled_dict = {}
            for t_idx in common_idx:
                # Extract the row for this time step across all combinations
                # and stack them vertically to shape (num_combos, num_samples)
                rows = []
                for combo in self.config.target_combinations:
                    row_array = combo_dfs[combo].loc[t_idx].values # shape: (n_samples,)
                    rows.append(row_array)
                    
                stacked_array = np.stack(rows, axis=0) # shape: (len(target_combinations), num_samples)
                sampled_dict[t_idx] = stacked_array
                

        
            metadata: Dict[str, Any] = {
                "index_col": self.config.index_col,
                "sampling_method": self.config.sampling_method,
                "n_samples": self.config.n_samples,
                "quantile_start": self.config.quantile_start if self.config.sampling_method == "quantile" else "N/A",
                "quantile_end": self.config.quantile_end if self.config.sampling_method == "quantile" else "N/A",
                "target_combinations_order": self.config.target_combinations,
                "shape_per_idx": [len(self.config.target_combinations), self.config.n_samples],
                "total_timestamps": len(common_idx),
                "base_dir": self.config.base_dir
            }
            
            # 6. Save Output    
            out_pkl = period_out_dir / "sampled_forecasts.pkl"
            with open(out_pkl, 'wb') as f:
                pickle.dump(sampled_dict, f)
                
            out_meta = period_out_dir / "sampled_metadata.json"
            with open(out_meta, 'w') as f:
                json.dump(metadata, f, indent=4)
                
        print(f"Sampling completed. Results stacked and saved to {output_dir}")

__all__ = ["BaseForecastSampler"]
