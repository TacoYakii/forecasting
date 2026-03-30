import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ReconciledResult:
    """
    Wrapper class to manage and export outputs of an evaluated Reconciliation model.
    """
    def __init__(
        self, 
        timestamps: List[pd.Timestamp], 
        forecasts: np.ndarray, 
        node_names: Optional[List[str]] = None
    ):
        self.timestamps = timestamps
        self.forecasts = forecasts
        self.node_names = node_names

    def save_temporal(self, res_save_dir: Union[str, Path], only_mu: bool = True):
        """
        Saves output in a directory structure driven by Temporal formats.
        node_names typically follows `{freq}_{horizon}` (e.g. "48_1").
        """
        if self.node_names is None:
            raise ValueError("node_names must be provided to use save_temporal() (e.g. ['48_1', '24_1']).")
        
        res_save_dir = Path(res_save_dir)
        
        for i, node_name in enumerate(self.node_names):
            parts = node_name.split("_")
            if len(parts) != 2:
                print(f"Skipping node {node_name} as it doesn't match expected freq_horizon layout.")
                continue
            
            freq, horizon = parts[0], parts[1]
            freq_dir = res_save_dir / str(freq)
            freq_dir.mkdir(parents=True, exist_ok=True)
            
            node_data = self.forecasts[:, i]
            
            if node_data.ndim == 1:
                df = pd.DataFrame(node_data, index=self.timestamps, columns=["mu"])
            else:
                if only_mu: 
                    node_data = node_data.mean(axis=-1)
                    df = pd.DataFrame(node_data, index=self.timestamps, columns=["mu"])
                else:
                    columns = [f"sample_{s}" for s in range(node_data.shape[1])]
                    df = pd.DataFrame(node_data, index=self.timestamps, columns=columns)
                
            df.index.name = "basis_time"
            df = df.reset_index()
            df["forecast_time"] = df["basis_time"] + pd.Timedelta(hours=float(freq) * float(horizon))
            
            # Reorder columns: basis_time, forecast_time, and then the rest (mu or samples)
            cols = ["basis_time", "forecast_time"] + [c for c in df.columns if c not in ["basis_time", "forecast_time"]]
            df = df[cols]
            
            save_path = freq_dir / f"horizon_{horizon}.csv"
            df.to_csv(save_path, index=False)

    def save_spatial(self, res_save_dir: Union[str, Path], hour_ahead: Union[int, str]):
        """
        Saves output in a directory structure driven by Spatial formats.
        node_names typically correspond to geographical or aggregation zones (e.g. "farm", "turbine_1").
        """
        if self.node_names is None:
            raise ValueError("node_names must be provided to use save_spatial() (e.g. ['farm', 'turbine_1']).")
        
        res_save_dir = Path(res_save_dir)
        
        for i, node_name in enumerate(self.node_names):
            agg_dir = res_save_dir / str(node_name)
            agg_dir.mkdir(parents=True, exist_ok=True)
            
            node_data = self.forecasts[:, i]
            
            if node_data.ndim == 1:
                df = pd.DataFrame(node_data, index=self.timestamps, columns=["forecast"])
            else:
                columns = [f"sample_{s}" for s in range(node_data.shape[1])]
                df = pd.DataFrame(node_data, index=self.timestamps, columns=columns)
                
            save_path = agg_dir / f"horizon_{hour_ahead}.csv"
            df.to_csv(save_path)

    def save_raw_dict(self, save_path: Union[str, Path]):
        """
        Collects all outputs as is and dumps them to a pickle dictionary.
        Format corresponds identically to standard input inputs:
        {"2024-01-01 00:00:00": np.array([...])}
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        out_dict = {}
        for i, ts in enumerate(self.timestamps):
            # Converting Timestamp into standard string representation
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            out_dict[ts_str] = self.forecasts[i]
            
        with open(save_path, "wb") as f:
            pickle.dump(out_dict, f)

def run_inference(
    model: nn.Module,
    forecast_dict: Dict[Union[str, pd.Timestamp], np.ndarray],
    node_names: Optional[List[str]] = None,
    pt_path: Optional[Union[str, Path]] = None,
    device: str = "cuda",
    batch_size: int = 32,
) -> ReconciledResult:
    """
    Extracts prediction input dictionaries through an evaluated Reconciliation model architecture.
    
    Args:
        model: Instantiated BaseReconciliationModel object.
        forecast_dict: Input raw forecasts mapping basis times to np.array predictions.
        node_names: Target identifiers for spatial ("Total", "Farm") or temporal ("48_1") mappings.
        pt_path: Target weights configuration, typically `.pt` file location. Left None if weights are already bundled.
        device: Operation context ("cuda" or "cpu").
        batch_size: Max elements to dynamically allocate into Torch batches.
    
    Returns:
        ReconciledResult payload incorporating parsing metadata and final evaluations natively.
    """
    # 1. Load weights onto instantiated graph if specified natively 
    if pt_path is not None:
        state_dict = torch.load(pt_path, map_location=device)
        model.load_state_dict(state_dict)
        
    model.eval()
    model.to(device)
    
    # 2. Re-configure dict arrays toward sequence arrays tracking basis times explicitly
    forecast_ts = {pd.Timestamp(k): v for k, v in forecast_dict.items()}
    valid_indices = sorted(forecast_ts.keys())
    
    if len(valid_indices) == 0:
        raise ValueError("Provided forecast_dict has 0 valid timestamps.")
        
    forecast_stack = np.stack([forecast_ts[key] for key in valid_indices])
    
    # 3. Simulate operational iterations across model boundaries 
    T = forecast_stack.shape[0]
    reconciled_list = []
    
    with torch.no_grad():
        for start_idx in range(0, T, batch_size):
            end_idx = min(start_idx + batch_size, T)
            batch_data = forecast_stack[start_idx:end_idx]
            
            batch_tensor = torch.from_numpy(batch_data).float().to(device)
            reconciled_batch = model(batch_tensor)
            
            reconciled_list.append(reconciled_batch.cpu().numpy())
            
    reconciled_forecasts = np.concatenate(reconciled_list, axis=0)
    
    # Wrap contextual metadata to payload bounds efficiently 
    result = ReconciledResult(
        timestamps=valid_indices,
        forecasts=reconciled_forecasts,
        node_names=node_names
    )
    
    return result
