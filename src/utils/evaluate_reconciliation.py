import torch
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Union
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

def evaluate_and_save_loss(
    model: nn.Module,
    pt_path: Union[str, Path],
    test_loader: DataLoader,
    loss_obj: nn.Module,
    save_dir: Union[str, Path],
    save_forward: bool = False, 
    device: str = "cuda"
):
    """
    Load a reconciliation model and its .pt weights, evaluate on test_loader using loss_obj,
    and save the test loss as test_loss.pkl in save_dir.
    
    Args:
        model: Model object (e.g., CVReconciliation)
        pt_path: Path to the .pt file containing model weights
        test_loader: DataLoader yielding (raw_forecast, target) batches
        loss_obj: Loss object from src.utils.loss (e.g., RandomCRPSLoss)
        save_dir: Directory where test_loss.pkl will be saved
        device: Device to use for evaluation (default: "cuda")
    """
    # 1. Load weights
    pt_path = Path(pt_path)
    state_dict = torch.load(pt_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 2. Set to evaluation mode and move to device
    model.eval()
    model.to(device)
    
    all_losses = []
    
    # 3. Evaluate on test_loader
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating test set"):
            # typically batch contains (raw_forecast, target)
            raw_forecast, target = batch
            raw_forecast = raw_forecast.to(device)
            target = target.to(device)
            
            # Forward pass
            reconciled_forecast = model(raw_forecast)
            
            # Calculate loss (depending on reduction, loss could be a scalar or a tensor)
            loss = loss_obj(reconciled_forecast, target)
            
            # Handle both scalar and tensor losses (take mean for tracking purposes if not already reduced)
            all_losses.append(loss.cpu().numpy())
    
    loss_res = np.concatenate(all_losses, axis=0).mean(axis=0).reshape(-1, 1) 
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "test_loss.pkl"
    
    with open(save_path, "wb") as f:
        pickle.dump(loss_res, f)

    return loss_res 

