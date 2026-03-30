import torch
import torch.nn as nn
import numpy as np

class BaseReconciliationModel(nn.Module):
    """
    Base class for all Hierarchical Reconciliation models.
    Handles the common S matrix registration and basic dimensions.
    """
    S: torch.Tensor    
    def __init__(self, S: np.ndarray, device: str = "cpu"):
        super().__init__()
        
        if isinstance(S, np.ndarray):
            S_tensor = torch.from_numpy(S).float()
        else:
            S_tensor = S.float()
            
        S_tensor = S_tensor.to(device)
        self.num_node: int = S_tensor.shape[0]
        self.num_low: int = S_tensor.shape[1]
        
        # Register S as a buffer so it moves with the model (e.g., model.to('cuda'))
        self.register_buffer('S', S_tensor)
        self.max_parents, self.padded_parents, self.parent_mask = self._build_padded_adjacency()
        
    def setup(self, train_loader):
        """
        Optional hook for models to perform data-dependent initializations 
        (e.g., empirical copula extraction) before the main training loop starts.
        """
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the forward pass.")

    def _build_padded_adjacency(self):
        """
        Creates a fixed-size padded matrix and mask for variable-length parent indices.
        Returns:
            max_parents (int): Maximum number of parents any bottom-level node has
            padded_parents (Tensor): (num_low, max_parents) mapped indices
            parent_mask (Tensor): (num_low, max_parents) validity mask
        """
        active_row, active_col = torch.where(self.S.T > 0)
        adj_list = [active_col[active_row == i] for i in range(self.num_low)]
        
        max_parents = int(max(len(adj_v) for adj_v in adj_list) if adj_list else 0)
        
        padded_parents = torch.zeros((self.num_low, max_parents), dtype=torch.long, device=self.S.device)
        parent_mask = torch.zeros((self.num_low, max_parents), dtype=torch.float, device=self.S.device)
        
        for i, parents in enumerate(adj_list): 
            num_p = len(parents) 
            padded_parents[i, :num_p] = parents 
            parent_mask[i, :num_p] = 1.0 
            
        return max_parents, padded_parents, parent_mask

    def _P_matrix_init(self, init_P: str): 
        if init_P == 'ols': # Assume that base forecasts are unconstrained 
            P_full_init = torch.linalg.pinv(self.S.T @ self.S) @ self.S.T 
            P_valid_init = torch.gather(P_full_init, 1, self.padded_parents)
        elif init_P == 'random': 
            P_valid_init = torch.randn(self.num_low, self.max_parents, device=self.S.device) * 0.01
        elif init_P == 'uniform': 
            P_valid_init = torch.ones((self.num_low, self.max_parents), device=self.S.device)
        else: 
            raise ValueError(f"P matrix initialization error. Available=['ols', 'random', 'uniform'],Selected={init_P}")
        
        return P_valid_init
