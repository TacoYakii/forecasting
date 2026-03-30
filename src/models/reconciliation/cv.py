import torch 
import torch.nn as nn 
import torch.nn.functional as F
from src.core.reconciliation_model import BaseReconciliationModel
from dataclasses import dataclass
from src.core.config import BaseConfig
from src.models.reconciliation.copula import PermutationEmpiricalCopula


@dataclass
class CVReconciliationConfig(BaseConfig):
    init_P: str = "ols" 
    constraint_P: str = "linear" 
    aggregation_method: str = "quantile"

class CVReconciliation(BaseReconciliationModel, PermutationEmpiricalCopula):
    empirical_distribution: torch.Tensor 
    def __init__(self, S, device, config: CVReconciliationConfig):
        BaseReconciliationModel.__init__(self, S, device)
        self.init_P:str = config.init_P
        self.constraint_P:str = config.constraint_P
        self.aggregation_method:str = config.aggregation_method

        P_valid_init = self._P_matrix_init(self.init_P) 
        self.P_valid = nn.Parameter(P_valid_init) 
        self.init_empirical_copula(self.S.device)

    def setup(self, train_loader):
        # Used only lowest node data to fit empirical copula 
        if train_loader is None:
            return
            
        y_hat_list, target_list = [], []
        for batch in train_loader:
            raw_forecast, target = batch
            y_hat_list.append(raw_forecast[:, -self.num_low:, :].to(self.S.device))
            target_list.append(target[:, -self.num_low:, :].to(self.S.device))
            
        y_hat_all = torch.cat(y_hat_list, dim=0)
        target_all = torch.cat(target_list, dim=0)
        
        self.fit_empirical_copula(y_hat_all, target_all)

    def get_P(self): 
        if self.constraint_P == "linear":  # sum to one & positive
            inf_mask = self.P_valid.masked_fill(self.parent_mask == 0, float("-inf"))
            return F.softmax(inf_mask, dim=1)

        elif self.constraint_P == "sum": # Only sum to one
            p_val = self.P_valid
            p_valid_masked = p_val * self.parent_mask
            return p_valid_masked / (p_valid_masked.sum(dim=1, keepdim=True) + 1e-9)

        elif self.constraint_P == "unconstrained":
            return self.P_valid * self.parent_mask
        
        else: 
            raise ValueError(f"P constraint error. Available=['linear', 'sum', 'unconstrained'],Selected={self.constraint_P}")

    def forward(self, y_hat: torch.Tensor):
        B, _, S = y_hat.shape 
        P = self.get_P()
        y_hat, _ = torch.sort(y_hat, dim=-1)
        
        parents = self.padded_parents
        num_active_low = parents.size(0)

        idx = parents.view(1, num_active_low, self.max_parents, 1).expand(B, -1, -1, S)
        local_y = torch.gather(
            y_hat.unsqueeze(1).expand(-1, num_active_low, -1, -1), 2, idx
        )

        weighted_y = P.view(1, num_active_low, self.max_parents, 1) * local_y
        PY = torch.sort(torch.sum(weighted_y, dim=2), dim=-1).values 
        
        if self.aggregation_method == "quantile": 
            res = torch.einsum("nl,slk->snk", self.S, PY) 
        elif self.aggregation_method == "permutation_copula": 
            low_node = y_hat[:, -self.num_low:, :]
            shift_amount = PY.mean(dim=-1, keepdim=True) - low_node.mean(dim=-1, keepdim=True)
            reordered_bottom_forecast = self.apply_rank_shuffle(low_node + shift_amount)
            
            res = torch.einsum("nl,slk->snk", self.S, reordered_bottom_forecast)

        return res.sort(dim=-1).values