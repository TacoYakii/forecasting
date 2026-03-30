import torch.nn as nn 

class LossBase(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction: str = reduction
    
    def forward(self, y_pred, y_true):
        raise NotImplementedError