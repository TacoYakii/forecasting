import torch
import torch.nn as nn
from src.core.loss import LossBase
from src.utils.loss.registry import LossRegistry


@LossRegistry.register("RandomCRPSLoss")
class RandomCRPSLoss(LossBase):
    def __init__(self, reduction="mean"):
        """
        Non-Parametric CRPS loss function
        Args:
            reduction (str): 'mean', 'sum', 'none'
        """
        super(RandomCRPSLoss, self).__init__(reduction)

    def forward(self, y_pred, y_true):
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(-1)
        y_pred = y_pred.float()
        y_true = y_true.float()
        M = y_pred.size(-1)

        term1 = torch.mean(torch.abs(y_pred - y_true), dim=-1)
        y_pred_sorted, _ = torch.sort(y_pred, dim=-1)

        indices = torch.arange(1, M + 1, device=y_pred.device, dtype=y_pred.dtype)
        weights = 2 * indices - M - 1
        term2 = torch.mean(y_pred_sorted * weights, dim=-1) / M
        crps_loss = term1 - term2

        if self.reduction == "mean":
            return torch.mean(crps_loss)
        elif self.reduction == "sum":
            return torch.sum(crps_loss)
        else:
            return crps_loss


@LossRegistry.register("PinballLoss")
class PinballLoss(LossBase):
    """
    Computes the Pinball Loss (Quantile Loss) for given quantiles.
    Returns tensor of shape (n, num_quantiles) if reduction='none'.
    """

    def __init__(self, q_vals, reduction="mean"):
        super(PinballLoss, self).__init__(reduction)
        self.q_vals = q_vals

    def forward(self, y_pred, y_true):
        if y_true.dim() == y_pred.dim() - 1:
            y_true = y_true.unsqueeze(-1)

        y_pred = y_pred.float()
        y_true = y_true.float()

        if not isinstance(self.q_vals, torch.Tensor):
            q_tensor = torch.tensor(
                self.q_vals, dtype=y_pred.dtype, device=y_pred.device
            )
        else:
            q_tensor = self.q_vals.to(dtype=y_pred.dtype, device=y_pred.device)

        error = y_true - y_pred

        pinball = torch.max(q_tensor * error, (q_tensor - 1.0) * error)

        if self.reduction == "mean":
            return torch.mean(pinball)
        elif self.reduction == "sum":
            return torch.sum(pinball)
        else:
            return pinball


@LossRegistry.register("QuantileCRPSLoss")
class QuantileCRPSLoss(LossBase):
    """
    CRPS Loss calculation wrapping Pinball Loss.
    """

    def __init__(self, q_vals, reduction="mean"):
        """
        Args:
            q_vals (list or 1D tensor): Quantiles to predict (e.g., torch.linspace(0.01, 0.99, 99))
            reduction (str): 'mean', 'sum', 'none'
        """
        super(QuantileCRPSLoss, self).__init__(reduction)
        self.pinball_fn = PinballLoss(q_vals, reduction="none")

    def forward(self, y_pred, y_true):
        """x
        y_pred: (n, num_quantiles)
        y_true: (n,) or (n, 1)
        """
        y_pred_sorted, _ = torch.sort(y_pred-y_true) 
        raw_pinball_loss = self.pinball_fn(y_pred, y_true)

        crps_per_sample = 2.0 * torch.mean(raw_pinball_loss, dim=-1)

        if self.reduction == "mean":
            return torch.mean(crps_per_sample)
        elif self.reduction == "sum":
            return torch.sum(crps_per_sample)
        elif self.reduction == "none":
            return crps_per_sample
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")
