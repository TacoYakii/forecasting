import torch
from src.core.reconciliation_model import BaseReconciliationModel


class BottomUp(BaseReconciliationModel):
    r"""
    Bottom-Up Reconciliation Model (Deterministic)

    This classic approach ensures completely coherent hierarchical summation
    by utterly ignoring top-level or aggregate base forecasts. It inherently constructs
    the hierarchy by merely identifying and aggregating the mathematically purest
    bottom-level components.

    The reconciled forecasts are given by:
        \tilde{y} = S P \hat{y}

    where P is the structurally masked bottom-level projection matrix:
        P = [0_{m \times (n-m)} | I_m]
    """

    def __init__(self, S, device="cpu", config=None):
        super().__init__(S, device=device)

        # Define structural derivations explicitly inheriting the execution context device natively
        zero_matrix = torch.zeros(
            (self.num_low, self.num_node - self.num_low), device=self.S.device
        )
        identity_bottom = torch.eye(self.num_low, device=self.S.device)
        P = torch.hstack([zero_matrix, identity_bottom])

        self.register_buffer("SP", self.S @ P)

    def forward(self, y_hat):
        """
        Applies deterministic structural hierarchy projections strictly executing aggregation.
        Expected input: y_hat (Batch, num_nodes)
        """
        return torch.einsum("ij, bj -> bi", self.SP, y_hat)


class BottomUpQuantile(BottomUp):
    r"""
    Bottom-Up Reconciliation for Quantile Forecasts.

    Extends the fundamental deterministic bottom-up aggregation to probabilistic
    realms natively. Raw prediction inputs are strictly organized via `.sort()`
    dimensionally extracting isolated quantiles independently prior to standard mapping constraints.

    Mathematics:
    1. Base forecasts are sorted natively aligning independently:
        \hat{y}^{(q)} = \text{sort}(\hat{y}_{sim})
    2. Summation hierarchy mapping relies entirely independently scaled quantile arrays natively:
        \tilde{y}^{(q)} = S P \hat{y}^{(q)}
    """

    def __init__(self, S, device="cpu", config=None):
        super().__init__(S, device=device, config=config)

    def forward(self, y_hat):
        """
        Target sorts raw outputs enforcing homologous probability scaling dependencies constraints natively.
        Expected input: y_hat (Batch, num_nodes, num_samples)
        """
        y_hat, _ = torch.sort(y_hat, dim=-1)
        return torch.einsum("ij, bjs -> bis", self.SP, y_hat)
