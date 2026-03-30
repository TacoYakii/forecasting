import torch
from src.core.reconciliation_model import BaseReconciliationModel


class TopDown(BaseReconciliationModel):
    r"""
    Top-Down Reconciliation Model (Deterministic)

    This approach reconciles hierarchical forecasts by distributing the top-level
    forecast down to the bottom-level series using historically derived proportions
    or simple equal weights, ensuring perfectly constrained summations.

    The reconciled forecasts are given by:
        \tilde{y} = S P \hat{y}

    where P is the projection matrix (assuming equal distributions from the top node):
        P = [q | 0_{m \times (n-1)}]
    and q = 1_m.
    """

    def __init__(self, S, device="cpu", config=None):
        super().__init__(S, device=device)

        # Define structural derivations mapping identically to device
        zero_matrix = torch.zeros(
            (self.num_low, self.num_node - 1), device=self.S.device
        )
        q = torch.ones((self.num_low, 1), device=self.S.device)
        P = torch.hstack([q, zero_matrix])

        self.register_buffer("SP", self.S @ P)

    def forward(self, y_hat):
        """
        Applies deterministic projection mathematically mapping flat 2D representations.
        Returns coherent outputs scaling dynamically.
        Expected input: y_hat (Batch, num_nodes)
        """
        # (SP: num_nodes x num_nodes) @ (y_hat: Batch x num_nodes)
        # Using einsum for safe batch projection alignments
        return torch.einsum("ij, bj -> bi", self.SP, y_hat)


class TopDownQuantile(TopDown):
    r"""
    Top-Down Reconciliation for Quantile Forecasts.

    This approach extends the base deterministic TopDown methodology prioritizing
    probabilistic evaluations. It sorts the raw base forecast outputs independently
    into aligned quantiles before executing structural Top-Down matrix projections.

    Mathematics:
    1. Base forecasts are sorted to align corresponding quantiles independently:
        \hat{y}^{(q)} = \text{sort}(\hat{y}_{sim})
    2. The nominal structural mapping matrix applies independently to all samples:
        \tilde{y}^{(q)} = S P \hat{y}^{(q)}
    """

    def __init__(self, S, device="cpu", config=None):
        super().__init__(S, device=device, config=config)

    def forward(self, y_hat):
        """
        Sorts the raw outputs targeting homogenous quantiles discretely assigning alignments.
        Expected input: (Batch, num_nodes, num_samples)
        """
        y_hat, _ = torch.sort(y_hat, dim=-1)
        return torch.einsum("ij, bjs -> bis", self.SP, y_hat)
