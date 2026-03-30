"""
Conditional Kernel Density (CKD) estimation model.

Implements CKD as a plain Python class with torch tensor operations:
- Config: CKDConfig(BaseConfig)
- Model: ConditionalKernelDensity with fit()/predict()

Bandwidth and decay are plain config values — no nn.Parameter or autograd.
Hyperparameter optimization is handled externally (Optuna via CKDOptunaTrainer).
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from .ckd_types import CKDConfig, PowerDensity, resolve_to_samples


class ConditionalKernelDensity:
    """
    Conditional Kernel Density Estimation model for wind power forecasting.

    Builds a conditional density estimate using Gaussian kernels with
    exponential time-decay weighting. All computations are torch-based
    for vectorization and GPU acceleration.

    Hyperparameters (bandwidth, decay) are read from CKDConfig.
    Use CKDOptunaTrainer for automated hyperparameter search.

    Workflow:
        1. model = ConditionalKernelDensity(config)
        2. model.fit(x_train, y_train, x_columns)  # builds density
        3. power_density = model.predict(samples)    # applies to new data
        4. crps = loss_fn(power_density.to_samples(), y_true)

    Args:
        config: CKDConfig dataclass with hyperparameters.
        device: Torch device string.
    """

    def __init__(self, config: CKDConfig, device: str = "cpu"):
        self.config: CKDConfig = config
        self.device: str = device
        self.is_fitted: bool = False

        self.x_columns: List[str] = []
        self.n_x: int = config.n_x_vars

        # Hyperparameters — plain values from config
        self.x_bandwidths: List[float] = config.get_x_bandwidths()
        self.y_bandwidth: float = config.y_bandwidth
        self.time_decay_factor: float = config.time_decay_factor

        # Populated by fit()
        self.density: Optional[torch.Tensor] = None
        self.y_basis: Optional[torch.Tensor] = None
        self.x_basis_list: List[torch.Tensor] = []

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_columns: List[str],
    ) -> "ConditionalKernelDensity":
        """
        Build the CKD from training data.

        Args:
            x: Training explanatory variables, shape (T, n_x_vars).
            y: Training response variable, shape (T,).
            x_columns: Names of the explanatory variables.

        Returns:
            self (for method chaining).
        """
        self.x_columns = x_columns
        self.n_x = len(x_columns)

        if self.n_x != len(self.x_bandwidths):
            raise ValueError(
                f"Config n_x_vars ({len(self.x_bandwidths)}) "
                f"doesn't match actual x_columns ({self.n_x}). "
                f"Set CKDConfig(n_x_vars={self.n_x}, ...) at model creation."
            )

        dtype = torch.float64
        x_vals = x.to(device=self.device, dtype=dtype)
        y_vals = y.to(device=self.device, dtype=dtype).squeeze()
        T = x_vals.shape[0]

        if T != y_vals.shape[0]:
            raise ValueError(
                f"Length mismatch: x has {T} rows but y has {y_vals.shape[0]}."
            )

        # Build basis grids
        self.x_basis_list = []
        for i in range(self.n_x):
            basis = torch.linspace(
                x_vals[:, i].min().item(),
                x_vals[:, i].max().item(),
                self.config.x_basis_points,
                dtype=dtype, device=self.device,
            )
            self.x_basis_list.append(basis)

        self.y_basis = torch.linspace(
            y_vals.min().item(), y_vals.max().item(),
            self.config.y_basis_points,
            dtype=dtype, device=self.device,
        )

        # Build density
        self.density = self._build_density(x_vals, y_vals)
        self.is_fitted = True
        return self

    def _build_density(
        self,
        x_vals: torch.Tensor,
        y_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized CKD construction using torch.einsum.

        Computes:
            density[b1,b2,...,y] = Σ_t w_t · K_x(x_t, b) · K_y(y_t, y) / Σ_t w_t · K_x(x_t, b)
        where K is a Gaussian kernel and w_t = decay^(T-1-t).

        Returns:
            density tensor of shape (B1, B2, ..., Y_size)
        """
        T = x_vals.shape[0]
        device = x_vals.device
        dtype = x_vals.dtype

        # Time decay weights: (T,)
        decay = torch.tensor(self.time_decay_factor, dtype=dtype, device=device)
        time_weights = decay ** torch.arange(
            T - 1, -1, -1, dtype=dtype, device=device
        )

        # X kernel PDFs: each (T, B_i) with per-variable bandwidth
        x_pdfs = []
        for i, basis in enumerate(self.x_basis_list):
            diff = x_vals[:, i:i+1] - basis
            bw = torch.tensor(self.x_bandwidths[i], dtype=dtype, device=device)
            dist = torch.distributions.Normal(
                torch.zeros(1, dtype=dtype, device=device), bw
            )
            x_pdfs.append(dist.log_prob(diff).exp())

        # Y kernel PDF: (T, Y_size)
        y_bw = torch.tensor(self.y_bandwidth, dtype=dtype, device=device)
        y_dist = torch.distributions.Normal(
            torch.zeros(1, dtype=dtype, device=device), y_bw
        )
        y_pdf = y_dist.log_prob(y_vals[:, None] - self.y_basis).exp()

        # Joint X density via broadcasting: (T, B1, B2, ...)
        joint_x = x_pdfs[0]
        for pdf in x_pdfs[1:]:
            joint_x = joint_x.unsqueeze(-1) * pdf.unsqueeze(-2)

        # Apply time weights
        n_basis_dims = joint_x.dim() - 1
        w_shape = (T,) + (1,) * n_basis_dims
        weighted_x = time_weights.view(*w_shape) * joint_x

        # Einsum for conditional density
        basis_letters = "".join(chr(ord("a") + i) for i in range(n_basis_dims))
        ein_str = f"t{basis_letters},ty->{basis_letters}y"
        numerator = torch.einsum(ein_str, weighted_x, y_pdf)
        denominator = weighted_x.sum(dim=0)

        density = numerator / denominator.unsqueeze(-1).clamp(min=1e-30)
        density = density / density.sum(dim=-1, keepdim=True).clamp(min=1e-30)

        return density

    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------

    def _resolve_inputs(
        self,
        inputs,
        time_index: Optional[np.ndarray],
        horizon: Optional[int],
        n_samples: Optional[int],
    ) -> Tuple[List[torch.Tensor], Optional[np.ndarray]]:
        """
        Normalize flexible inputs to (List[torch.Tensor], time_index).

        Handles:
            - List[torch.Tensor]: backward compatible, returned as-is
            - Single torch.Tensor: wrapped in list (for n_x_vars=1)
            - Single Result object: resolved and wrapped in list
            - List of Result objects: each resolved independently
        """
        _n_samples = n_samples if n_samples is not None else self.config.n_samples

        # Already a list of tensors → backward compatible path
        if (
            isinstance(inputs, list)
            and len(inputs) > 0
            and isinstance(inputs[0], torch.Tensor)
        ):
            return inputs, time_index

        # Single tensor → wrap
        if isinstance(inputs, torch.Tensor):
            return [inputs], time_index

        # Single non-list object → wrap in list
        if not isinstance(inputs, list):
            inputs = [inputs]

        # List of Result objects → resolve each
        samples_list: List[torch.Tensor] = []
        resolved_time_index = time_index
        for obj in inputs:
            s, ti = resolve_to_samples(
                obj,
                n_samples=_n_samples,
                horizon=horizon,
                device=self.device,
            )
            samples_list.append(s)
            if resolved_time_index is None and ti is not None:
                resolved_time_index = ti

        return samples_list, resolved_time_index

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        inputs,
        time_index: Optional[np.ndarray] = None,
        *,
        horizon: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> PowerDensity:
        """
        Apply the fitted CKD to forecast inputs.

        Accepts flexible input formats — raw tensors or any forecast result
        object from ``src.core.forecast_distribution``.

        Args:
            inputs: Forecast data in any supported format.
            time_index: Optional time labels.
            horizon: Forecast horizon (1-indexed). Required for multi-horizon types.
            n_samples: Samples to draw from distribution-based types.

        Returns:
            PowerDensity with (T, Y_size) density values.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not fitted. Call .fit(x, y, x_columns) first."
            )

        samples, resolved_time_index = self._resolve_inputs(
            inputs, time_index, horizon, n_samples,
        )

        if len(samples) != self.n_x:
            raise ValueError(
                f"Expected {self.n_x} input variables, got {len(samples)}."
            )

        dtype = self.density.dtype

        # Map samples to grid indices
        indices = []
        for i, s in enumerate(samples):
            basis = self.x_basis_list[i]
            s = s.to(device=self.device, dtype=dtype)
            idx = torch.bucketize(s, basis)
            idx = idx.clamp(0, basis.shape[0] - 1)
            indices.append(idx)

        # Fancy indexing: density[idx_0, idx_1, ...] → (T, n_samples, Y_size)
        looked_up = self.density[tuple(indices)]

        # Average over simulations → (T, Y_size)
        power_density = looked_up.mean(dim=1)

        # Re-normalize
        power_density = power_density / power_density.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-30)

        return PowerDensity(
            values=power_density,
            y_basis=self.y_basis,
            time_index=resolved_time_index,
        )

    def get_hyperparameters(self) -> dict:
        """Return current hyperparameter values as a dict."""
        return {
            "x_bandwidth": {
                col: bw for col, bw in zip(self.x_columns, self.x_bandwidths)
            } if self.x_columns else self.x_bandwidths,
            "y_bandwidth": self.y_bandwidth,
            "time_decay_factor": self.time_decay_factor,
        }

    def point_estimate_from_samples(
        self,
        inputs,
        time_index: Optional[np.ndarray] = None,
        *,
        horizon: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convenience: apply CKD → point estimation → DataFrame.

        Accepts the same flexible input types as ``predict()``.
        """
        power_density = self.predict(
            inputs, time_index, horizon=horizon, n_samples=n_samples,
        )
        point = power_density.point_estimate().detach().cpu().numpy()
        res = pd.DataFrame(point, columns=["mu"])
        if power_density.time_index is not None:
            res.index = pd.to_datetime(power_density.time_index)
        res.index.name = "forecast_time"
        return res
