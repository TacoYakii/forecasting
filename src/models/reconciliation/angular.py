import torch
import torch.nn as nn
import torch.nn.functional as F
from src.core.reconciliation_model import BaseReconciliationModel
from dataclasses import dataclass, field
from src.core.config import BaseConfig


@dataclass
class AngularReconciliationConfig(BaseConfig):
    mode: str = "angular"
    angle_activation: str = "sigmoid"
    init_P: str = "ols"
    constraint_P: str = "linear"
    tau: float = 0.1

    q_start: float = field(default=0.001)
    q_end: float = field(default=0.999)
    n_samples: int = field(default=1000)


class AngularReconciliation(BaseReconciliationModel):
    padded_parents: torch.Tensor
    parent_mask: torch.Tensor
    q_val: torch.Tensor
    config: AngularReconciliationConfig

    def __init__(self, S, device, config: AngularReconciliationConfig):
        super().__init__(S, device=device)
        self.config = config  # ty:ignore[unresolved-attribute]
        self.mode: str = config.mode

        self.constraint_P: str = config.constraint_P
        self.tau: float = config.tau
        self.angle_activation: str = config.angle_activation

        P_valid_init = self._P_matrix_init(config.init_P)
        self.P_valid = nn.Parameter(P_valid_init)

        init_val = self._initialize_angle()
        self.angle_raw = nn.Parameter(init_val)

        q_start: float = config.q_start
        q_end: float = config.q_end
        q_num: int = config.n_samples
        q_val = torch.linspace(q_start, q_end, q_num, device=self.S.device)

        # self.register_buffer('padded_parents', self.padded_parents) # index map -> point array for P
        # self.register_buffer('parent_mask', self.parent_mask) # array for showing value of padded_parents[i] is valid
        self.register_buffer("q_val", q_val)

    def _initialize_angle(self):
        """
        Initializes the raw angle parameter based on the reconciliation mode and activation function.
        """
        if self.mode == "angular":
            if self.angle_activation == "relu":
                init_val = (
                    torch.full((self.num_low,), 45.0) + torch.randn(self.num_low) * 2.0
                )
            elif self.angle_activation == "sigmoid":
                init_val = torch.randn(self.num_low) * 0.1
            elif self.angle_activation == "softplus":
                init_val = torch.full((self.num_low,), 45.0) + torch.randn(self.num_low) * 2.0
            else:
                raise ValueError(
                    f"Angle activation must be 'relu', 'sigmoid', or 'softplus' when mode='angular'. Got: {self.angle_activation}"
                )
        elif self.mode == "horizontal":
            init_val = torch.zeros(self.num_low)
        elif self.mode == "vertical":
            init_val = torch.full((self.num_low,), 90.0)
        else:
            raise ValueError(
                f"Reconciliation mode error. Available=['angular', 'horizontal', 'vertical'], Selected={self.mode}"
            )

        return init_val.to(self.S.device)

    def get_P(self):
        # return valid P based on constraint setting
        if self.constraint_P == "linear":  # sum to one & positive
            inf_mask = self.P_valid.masked_fill(self.parent_mask == 0, float("-inf"))
            return F.softmax(inf_mask, dim=1)

        elif self.constraint_P == "sum": # Only sum to one
            p_val = (
                F.softplus(self.P_valid)
                if self.mode == "angular" or self.mode == "vertical" # if angle appears -> every elements should be positive 
                else self.P_valid
            )
            p_valid_masked = p_val * self.parent_mask
            return p_valid_masked / (p_valid_masked.sum(dim=1, keepdim=True) + 1e-9)

        elif self.constraint_P == "unconstrained":
            if self.mode == "angular" or self.mode == "vertical":
                return F.softplus(self.P_valid) * self.parent_mask # every elements should be positive 
            return self.P_valid * self.parent_mask

        else:
            raise ValueError(
                f"P constraint error. Available=['linear', 'sum', 'unconstrained'],Selected={self.constraint_P}"
            )

    def get_angle_vector(self):
        if self.mode == "angular":
            # Prevent angle from becoming exactly 0 or 90 to avoid numerical instability
            # tan(0) = 0 -> division by zero. tan(90) -> inf.
            min_angle, max_angle = 1.0, 90.0
            
            if self.angle_activation == "relu":
                # Note: 'relu' + clamp can lead to dead gradients if angle falls out of bounds.
                return torch.clamp(self.angle_raw, min=min_angle, max=max_angle)
            elif self.angle_activation == "sigmoid":
                return torch.sigmoid(self.angle_raw) * (max_angle - min_angle) + min_angle
            elif self.angle_activation == "softplus":
                return torch.clamp(F.softplus(self.angle_raw), min=min_angle, max=max_angle)
            else:
                return self.angle_raw
        else:
            return self.angle_raw

    def horizontal_combine(self, y_hat, P, target_idx=None):
        """
        Horizontal combining, ensures monotonicity

        :param y_hat: Forecast, Normalized
        :param P: Combining weights
        :param target_idx: Specify a particular low-level node for which the function should be performed
        """
        B, _, S = y_hat.shape
        idx_s = (
            slice(target_idx, target_idx + 1) if target_idx is not None else slice(None)
        )

        local_P = P[idx_s]
        parents = self.padded_parents[idx_s]

        num_active_low = parents.size(0)

        idx = parents.view(1, num_active_low, self.max_parents, 1).expand(B, -1, -1, S)
        local_y = torch.gather(
            y_hat.unsqueeze(1).expand(-1, num_active_low, -1, -1), 2, idx
        )

        weighted_y = local_P.view(1, num_active_low, self.max_parents, 1) * local_y
        res = torch.sum(weighted_y, dim=2)

        return torch.sort(res, dim=-1).values

    def angular_combine(self, y_hat, P, angle_vector):
        r"""
        Performs angular combining of hierarchical forecasts, ensuring appropriate quantile sorting
        and structural constraints based on the provided angle parameters.

        This process involves selecting a parent's forecast using Gumbel-Softmax sampling, normalizing
        the forecast range, applying an angular transformation to the quantiles, and then inverse
        normalizing to produce the final reconciled forecast for the lower level nodes.

        :param y_hat: Base forecasts for all nodes. Shape: (Batch, Nodes, Quantiles)
        :param P: Combining weights/probabilities for parents. Shape: (Nodes, Max_Parents)
        :param angle_vector: Vector of angles (in degrees) used for the quantile transformation. Shape: (Num_Low_Nodes,)
        :return: Reconciled, sorted forecasts for the lower-level nodes.

        ### Mathematical Equations & Logic

        The angular combining logic follows a four-step process:

        #### 1. Parent Selection (Gumbel-Softmax Sampling)
        We select a representative parent forecast for each low-level node by sampling from the
        probability distribution \( P \) corresponding to its valid parents. We use the Gumbel-Softmax
        trick for differentiable sampling.

        $$ W \sim \text{Gumbel-Softmax}(\log(P), \tau) $$
        $$ \bar{y} = \sum_{k} W_k \cdot y^{(parent_k)} $$

        - \( P \): Given probabilities for each parent (`P`).
        - \( W \): Sampled one-hot weight vector (`weight_angular`).
        - \( \tau \): Temperature hyperparameter (`self.tau`).
        - \( y^{(parent_k)} \): Parent forecast predictions (`local_y`).
        - \( \bar{y} \): The raw selected parent forecast (`raw_selected`).

        #### 2. Min-Max Normalization
        To apply the angular transformation consistently, the selected forecast is normalized into a
        [0, 1] range using the minimum and maximum forecasts among all valid parents.

        $$ y_{min} = \min_{k} y^{(parent_k)} $$
        $$ y_{max} = \max_{k} y^{(parent_k)} $$
        $$ \tilde{y} = \frac{\bar{y} - y_{min}}{y_{max} - y_{min}} $$

        - \( y_{min}, y_{max} \): Min and max bounds of parent forecasts (`y_hat_min`, `y_hat_max`).
        - \( \tilde{y} \): Normalized parent forecast (`selected_scaled`).
        - \( y_{max} - y_{min} \): Denominator for scaling (`denominator`).

        #### 3. Quantile Angular Transformation (`quantile_transform`)
        The normalized forecast quantiles are shifted based on the `angle_vector` and base quantiles `q_val`,
        then sorted to maintain monotonicity.

        $$ \theta = \tan(\text{angle}) $$
        $$ \tilde{q} = \frac{q_{val}}{\theta} $$
        $$ S = \text{sort}(\tilde{y} + \tilde{q}) $$
        $$ y_{\text{angular}} = S - \tilde{q} $$

        - \( \text{angle} \): Angle parameters in degrees (`angle_vector`).
        - \( \theta \): Tangent of the angle (`theta`).
        - \( q_{val} \): Base quantile values (`self.q_val`).
        - \( \tilde{q} \): Transformed quantiles (`q_val_transformed`).
        - \( S \): Shifted and sorted intermediate values (`s_sorted`).
        - \( y_{\text{angular}} \): The properly transformed and sorted normalized forecast (`angular_scaled` or `angular_raw`).

        #### 4. Inverse Normalization
        The transformed values are scaled back to their original numerical domain using the initial bounds.

        $$ \hat{y}_{low} = y_{\text{angular}} \cdot (y_{max} - y_{min}) + y_{min} $$

        - \( \hat{y}_{low} \): The final reconciled forecast for the low-level node (`angular`).
        """

        def quantile_transform(selected_scaled, angle):
            theta = torch.tan(torch.deg2rad(angle))
            theta = torch.clamp(theta, min=1e-5) # Prevent division by zero
            q_val_transformed = self.q_val[None, :] / theta[:, None]
            s_val = selected_scaled + q_val_transformed.unsqueeze(
                0
            )  # (Batch, num_low, 1000)
            s_sorted = torch.sort(s_val, dim=-1).values
            angular_scaled = s_sorted - q_val_transformed.unsqueeze(0)

            return angular_scaled

        B, _, S = y_hat.shape
        num_active_low = self.num_low
        parents = self.padded_parents
        mask = self.parent_mask.view(self.num_low, self.max_parents, 1)

        idx = parents.view(1, num_active_low, self.max_parents, 1).expand(B, -1, -1, S)
        local_y = torch.gather(
            y_hat.unsqueeze(1).expand(-1, num_active_low, -1, -1), 2, idx
        )

        y_hat_max = local_y.masked_fill(mask.unsqueeze(0) == 0, -1e18).amax(
            dim=[2, 3], keepdim=True
        )
        y_hat_min = local_y.masked_fill(mask.unsqueeze(0) == 0, 1e18).amin(
            dim=[2, 3], keepdim=True
        )
        denominator = (y_hat_max - y_hat_min).clamp(min=1e-8)

        local_logits = torch.log(P + 1e-9).masked_fill(self.parent_mask == 0, -1e9)
        expanded_logits = local_logits.view(
            1, num_active_low, 1, self.max_parents
        ).expand(B, -1, S, -1)

        weight_angular = F.gumbel_softmax(expanded_logits, tau=self.tau, hard=True)
        raw_selected = torch.sum(local_y.permute(0, 1, 3, 2) * weight_angular, dim=-1)
        selected_scaled = (raw_selected - y_hat_min.squeeze(-1)) / denominator.squeeze(
            -1
        )

        angular_raw = quantile_transform(selected_scaled, angle_vector)

        angular = angular_raw * denominator.squeeze(-1) + y_hat_min.squeeze(-1)
        return torch.sort(angular, dim=-1).values

    def forward(self, y_hat, target_idx=None, fixed_angle=None):
        P = self.get_P()
        y_hat = torch.sort(y_hat, dim=-1).values
        angle_vector = self.get_angle_vector()

        # Base configuration
        is_horizontal = (angle_vector < 1.0).float().view(1, -1, 1)

        # Overrides
        if fixed_angle is not None:
            is_h_val = torch.tensor(float(fixed_angle) < 1.0, device=self.S.device).float()

            if target_idx is not None:
                angle_vector = angle_vector.clone()
                is_horizontal = is_horizontal.clone()
                angle_vector[target_idx] = float(fixed_angle)
                is_horizontal[:, target_idx, :] = is_h_val
            else:
                angle_vector = torch.full_like(angle_vector, float(fixed_angle))
                is_horizontal = torch.full_like(is_horizontal, is_h_val.item())

        # Combine
        if is_horizontal.all():
            PY = self.horizontal_combine(y_hat, P)
        elif (1 - is_horizontal).all():
            PY = self.angular_combine(y_hat, P, angle_vector=angle_vector)
        else:
            h_PY = self.horizontal_combine(y_hat, P)
            a_PY = self.angular_combine(y_hat, P, angle_vector=angle_vector)
            PY = (is_horizontal * h_PY) + ((1 - is_horizontal) * a_PY)

        return torch.einsum("nl,slk->snk", self.S, PY)
