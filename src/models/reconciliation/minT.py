import torch
from src.core.reconciliation_model import BaseReconciliationModel
from src.core.config import BaseConfig
from dataclasses import dataclass
from .copula import PermutationEmpiricalCopula

@dataclass
class MinTConfig(BaseConfig):
    mode: str = "OLS"



class MinT(BaseReconciliationModel):
    r"""
    Minimum Trace (MinT) Reconciliation Model

    This module implements the Minimum Trace (MinT) reconciliation approach
    proposed by Wickramasuriya et al. (2019) for hierarchical forecasts.
    The MinT estimator minimizes the trace of the covariance matrix of the
    reconciled forecast errors.

    The reconciled forecasts are given by:
        \tilde{y} = S P \hat{y}

    where P is the projection matrix:
        P = J - J W U (U' W U)^{-1} U'

    Definitions:
    - J = [0_{m \times (n-m)} | I_{m \times m}]
    - U = [I_{(n-m) \times (n-m)} | -C]^T
    - W is the covariance matrix of base forecast errors.
    """

    is_fitted: bool
    mode: str

    def __init__(self, S, device, config: MinTConfig):
        super().__init__(S, device=device)
        self.mode: str = config.mode
        object.__setattr__(self, "is_fitted", False)

        self.num_top: int = self.num_node - self.num_low
        C = (self.S.T[:, : self.num_top]).T
        self.J = torch.concat(
            (torch.zeros((self.num_low, self.num_top), device=self.S.device), torch.eye(self.num_low, device=self.S.device)), dim=1
        )
        self.U_transposed = torch.concat((torch.eye(self.num_top, device=self.S.device), -C), dim=1)
        self.U = self.U_transposed.T

    def fit(self, train_y_hat, train_y_obs):
        """
        Dynamically calculates and freezes the projection matrix P based on training error covariances.
        Must be called prior to evaluation.
        """
        residuals = self.get_residual(train_y_hat, train_y_obs)
        P_fitted = self.compute_P(residuals)
        self.register_buffer("P", P_fitted)
        object.__setattr__(self, "is_fitted", True)

    def compute_P(self, residuals):
        r"""
        Computes the reconciliation projection matrix P based on the specified mode.
        Follows notation from Wickramasuriya et al (2019).

        mode == "OLS":
            W_h = k_h I for all h, k_h > 0.
            MinT estimator collapses to the OLS estimator of Hyndman et al.
            Assumes base forecast errors are uncorrelated and equivariant
            (often impossible in practical hierarchical series).

        mode == "WLS": Variance scaling WLS.
            W_h = k_h diag(\hat{W}_1).
            \hat{W}_1 is the unbiased sample covariance estimator of the in-sample errors.

        mode == "MINT_SAMPLE":
            MinT with unrestricted sample covariance estimator for h = 1.
            Relatively simple to obtain, but may not be a valid estimate when m > T.

        mode == "MINT_SHRINK":
            Shrinkage estimation of sample covariance estimator for h = 1.

        Dynamically handles 2D representations (m, T) and 3D representations (T, m, S)
        to natively support probabilistic Quantile reconciliation pipelines.
        """
        mode = self.mode.upper()
        is_3d = residuals.dim() == 3

        if mode == "OLS":
            if is_3d:
                num_samples = residuals.shape[2]
                W = (
                    torch.eye(self.num_node, device=self.S.device)
                    .unsqueeze(-1)
                    .expand(-1, -1, num_samples)
                )
            else:
                W = torch.eye(self.num_node, device=self.S.device)
        elif mode == "WLS":
            W_sample = self.get_sample_covariance(residuals)
            if is_3d:
                num_samples = residuals.shape[2]
                diag_W = torch.diagonal(W_sample, dim1=0, dim2=1)  # (S, m)
                W = torch.zeros(
                    (self.num_node, self.num_node, num_samples), device=self.S.device
                )
                indices = torch.arange(self.num_node)
                W[indices, indices, :] = diag_W.T
            else:
                W = torch.diag(torch.diagonal(W_sample)).to(self.S.device)
        elif mode == "MINT_SAMPLE":
            W = self.get_sample_covariance(residuals).to(self.S.device)
        elif mode == "MINT_SHRINK":
            W = self.get_shrinkage_estimator(residuals).to(self.S.device)
        else:
            raise ValueError(
                f"Mode error. Available=['OLS', 'WLS', 'MINT_SAMPLE', 'MINT_SHRINK'], Selected={mode}"
            )

        if is_3d:
            W_p = W.permute(2, 0, 1)  # (S, m, m)
            middle_inv = torch.inverse(self.U_transposed @ W_p @ self.U)
            P_batch = self.J - self.J @ W_p @ self.U @ middle_inv @ self.U_transposed
            return P_batch.permute(1, 2, 0)
        else:
            P = (
                self.J
                - self.J
                @ W
                @ self.U
                @ torch.linalg.pinv(self.U_transposed @ W @ self.U)
                @ self.U_transposed
            )
            return P

    def get_sample_covariance(self, residuals):
        """
        Calculates the sample covariance matrix of the base forecast residuals.
        Dynamically supports 2D configurations & 3D batched simulations.
        """
        if residuals.dim() == 3:
            T, m, S = residuals.shape
            means = residuals.mean(dim=0, keepdim=True)
            X = residuals - means
            factor_emp_cov = 1.0 / (T - 1)
            
            chunk_size = 200
            W_chunks = []
            for i in range(0, S, chunk_size):
                X_chunk = X[:, :, i:i+chunk_size]
                W_chunk = torch.einsum('tms,tns->mns', X_chunk, X_chunk) * factor_emp_cov
                W_chunks.append(W_chunk)
            return torch.cat(W_chunks, dim=2)
        else:
            n_samples = residuals.shape[1]
            means = residuals.mean(dim=1, keepdim=True)
            X = residuals - means
            factor_emp_cov = 1.0 / (n_samples - 1)
            W = torch.mm(X, X.T) * factor_emp_cov
            return W

    def get_shrinkage_estimator(self, residuals):
        r"""
        Schäfer & Strimmer (2005) shrinkage estimator components using PyTorch.
        Returns the shrunk covariance matrix ensuring positive semi-definiteness.

        Formula:
            W_{shrunk} = \lambda^* W_{diag} + (1 - \lambda^*) W_{sample}

        Args:
            residuals: (m, T) input tensor containing base forecast residuals.
        """
        epsilon = 2e-8
        W = self.get_sample_covariance(residuals)

        if residuals.dim() == 3:
            T, m, S = residuals.shape
            means = residuals.mean(dim=0, keepdim=True)
            X = residuals - means

            stds = residuals.std(dim=0, unbiased=False, keepdim=True)
            Xs = X / (stds + epsilon)
            
            chunk_size = 200
            W_shrunk_chunks = []
            
            for i in range(0, S, chunk_size):
                Xs_c = Xs[:, :, i:i+chunk_size]
                W_c = W[:, :, i:i+chunk_size]
                S_c = Xs_c.shape[2]
                
                R_biased_c = torch.einsum('tms,tns->mns', Xs_c, Xs_c) / T
                R_sq_c = R_biased_c.pow(2)
                sum_sq_emp_corr_c = R_sq_c.sum(dim=(0, 1)) - torch.diagonal(R_sq_c, dim1=0, dim2=1).sum(dim=1)
                
                Xs_sq_c = Xs_c**2
                sum_w_sq_c = torch.einsum('tms,tns->mns', Xs_sq_c, Xs_sq_c)
                var_matrix_c = sum_w_sq_c - T * R_sq_c
                sum_var_emp_corr_c = var_matrix_c.sum(dim=(0, 1)) - torch.diagonal(var_matrix_c, dim1=0, dim2=1).sum(dim=1)
                
                factor_shrinkage = 1.0 / (T * (T - 1))
                lambda_star_c = (factor_shrinkage * sum_var_emp_corr_c) / (sum_sq_emp_corr_c + epsilon)
                lambda_star_c = torch.clamp(lambda_star_c, min=0.0, max=1.0)
                shrinkage_c = 1.0 - lambda_star_c

                W_shrunk_c = W_c * shrinkage_c.view(1, 1, S_c)
                diag_indices = torch.arange(m)
                W_shrunk_c[diag_indices, diag_indices, :] = W_c[diag_indices, diag_indices, :]
                W_shrunk_chunks.append(W_shrunk_c)

            return torch.cat(W_shrunk_chunks, dim=2)
        else:
            T = residuals.shape[1]
            means = residuals.mean(dim=1, keepdim=True)
            X = residuals - means
            factor_shrinkage = 1.0 / (T * (T - 1))

            stds = residuals.std(dim=1, unbiased=False, keepdim=True)
            Xs = X / (stds + epsilon)

            # Correlation formulation
            Xs_mean = Xs.mean(dim=1, keepdim=True)
            Xs_centered = Xs - Xs_mean
            R_biased = torch.mm(Xs_centered, Xs_centered.T) / T
            R_sq = R_biased.pow(2)
            diag_R_sq = torch.diagonal(R_sq)
            sum_sq_emp_corr = R_sq.sum() - diag_R_sq.sum()

            # Variance formulation
            Xs_sq = Xs_centered ** 2 
            sum_w_sq = torch.mm(Xs_sq, Xs_sq.t())
            var_matrix = sum_w_sq - T * (R_biased ** 2) 
            diag_var = torch.diagonal(var_matrix)
            sum_var_emp_corr = var_matrix.sum() - diag_var.sum()

            # Lambda thresholding
            lambda_star = (factor_shrinkage * sum_var_emp_corr) / (
                sum_sq_emp_corr + epsilon
            )
            lambda_star = torch.clamp(lambda_star, min=0.0, max=1.0)
            shrinkage = 1.0 - lambda_star

            diag_W = torch.diagonal(W).clone()
            W = W * shrinkage
            W.diagonal().copy_(diag_W)
            return W

    def get_residual(self, train_y_hat, train_y_obs):
        """
        Calculates out-of-sample or in-sample residuals:
        y_obs - y_hat (represented inversely or consistently via mapping)
        """
        residuals = (train_y_hat - train_y_obs).squeeze(-1).T
        return residuals

    def forward(self, y_hat):
        """
        Applies discrete MinT mean reconciliation mathematically mapping nodes natively.
        Returns coherent outputs scaling dynamically.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is missing parameterized projections. Run `.fit(train_y_hat, train_y_obs)` first."
            )
        PY = torch.einsum("nm,bml->bnl", self.P, y_hat)
        res = torch.einsum("mn,bnl->bml", self.S, PY)
        return res


class MinTQuantile(MinT):
    r"""
    MinT Reconciliation for Quantile Forecasts.

    This approach reconciles probabilistic forecasts by sorting the selected
    forecasts into quantiles and independently applying MinT reconciliation
    to each quantile level, ensuring independent quantile alignment.

    Mathematics:
    1. Base forecasts are sorted to align corresponding quantiles independently:
        \hat{y}^{(q)} = \text{sort}(\hat{y}_{sim})
    2. MinT projection is applied homogeneously on each quantile:
        \tilde{y}^{(q)} = S P_q \hat{y}^{(q)}

    The covariance matrix W_q is scaled using the batched quantile forecast residuals.
    """
    set_different_copula: bool 

    def __init__(self, S, device, config: MinTConfig):
        super().__init__(S, device, config)
        self.relation = extract_tree_from_S(self.S)
        object.__setattr__(self, "set_different_copula", True)

    def fit(self, train_y_hat, train_y_obs):
        train_y_hat, _ = torch.sort(train_y_hat, dim=-1)
        residuals_quantile = self.get_residual_quantile(train_y_hat, train_y_obs)
        P_fitted = self.compute_P(residuals_quantile)
        self.register_buffer("P", P_fitted)
        object.__setattr__(self, "is_fitted", True)

    def get_residual_quantile(self, train_y_hat, train_y_obs):
        return train_y_hat - train_y_obs

    def forward(self, y_hat):
        """
        Sorts the raw outputs targeting homogenous quantiles discretely applying matrix alignments.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is missing parameterized projections. Run `.fit(train_y_hat_quantile, train_y_obs)` first."
            )

        y_hat, _ = torch.sort(y_hat, dim=-1)
        bottom_node = torch.einsum("nms,bms->bns", self.P, y_hat)
        coherent_res = torch.einsum("mn,bns->bms", self.S, bottom_node)
        return coherent_res


class MinTSchaake(MinT, PermutationEmpiricalCopula):
    r"""
    MinT Reconciliation via Schaake Shuffle (Empirical Copula).

    This approach ensures multivariate dependency modeling across the hierarchy
    by adhering to an empirical copula mapping (Schaake Shuffle) after applying
    a discrete mean-shift MinT reconciliation to base forecasts.

    Procedure Logic:
    1. Base Error Copula Collection:
        Compute the empirical copula (rank dependencies) \mathcal{C} from
        in-sample base forecast representations and actual observations.
    2. Mean-Shift Application:
        Apply nominal MinT projection iteratively across distributions shifting
        the mean of the bottom-level base forecast boundaries:
        \tilde{y}_{bottom} = \hat{y}_{bottom} + (\mu_{reconciled} - \mu_{base})
    3. Copula Restoration:
        Sort the samples of the shifted bottom distributions.
        Reorder (Schaake shuffle) the sorted samples mapped universally to the
        prior rank dependencies extracted via the empirical copula \mathcal{C}.
    4. S-Matrix Aggregation (Bottom-Up):
        Because temporal or cross-sectional matrices may present acyclic graphs,
        the hierarchy is reconstructed organically through direct S-matrix
        multiplications rather than fragile recursive routines:
        \tilde{y} = S \tilde{y}_{bottom\_reordered}
    """
    empirical_distribution: torch.Tensor

    def __init__(self, S, device, config: MinTConfig):
        MinT.__init__(self, S, device, config)
        self.relation = extract_tree_from_S(self.S)
        self.init_empirical_copula(device) 

    def fit(self, train_y_hat, train_y_obs):
        """
        Computes baseline parameter matrix adjustments and builds the empirical copula distribution.
        """
        train_y_hat = torch.sort(train_y_hat, dim=-1).values
        train_y_hat_mean = train_y_hat.mean(dim=-1, keepdim=True)

        MinT.fit(self, train_y_hat_mean, train_y_obs)
        self.fit_empirical_copula(train_y_hat[:, -self.num_low:, :], train_y_obs[:, -self.num_low:, :])
        object.__setattr__(self, "is_fitted", True)

    def get_shifted_bottom_level(self, y_hat):
        """
        Applies mathematical shifts modifying absolute predictions aligned toward MinT bounds.
        """
        mean_y_hat = y_hat.mean(dim=-1, keepdim=True)
        mean_PY = torch.einsum("nm,bml->bnl", self.P, mean_y_hat)

        mean_bottom = mean_y_hat[:, -self.num_low:, :]
        shift_amount = mean_PY - mean_bottom

        reconcilied_bottom_forecast = y_hat[:, -self.num_low:, :] + shift_amount
        return reconcilied_bottom_forecast

    def forward(self, y_hat, set_different_copula=True):
        """
        [Aggregation Strategy: Direct S-Matrix Multiplication]

        This method performs direct summation matrix (S-matrix) multiplication using only the bottom-level nodes, avoiding recursive parent-child tree traversals.

        1. Fundamental Differences in Hierarchies:
            - Cross-sectional (e.g., Country -> State -> City): Forms a strict tree where each node has exactly one distinct parent.
            - Temporal (e.g., 24h -> 12h, 8h, 6h -> 1h): Forms a Directed Acyclic Graph (DAG) where nodes overlap across multiple higher-level aggregates simultaneously.

        2. Limitations of Recursive (Tree-based) Traversals:
            Attempting to extract subsets or build trees for Temporal hierarchies leads to structural collapse.
            For example, an 8h accumulated node (1~8h) is technically a subset of a 12h node (1~12h),
            so a naive algorithm might incorrectly assign the 8h node as a direct child of the 12h node.
            (In reality, 12h should be constructed strictly from two 6h nodes or three 4h nodes).
            This overlap causes weight miscalculations.

        3. The Solution (Bottom-Up with S Matrix):
            By bypassing the complex entanglements of intermediate nodes, we focus entirely on the bottom-level nodes (e.g., the 1h resolution). The samples at this fundamental level are perfectly reordered (Schaake Shuffle) according to the empirical copula ranks. Once the bottom-level is aligned, simply multiplying by the S-matrix (S @ Bottom) guarantees that all higher-level nodes (12h, 8h, etc.) aggregate perfectly, achieving strict Probabilistic Coherence without conflicts.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is missing parameterized projections. Run `.fit(train_y_hat, train_y_obs)` first."
            )

        shifted_bottom = self.get_shifted_bottom_level(y_hat)

        shifted_bottom_sorted = torch.sort(shifted_bottom, dim=-1).values
        shuffled_bottom = self.apply_rank_shuffle(shifted_bottom_sorted)
        coherent_res = torch.einsum("nl,slk->snk", self.S, shuffled_bottom)

        return coherent_res

def extract_tree_from_S(S):
    """
    Find relation(parent - child) of given Summing matrix S.
    return dict[int, List[Tuple[int, float]]] = {parent_idx: [(child_idx, weight), ...ren_idxs]}
    """
    n = S.shape[0]
    supports = [set(torch.where(row > 0)[0].tolist()) for row in S]
    children_dict = {}
    for i in range(n):
        child_list = []
        for j in range(n):
            if i == j:
                continue
            if supports[j].issubset(supports[i]) and supports[j] != supports[i]:
                is_direct_child = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (
                        supports[j].issubset(supports[k])
                        and supports[k].issubset(supports[i])
                        and supports[k] != supports[j]
                        and supports[k] != supports[i]
                    ):  # node j is not a grand child of i, find node k which is in the middle of i and j
                        is_direct_child = False
                        break
                if is_direct_child:
                    c = next(iter(supports[j]))
                    weight = S[i, c] / S[j, c]
                    child_list.append((j, weight))
        if child_list:
            children_dict[i] = child_list
    return children_dict
