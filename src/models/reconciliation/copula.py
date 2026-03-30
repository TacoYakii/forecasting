import torch 
import torch.nn as nn 

class PermutationEmpiricalCopula(nn.Module):
    """
    Samples rank dependencies from the Empirical Copula to reorder out-of-sample forecasts.

    Procedural Steps:
    1. Base Empirical Distribution: Extract the CDF probabilities (u_t) of the actual observed values 
        relative to the in-sample base forecast distributions. This Empirical Distribution has a shape of (T, num_nodes).
    2. Rank Extraction: Sample `num_sim` historical time indices (between 0 and T) from the empirical distribution, 
        and compute their cross-sectional ascending order ranks. This gives us the rank of each u_it within its respective sample.
    3. Gather (Schaake Shuffle): Generate K independent out-of-sample forecasts and sort them in ascending order. 
        Then, using the sampled indices (which have a shape needed for each forward pass batch), 
        gather the sorted samples to restore the empirical dependency structure.
    """

    def init_empirical_copula(self, device):
        self.register_buffer("empirical_distribution", torch.empty(0, device=device))
        self.register_buffer("bottom_ranks", torch.empty(0, device=device)) 

    def fit_empirical_copula(self, in_sample_hat, in_sample_obs):
        self.T, self.num_nodes, self.num_sim = in_sample_hat.shape
        in_sample_hat_sorted, _ = torch.sort(in_sample_hat, dim=-1)
        self.empirical_distribution = (in_sample_hat_sorted <= in_sample_obs).float().mean(dim=-1) # Probability Integral Transform - (u_t,num_nodes) 
        
    def sample_rank(self, batch_size):
        indices = torch.randint(
            low=0, high=self.T, size=(batch_size, self.num_sim),
            device=self.empirical_distribution.device
        )
        selected_dist = self.empirical_distribution[indices]
        ranks = selected_dist.argsort(dim=1).argsort(dim=1)
        ranks = ranks.transpose(1, 2)

        return ranks

    def apply_rank_shuffle(self, out_sample_hat):
        if out_sample_hat.shape[1:] != (self.num_nodes, self.num_sim): 
            raise ValueError("Out-sample forecast shape does not match empirical distribution shape")
        
        sorted_out_sample_hat  = torch.sort(out_sample_hat, dim=-1).values
        batch_size = out_sample_hat.shape[0] 

        ranks = self.sample_rank(batch_size) 
        reordered_hat = torch.gather(sorted_out_sample_hat, dim=-1, index=ranks.long())

        return reordered_hat
