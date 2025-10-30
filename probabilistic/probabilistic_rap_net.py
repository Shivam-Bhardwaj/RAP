"""
Probabilistic RAPNet using a Mixture Density Network head.
"""
import torch
import torch.nn as nn
import torch.distributions as D
from models.apr.rapnet import RAPNet

class ProbabilisticRAPNet(RAPNet):
    def __init__(self, args, num_gaussians=5):
        super().__init__(args)
        self.num_gaussians = num_gaussians
        self.mdn_head = nn.Linear(self.feature_dim, num_gaussians * (1 + 6 + 6)) # pi, mu, sigma for 6-DoF pose

    def forward(self, x, return_feature=False):
        features, _ = super().forward(x, return_feature=True)
        
        mdn_params = self.mdn_head(features)
        
        pi_logits, mu, log_sigma = mdn_params.split([self.num_gaussians, 
                                                     self.num_gaussians * 6, 
                                                     self.num_gaussians * 6], dim=-1)
        
        pi = D.Categorical(logits=pi_logits)
        mus = mu.view(-1, self.num_gaussians, 6)
        # Clamp log_sigma for numerical stability
        log_sigma_clamped = torch.clamp(log_sigma, min=-10, max=10)
        sigmas = torch.exp(log_sigma_clamped).view(-1, self.num_gaussians, 6)
        # Ensure minimum sigma to avoid numerical issues
        sigmas = torch.clamp(sigmas, min=1e-6)
        component_distribution = D.Independent(D.Normal(loc=mus, scale=sigmas), 1)
        mixture_distribution = D.MixtureSameFamily(pi, component_distribution)

        if return_feature:
            return (features, mixture_distribution)
        else:
            return mixture_distribution
