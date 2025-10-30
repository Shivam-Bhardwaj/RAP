"""
Negative Log-Likelihood Loss for Mixture Density Networks.
"""
import torch
import torch.nn as nn

class MixtureNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mixture_distribution, target):
        """
        Computes the negative log-likelihood of the target under the mixture distribution.
        
        Args:
            mixture_distribution: A torch.distributions object (e.g., MixtureSameFamily).
            target: The ground truth tensor.
        
        Returns:
            The NLL loss.
        """
        log_prob = mixture_distribution.log_prob(target)
        return -log_prob.mean()
