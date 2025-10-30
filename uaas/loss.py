"""
Uncertainty-weighted adversarial loss.
"""
import torch
import torch.nn as nn

class UncertaintyWeightedAdversarialLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss()):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, disc_out_fake, valid_labels, uncertainty_weights):
        """
        Args:
            disc_out_fake: Discriminator output for fake samples.
            valid_labels: Real labels for adversarial loss.
            uncertainty_weights: Weights based on uncertainty.
        """
        adversarial_loss = self.base_loss(disc_out_fake, valid_labels)
        weighted_loss = (adversarial_loss * uncertainty_weights).mean()
        return weighted_loss
