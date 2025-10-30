"""
Uncertainty estimation utilities.
"""
import torch
import torch.nn as nn

def epistemic_uncertainty(samples):
    """
    Compute epistemic uncertainty from a set of samples (e.g., from Monte Carlo Dropout).
    Args:
        samples (Tensor): Tensor of shape (n_samples, batch_size, output_dim)
    Returns:
        Tensor: Epistemic uncertainty of shape (batch_size, output_dim)
    """
    return torch.var(samples, dim=0)

def aleatoric_uncertainty_regression(log_variance):
    """
    Compute aleatoric uncertainty for regression tasks.
    The model should output a log_variance.
    Args:
        log_variance (Tensor): Tensor of shape (batch_size, output_dim)
    Returns:
        Tensor: Aleatoric uncertainty (variance) of shape (batch_size, output_dim)
    """
    return torch.exp(log_variance)

class UncertaintyVisualizer:
    """
    A class to help visualize uncertainty maps.
    """
    def __init__(self):
        pass

    def plot_uncertainty_map(self, image, uncertainty, save_path):
        """
        Plots the uncertainty map over an image.
        """
        # Placeholder for visualization logic
        print(f"Visualizing uncertainty map and saving to {save_path}")
        pass
