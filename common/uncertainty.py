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
    Utility class for visualizing uncertainty maps.
    
    Provides methods to visualize epistemic and aleatoric uncertainty
    across training sets and synthetic samples.
    """
    
    def __init__(self):
        """Initialize the uncertainty visualizer."""
        pass

    def plot_uncertainty_map(self, image: torch.Tensor, uncertainty: torch.Tensor, save_path: str) -> None:
        """
        Plot uncertainty map overlaid on an image.
        
        Args:
            image: Input image tensor of shape (C, H, W) or (H, W, C)
            uncertainty: Uncertainty tensor of shape (H, W) or matching image dimensions
            save_path: Path to save the visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import numpy as np
        from pathlib import Path
        
        # Convert to numpy
        if isinstance(image, torch.Tensor):
            img_np = image.detach().cpu().numpy()
        else:
            img_np = image
        
        if isinstance(uncertainty, torch.Tensor):
            unc_np = uncertainty.detach().cpu().numpy()
        else:
            unc_np = uncertainty
        
        # Handle different image formats
        if img_np.shape[0] == 3 or img_np.shape[0] == 1:  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # Normalize image to [0, 1]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
        img_np = np.clip(img_np, 0, 1)
        
        # Handle grayscale images
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        
        # Handle uncertainty dimensions
        if unc_np.ndim == 3:
            # If multi-channel uncertainty, use mean
            unc_np = unc_np.mean(axis=0) if unc_np.shape[0] == 3 else unc_np.mean(axis=2)
        
        # Resize uncertainty to match image if needed
        if unc_np.shape != img_np.shape[:2]:
            from scipy.ndimage import zoom
            scale_h = img_np.shape[0] / unc_np.shape[0]
            scale_w = img_np.shape[1] / unc_np.shape[1]
            unc_np = zoom(unc_np, (scale_h, scale_w), order=1)
        
        # Normalize uncertainty to [0, 1] for colormap
        if unc_np.max() > unc_np.min():
            unc_norm = (unc_np - unc_np.min()) / (unc_np.max() - unc_np.min())
        else:
            unc_norm = unc_np
        
        # Create colormap overlay
        cmap = cm.get_cmap('hot')
        uncertainty_colored = cmap(unc_norm)[:, :, :3]  # RGB only
        
        # Blend with image
        alpha = 0.5
        blended = (1 - alpha) * img_np + alpha * uncertainty_colored
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Uncertainty map
        im = axes[1].imshow(unc_np, cmap='hot')
        axes[1].set_title('Uncertainty Map')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(blended)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
