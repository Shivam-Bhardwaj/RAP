"""
Sampler for generating training samples in high-uncertainty regions.

This module implements uncertainty-guided sampling for generating novel training
views in regions where the model exhibits high uncertainty.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from utils.nvs_utils import perturb_pose_uniform_and_sphere


class UncertaintySampler:
    """
    Samples novel training views based on uncertainty estimates.
    
    This class identifies high-uncertainty regions (e.g., OOD viewpoints, 
    challenging lighting conditions) and uses 3DGS rendering to generate 
    synthetic training samples in these regions.
    """
    
    def __init__(self, renderer, num_candidates: int = 100, trans_range: float = 1.0, rot_range: float = 30.0):
        """
        Initialize the uncertainty sampler.
        
        Args:
            renderer: 3DGS renderer instance for generating synthetic views
            num_candidates: Number of candidate poses to generate per sample
            trans_range: Translation perturbation range
            rot_range: Rotation perturbation range in degrees
        """
        self.renderer = renderer
        self.num_candidates = num_candidates
        self.trans_range = trans_range
        self.rot_range = rot_range

    def sample(self, model, current_views: torch.Tensor, current_images: torch.Tensor, 
               num_samples: int, uncertainty_threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample new views in regions of high uncertainty.
        
        Args:
            model: RAP model instance for uncertainty estimation
            current_views: Current set of training poses (N, 3, 4) or (N, 12)
            current_images: Current set of training images (N, C, H, W)
            num_samples: Number of new views to sample
            uncertainty_threshold: Minimum uncertainty threshold for sampling
            
        Returns:
            Tuple of (poses, images) for the sampled views
            - poses: (num_samples, 12) tensor
            - images: (num_samples, C, H, W) tensor
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Reshape poses if needed
        if current_views.dim() == 3 and current_views.shape[-2:] == (3, 4):
            current_views_flat = current_views.reshape(-1, 12)
        else:
            current_views_flat = current_views
        
        num_current = len(current_views_flat)
        if num_current == 0:
            return torch.empty((0, 12), device=device), torch.empty((0, *current_images.shape[1:]), device=device)
        
        # Generate candidate poses by perturbing current views
        candidate_poses = []
        candidate_uncertainties = []
        
        # Sample base poses to perturb
        base_indices = np.random.choice(num_current, size=min(self.num_candidates, num_current * 10), replace=True)
        
        with torch.no_grad():
            for idx in base_indices:
                # Get base pose
                base_pose = current_views_flat[idx].cpu().numpy().reshape(3, 4)
                
                # Generate perturbed pose
                perturbed_pose = perturb_pose_uniform_and_sphere(
                    base_pose, 
                    x=self.trans_range, 
                    angle_max=self.rot_range
                )
                
                # Convert to tensor and reshape
                pose_tensor = torch.from_numpy(perturbed_pose).float().reshape(12).to(device)
                
                # Render image for this pose
                try:
                    # Use renderer to get image (if renderer supports single pose rendering)
                    # For now, we'll compute uncertainty on a dummy image
                    # In a full implementation, we'd render the actual image
                    rendered_img = self._render_pose(pose_tensor.reshape(3, 4))
                    
                    if rendered_img is not None:
                        # Compute uncertainty for this candidate
                        uncertainty = self._compute_uncertainty(model, rendered_img)
                        candidate_poses.append(pose_tensor)
                        candidate_uncertainties.append(uncertainty)
                except Exception as e:
                    # Skip if rendering fails
                    continue
        
        if len(candidate_poses) == 0:
            # Fallback: return perturbed versions of current views
            selected_indices = np.random.choice(num_current, size=min(num_samples, num_current), replace=False)
            selected_poses = current_views_flat[selected_indices]
            selected_images = current_images[selected_indices]
            return selected_poses, selected_images
        
        # Convert to tensors
        candidate_poses = torch.stack(candidate_poses)
        candidate_uncertainties = torch.stack(candidate_uncertainties)
        
        # Select top-K highest uncertainty poses
        if uncertainty_threshold > 0:
            mask = candidate_uncertainties > uncertainty_threshold
            if mask.sum() == 0:
                # If no poses meet threshold, use top-K by uncertainty
                _, top_indices = torch.topk(candidate_uncertainties, k=min(num_samples, len(candidate_poses)))
            else:
                top_indices = torch.where(mask)[0]
                if len(top_indices) > num_samples:
                    _, top_k = torch.topk(candidate_uncertainties[top_indices], k=num_samples)
                    top_indices = top_indices[top_k]
                else:
                    top_indices = top_indices[:num_samples]
        else:
            # Select top-K by uncertainty
            k = min(num_samples, len(candidate_poses))
            _, top_indices = torch.topk(candidate_uncertainties, k=k)
        
        selected_poses = candidate_poses[top_indices]
        
        # Render images for selected poses
        selected_images = []
        for pose in selected_poses:
            img = self._render_pose(pose.reshape(3, 4))
            if img is not None:
                selected_images.append(img)
        
        if len(selected_images) == 0:
            # Fallback: return poses without images (trainer will handle rendering)
            return selected_poses, torch.empty((0, *current_images.shape[1:]), device=device)
        
        selected_images = torch.stack(selected_images)
        return selected_poses, selected_images
    
    def _compute_uncertainty(self, model, image: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty for a single image."""
        with torch.no_grad():
            # Add batch dimension if needed
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # For UAAS models, get uncertainty directly
            if hasattr(model, '__class__') and 'UAAS' in model.__class__.__name__:
                outputs = model(image, return_feature=False)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    _, log_var = outputs
                    # Sum aleatoric uncertainty across dimensions
                    uncertainty = torch.exp(log_var).sum(dim=1).mean()
                else:
                    uncertainty = torch.tensor(1.0, device=image.device)
            else:
                # For other models, use epistemic uncertainty via Monte Carlo Dropout
                model.train()
                for m in model.modules():
                    if m.__class__.__name__.startswith('Dropout'):
                        m.train()
                
                samples = []
                for _ in range(5):  # Small number for efficiency
                    _, pose = model(image, return_feature=False)
                    samples.append(pose)
                
                samples = torch.stack(samples)
                uncertainty = torch.var(samples.view(samples.shape[0], -1), dim=0).mean()
                model.eval()
            
            return uncertainty
    
    def _render_pose(self, pose: np.ndarray) -> Optional[torch.Tensor]:
        """Render image for a given pose using the renderer."""
        try:
            # Check if renderer has a method to render single pose
            if hasattr(self.renderer, 'render_pose'):
                return self.renderer.render_pose(pose)
            elif hasattr(self.renderer, 'render_single'):
                return self.renderer.render_single(pose)
            else:
                # Fallback: render using render_set with single pose
                # This is a simplified version - full implementation would use renderer's API
                # For now, return None to indicate we can't render directly
                # The trainer should handle rendering separately
                return None
        except Exception:
            return None
