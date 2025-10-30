"""
Mines hard negative samples by adversarially tweaking semantic regions.

This module implements adversarial hard negative mining to create challenging
training samples that maximize prediction errors through semantic region manipulation.
"""
import torch
import numpy as np
from typing import Tuple, Optional
from utils.nvs_utils import perturb_pose_uniform_and_sphere


class HardNegativeMiner:
    """
    Mines hard negative samples through adversarial semantic scene manipulation.
    
    Creates synthetic scenes designed to maximize RAP prediction errors by
    adversarially modifying semantic regions (e.g., changing sky appearance,
    occluding buildings).
    """
    
    def __init__(self, renderer, num_candidates: int = 50):
        """
        Initialize the hard negative miner.
        
        Args:
            renderer: 3DGS renderer instance for generating synthetic scenes
            num_candidates: Number of candidate modifications to try
        """
        self.renderer = renderer
        self.num_candidates = num_candidates

    def mine(self, model, base_poses: torch.Tensor, base_images: torch.Tensor,
             semantic_maps: Optional[torch.Tensor] = None,
             difficulty: float = 0.5) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Create synthetic scenes designed to maximize RAP prediction error.
        
        Args:
            model: The RAP model to attack
            base_poses: Tensor of shape (N, 12) containing starting poses
            base_images: Tensor of shape (N, C, H, W) containing base images
            semantic_maps: Optional tensor of shape (N, H, W) with semantic segmentation
            difficulty: Current difficulty level from curriculum (0.0 to 1.0)
            
        Returns:
            Tuple of (poses, images) for generated hard negatives, or (None, None) if empty
        """
        device = next(model.parameters()).device
        model.eval()
        
        num_base = len(base_poses)
        if num_base == 0:
            return None, None
        
        # Sample subset of poses to perturb
        num_to_perturb = min(self.num_candidates, num_base)
        indices = np.random.choice(num_base, size=num_to_perturb, replace=False)
        
        hard_negative_poses = []
        hard_negative_images = []
        hard_negative_scores = []
        
        with torch.no_grad():
            for idx in indices:
                base_pose = base_poses[idx]
                base_img = base_images[idx] if base_images is not None else None
                
                # Generate perturbed pose
                pose_np = base_pose.cpu().numpy().reshape(3, 4)
                
                # Scale perturbation by difficulty
                trans_range = difficulty * 2.0  # Max 2.0 at full difficulty
                rot_range = difficulty * 45.0  # Max 45 degrees at full difficulty
                
                perturbed_pose = perturb_pose_uniform_and_sphere(
                    pose_np,
                    x=trans_range,
                    angle_max=rot_range
                )
                
                perturbed_pose_tensor = torch.from_numpy(perturbed_pose).float().reshape(12).to(device)
                
                # Render perturbed image
                rendered_img = self._render_pose(perturbed_pose_tensor.reshape(3, 4))
                
                if rendered_img is None:
                    continue
                
                # Apply semantic modifications if available
                if semantic_maps is not None:
                    sem_map = semantic_maps[idx]
                    modified_img = self._apply_semantic_perturbation(
                        rendered_img, sem_map, difficulty
                    )
                else:
                    modified_img = rendered_img
                
                # Compute prediction error (how "hard" this negative is)
                error_score = self._compute_prediction_error(
                    model, modified_img, base_pose
                )
                
                hard_negative_poses.append(perturbed_pose_tensor)
                hard_negative_images.append(modified_img)
                hard_negative_scores.append(error_score)
        
        if len(hard_negative_poses) == 0:
            return None, None
        
        # Sort by error score (highest first) and select top-K
        hard_negative_scores = torch.stack(hard_negative_scores)
        _, top_indices = torch.topk(hard_negative_scores, k=min(10, len(hard_negative_poses)))
        
        selected_poses = torch.stack([hard_negative_poses[i] for i in top_indices])
        selected_images = torch.stack([hard_negative_images[i] for i in top_indices])
        
        return selected_poses, selected_images
    
    def _render_pose(self, pose: torch.Tensor) -> Optional[torch.Tensor]:
        """Render image for a given pose."""
        try:
            if hasattr(self.renderer, 'render_pose'):
                return self.renderer.render_pose(pose)
            elif hasattr(self.renderer, 'render_single'):
                return self.renderer.render_single(pose)
            else:
                # Fallback: return None (trainer will handle rendering)
                return None
        except Exception:
            return None
    
    def _apply_semantic_perturbation(self, image: torch.Tensor, 
                                    semantic_map: torch.Tensor,
                                    difficulty: float) -> torch.Tensor:
        """Apply semantic-aware perturbations to image."""
        # Convert to numpy
        img_np = image.cpu().numpy()
        sem_map = semantic_map.cpu().numpy()
        
        # Handle image format
        if img_np.shape[0] == 3:  # (C, H, W)
            img_np = np.transpose(img_np, (1, 2, 0))
        
        # Normalize if needed
        if img_np.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
        
        # Find most common semantic classes
        unique_classes, counts = np.unique(sem_map, return_counts=True)
        top_classes = unique_classes[np.argsort(counts)[-3:]]  # Top 3 classes
        
        # Apply perturbations to each class
        modified = img_np.copy()
        for cls in top_classes:
            mask = (sem_map == cls).astype(np.float32)
            if mask.sum() == 0:
                continue
            
            # Random perturbation type based on difficulty
            if np.random.rand() < 0.5:
                # Brightness change
                factor = 1.0 + difficulty * np.random.uniform(-0.5, 0.5)
                modified = modified * (1.0 - mask[:, :, None]) + modified * mask[:, :, None] * factor
            else:
                # Color shift
                shift = difficulty * np.random.uniform(-0.2, 0.2, 3)
                modified = modified + mask[:, :, None] * shift
        
        modified = np.clip(modified, 0, 1)
        
        # Convert back to tensor
        if modified.shape[2] == 3:
            modified = np.transpose(modified, (2, 0, 1))
        
        # Re-normalize
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        modified = (modified - mean) / std
        
        return torch.from_numpy(modified).float().to(image.device)
    
    def _compute_prediction_error(self, model, image: torch.Tensor, 
                                  gt_pose: torch.Tensor) -> torch.Tensor:
        """Compute prediction error for a given image and ground truth pose."""
        # Add batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if gt_pose.dim() == 1:
            gt_pose = gt_pose.unsqueeze(0)
        
        # Enable gradients for adversarial optimization
        image.requires_grad = True
        
        # Get prediction
        outputs = model(image, return_feature=False)
        if isinstance(outputs, tuple):
            pred_pose = outputs[0]
        else:
            _, pred_pose = outputs if isinstance(outputs, tuple) else (None, outputs)
        
        # Compute error (L2 norm)
        error = torch.norm(pred_pose - gt_pose, dim=1).mean()
        
        # Adversarial: maximize error through gradient ascent
        if error.requires_grad:
            error.backward()
            
            # Gradient-based perturbation (PGD-style)
            with torch.no_grad():
                # Get gradient magnitude
                grad_norm = image.grad.norm()
                if grad_norm > 0:
                    # Normalize gradient
                    normalized_grad = image.grad / (grad_norm + 1e-8)
                    # Apply perturbation proportional to difficulty
                    perturbation = normalized_grad * 0.1  # Small step size
                    image = image + perturbation
                    image = image.clamp(0, 1)
                    image.requires_grad = False
        
        return error.detach()
