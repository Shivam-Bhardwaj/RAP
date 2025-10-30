"""
3DGS synthesizer for semantic-aware appearance changes.

This module implements semantic-aware scene synthesis for generating appearance
variations targeting specific semantic regions.
"""
import torch
import numpy as np
from typing import Optional
from torch.nn import functional as F


class SemanticSynthesizer:
    """
    Synthesizes scenes with semantic-aware appearance modifications.
    
    Produces appearance variations targeting specific semantic classes (e.g., sky,
    building, road) using 3DGS rendering capabilities.
    """
    
    def __init__(self, renderer):
        """
        Initialize the semantic synthesizer.
        
        Args:
            renderer: 3DGS renderer instance for generating modified scenes
        """
        self.renderer = renderer

    def synthesize(self, base_view: torch.Tensor, semantic_map: torch.Tensor, 
                  target_semantic_class: int, appearance_change: str = "brighten",
                  pose: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Synthesize a new image with modified appearance for a specific semantic class.
        
        Args:
            base_view: The original view to modify (C, H, W) or rendered image
            semantic_map: Tensor of shape (H, W) containing semantic segmentation
            target_semantic_class: Integer class ID to modify (e.g., class for 'sky', 'building')
            appearance_change: String describing the modification ('brighten', 'darken', 'occlude', 'color_shift')
            pose: Optional pose tensor (3, 4) if we need to re-render
            
        Returns:
            Tensor of modified image, or None if synthesis fails
        """
        try:
            # Convert to numpy if needed
            if isinstance(semantic_map, torch.Tensor):
                sem_map = semantic_map.cpu().numpy()
            else:
                sem_map = semantic_map
            
            if isinstance(base_view, torch.Tensor):
                img = base_view.cpu().numpy()
            else:
                img = base_view
            
            # Handle different image formats
            if img.shape[0] == 3 or img.shape[0] == 1:  # (C, H, W)
                img = np.transpose(img, (1, 2, 0))
            
            # Create mask for target semantic class
            mask = (sem_map == target_semantic_class).astype(np.float32)
            
            if mask.sum() == 0:
                # No pixels of this class found
                return None
            
            # Apply appearance change
            modified_img = self._apply_appearance_change(img, mask, appearance_change)
            
            # Convert back to tensor format
            if modified_img.shape[2] == 3:
                modified_img = np.transpose(modified_img, (2, 0, 1))
            
            # Normalize if needed (ImageNet normalization)
            if modified_img.max() <= 1.0:
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])[:, None, None]
                std = np.array([0.229, 0.224, 0.225])[:, None, None]
                modified_img = modified_img * std + mean
                modified_img = np.clip(modified_img, 0, 1)
            
            # Re-normalize
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std = np.array([0.229, 0.224, 0.225])[:, None, None]
            modified_img = (modified_img - mean) / std
            
            return torch.from_numpy(modified_img).float()
            
        except Exception as e:
            return None
    
    def _apply_appearance_change(self, image: np.ndarray, mask: np.ndarray, 
                                change_type: str) -> np.ndarray:
        """Apply appearance modification to masked regions."""
        modified = image.copy()
        
        # Expand mask to 3 channels if needed
        if mask.ndim == 2:
            mask_3d = np.stack([mask, mask, mask], axis=2)
        else:
            mask_3d = mask
        
        if change_type == "brighten":
            # Increase brightness
            brightness_factor = 1.5
            modified = modified + (mask_3d * (brightness_factor - 1.0) * image)
            modified = np.clip(modified, 0, 1)
            
        elif change_type == "darken":
            # Decrease brightness
            brightness_factor = 0.5
            modified = modified * (1.0 - mask_3d * (1.0 - brightness_factor))
            
        elif change_type == "occlude":
            # Simulate occlusion with black/masked region
            modified = modified * (1.0 - mask_3d * 0.8)
            
        elif change_type == "color_shift":
            # Shift color towards a specific hue
            # For sky: shift towards blue
            # For building: shift towards gray
            color_shift = np.array([0.1, 0.1, 0.2])  # Blue shift
            modified = modified + (mask_3d * color_shift)
            modified = np.clip(modified, 0, 1)
            
        elif change_type == "saturate":
            # Increase saturation
            gray = np.sum(modified * np.array([0.299, 0.587, 0.114]), axis=2, keepdims=True)
            modified = gray + (modified - gray) * 1.5
            modified = np.clip(modified, 0, 1)
            
        elif change_type == "desaturate":
            # Decrease saturation (grayscale)
            gray = np.sum(modified * np.array([0.299, 0.587, 0.114]), axis=2, keepdims=True)
            modified = gray + (modified - gray) * 0.3
        
        return modified
