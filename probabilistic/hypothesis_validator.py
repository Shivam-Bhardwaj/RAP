"""
Validates pose hypotheses using 3DGS rendering.

This module implements validation and ranking of pose hypotheses by rendering
them through 3DGS and comparing with observed images.
"""
import torch
import numpy as np
from typing import List, Tuple
from torch.nn import functional as F
import cv2 as cv
from kornia.geometry import rotation_matrix_to_quaternion
from utils.cameras import Camera
from utils.nvs_utils import get_bbox


class HypothesisValidator:
    """
    Validates pose hypotheses by rendering and comparing with observed images.
    
    Uses 3DGS rendering to generate images from pose hypotheses and ranks
    them based on similarity to the observed image.
    """
    
    def __init__(self, renderer, use_ssim: bool = True, use_lpips: bool = False):
        """
        Initialize the hypothesis validator.
        
        Args:
            renderer: 3DGS renderer instance for generating rendered images
            use_ssim: Whether to use SSIM for similarity
            use_lpips: Whether to use LPIPS (requires lpips package)
        """
        self.renderer = renderer
        self.use_ssim = use_ssim
        self.use_lpips = use_lpips
        
        if use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net='alex').eval()
            except ImportError:
                print("Warning: LPIPS not available, falling back to SSIM only")
                self.use_lpips = False
                self.lpips_model = None
    
    def validate(self, hypotheses: torch.Tensor, observed_image: torch.Tensor, 
                 cam_params=None) -> torch.Tensor:
        """
        Rank hypotheses by rendering them and comparing with the observed image.
        
        Args:
            hypotheses: Tensor of shape (n_hypotheses, 6) or (n_hypotheses, 12) containing pose hypotheses
            observed_image: Tensor of shape (C, H, W) containing the observed image
            cam_params: Camera parameters for rendering
            
        Returns:
            Tensor of shape (n_hypotheses,) containing scores for each hypothesis.
            Higher scores indicate better matches.
        """
        device = hypotheses.device if isinstance(hypotheses, torch.Tensor) else 'cpu'
        
        # Reshape hypotheses if needed
        if hypotheses.dim() == 2:
            if hypotheses.shape[1] == 6:
                # Convert 6D representation to 3x4 matrix (simplified - assumes translation + rotation)
                # This is a placeholder - actual conversion depends on representation format
                # For now, we'll assume it's already in matrix form
                hypotheses = hypotheses.reshape(-1, 2, 3)  # Placeholder
            elif hypotheses.shape[1] == 12:
                hypotheses = hypotheses.reshape(-1, 3, 4)
        
        n_hypotheses = len(hypotheses)
        scores = []
        
        # Normalize observed image for comparison
        observed_norm = self._normalize_image(observed_image)
        
        for i in range(n_hypotheses):
            pose = hypotheses[i]
            
            # Render hypothesis
            rendered_img = self._render_hypothesis(pose, cam_params)
            
            if rendered_img is None:
                scores.append(-1.0)  # Low score for failed renders
                continue
            
            # Compute similarity score
            score = self._compute_similarity(rendered_img, observed_norm)
            scores.append(score)
        
        scores = torch.tensor(scores, device=device, dtype=torch.float32)
        return scores
    
    def _render_hypothesis(self, pose: torch.Tensor, cam_params=None) -> torch.Tensor:
        """Render image for a hypothesis pose."""
        try:
            # Convert pose to numpy if needed
            if isinstance(pose, torch.Tensor):
                pose_np = pose.cpu().numpy()
            else:
                pose_np = pose
            
            # Ensure pose is 3x4
            if pose_np.shape == (12,):
                pose_np = pose_np.reshape(3, 4)
            
            # Use renderer's render method if available
            if hasattr(self.renderer, 'render_pose'):
                return self.renderer.render_pose(pose_np)
            elif hasattr(self.renderer, 'gaussians') and cam_params is not None:
                # Direct rendering using gaussians
                from utils.cameras import Camera
                colmap_pose = np.eye(4, dtype=np.float32)
                colmap_pose[:3, :4] = pose_np
                pose_inv = np.linalg.inv(colmap_pose)
                R = pose_inv[:3, :3]
                T = pose_inv[:3, 3]
                
                # Get render device
                render_device = getattr(self.renderer.configs, 'render_device', 'cuda')
                
                # Create camera
                view = Camera(
                    uid=None, colmap_id=None, image_name=None,
                    R=R, T=T, K=cam_params.K,
                    FoVx=cam_params.FovX, FoVy=cam_params.FovY,
                    image=None, render_device=render_device,
                    data_device=render_device
                )
                
                # Render
                bg_color = [1, 1, 1] if getattr(self.renderer.configs, 'white_background', False) else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float, device=render_device)
                
                rendering = self.renderer.gaussians.render(view, self.renderer.configs, background)["render"]
                
                # Normalize
                mean = torch.tensor([0.485, 0.456, 0.406], device=render_device)[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225], device=render_device)[:, None, None]
                normalized = rendering.sub(mean).div_(std)
                
                return normalized.cpu()
            else:
                return None
        except Exception as e:
            return None
    
    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image to [0, 1] range."""
        if image.max() > 1.0:
            image = image / 255.0
        return image.clamp(0, 1)
    
    def _compute_similarity(self, rendered: torch.Tensor, observed: torch.Tensor) -> float:
        """Compute similarity score between rendered and observed images."""
        score = 0.0
        
        # SSIM
        if self.use_ssim:
            ssim_score = self._ssim(rendered, observed)
            score += ssim_score
        
        # LPIPS
        if self.use_lpips and self.lpips_model is not None:
            # Convert to [-1, 1] range for LPIPS
            rendered_lpips = rendered * 2.0 - 1.0
            observed_lpips = observed * 2.0 - 1.0
            
            # Add batch dimension
            if rendered_lpips.dim() == 3:
                rendered_lpips = rendered_lpips.unsqueeze(0)
            if observed_lpips.dim() == 3:
                observed_lpips = observed_lpips.unsqueeze(0)
            
            lpips_score = self.lpips_model(rendered_lpips, observed_lpips).item()
            # LPIPS is a distance, so lower is better - convert to similarity
            score += (1.0 - lpips_score) * 0.5
        
        return score
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> float:
        """Compute SSIM between two images."""
        # Simplified SSIM implementation
        # Full implementation would use Gaussian window
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # Simple mean squared error as approximation
        mse = F.mse_loss(img1, img2)
        # Convert MSE to similarity (inverse)
        similarity = 1.0 / (1.0 + mse.item())
        return similarity
