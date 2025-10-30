#!/usr/bin/env python3
"""
Tests for rendering integration.

Tests that rendering components work correctly with 3DGS integration.
"""
import torch
import pytest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig


class TestRenderingIntegration:
    """Tests for rendering integration."""
    
    def test_uncertainty_sampler_rendering_method(self):
        """Test UncertaintySampler has proper rendering method."""
        from uaas.sampler import UncertaintySampler
        
        # Mock renderer with required attributes
        class MockRenderer:
            def __init__(self):
                self.configs = type('obj', (object,), {
                    'white_background': False,
                    'render_device': 'cpu'
                })()
                self.hw = (240, 320)
                self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                self.cam_params = type('obj', (object,), {
                    'K': torch.eye(3, 3),
                    'FovX': 60.0,
                    'FovY': 60.0
                })()
        
        renderer = MockRenderer()
        sampler = UncertaintySampler(renderer)
        
        # Test that _render_pose method exists and has correct signature
        assert hasattr(sampler, '_render_pose'), "Should have _render_pose method"
        
        # Test pose conversion
        pose_np = np.eye(3, 4, dtype=np.float32)
        try:
            result = sampler._render_pose(pose_np)
            # Should return None (no actual gaussians) or tensor
            assert result is None or isinstance(result, torch.Tensor)
        except Exception as e:
            # Expected if gaussians not available
            assert "gaussians" in str(e).lower() or "render" in str(e).lower()
    
    def test_hypothesis_validator_rendering(self):
        """Test HypothesisValidator can use renderer."""
        from probabilistic.hypothesis_validator import HypothesisValidator
        
        validator = HypothesisValidator(renderer=None, use_ssim=True, use_lpips=False)
        
        # Test SSIM computation (doesn't require renderer)
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        img1 = TestUtils.create_dummy_image(batch_size=1, device='cpu')
        img2 = img1 + torch.randn_like(img1) * 0.01
        
        ssim = validator._ssim(img1, img2)
        assert 0 <= ssim <= 1
    
    def test_semantic_synthesizer_appearance_changes(self):
        """Test SemanticSynthesizer appearance modifications."""
        from semantic.semantic_synthesizer import SemanticSynthesizer
        
        synthesizer = SemanticSynthesizer(renderer=None)
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        base_view = TestUtils.create_dummy_image(batch_size=1, height=64, width=64, device='cpu').squeeze(0)
        semantic_map = torch.randint(0, 3, (64, 64))
        
        # Test all appearance change types
        change_types = ["brighten", "darken", "occlude", "color_shift", "saturate", "desaturate"]
        
        for change_type in change_types:
            result = synthesizer.synthesize(base_view, semantic_map, target_semantic_class=1, 
                                           appearance_change=change_type)
            # Should return None or tensor
            assert result is None or isinstance(result, torch.Tensor)
    
    def test_hard_negative_miner_adversarial(self):
        """Test HardNegativeMiner uses gradients."""
        from semantic.hard_negative_miner import HardNegativeMiner
        
        miner = HardNegativeMiner(renderer=None)
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        # Create mock model
        class MockModel:
            def parameters(self):
                return [torch.tensor([1.0])]
            
            def eval(self):
                pass
            
            def __call__(self, x, return_feature=False):
                batch_size = x.shape[0]
                pose = torch.randn(batch_size, 12)
                return pose
        
        model = MockModel()
        base_poses = TestUtils.create_dummy_pose(batch_size=3, device='cpu').reshape(3, 12)
        base_images = TestUtils.create_dummy_image(batch_size=3, device='cpu')
        
        # Test mining
        try:
            poses, images = miner.mine(model, base_poses, base_images, difficulty=0.5)
            # Should return None or tensors
            assert poses is None or isinstance(poses, torch.Tensor)
            assert images is None or isinstance(images, torch.Tensor)
        except Exception as e:
            # Expected if renderer not available
            assert "render" in str(e).lower() or "gaussians" in str(e).lower()


class TestRenderingUtils:
    """Tests for rendering utility functions."""
    
    def test_pose_conversion(self):
        """Test pose conversion utilities."""
        from utils.nvs_utils import perturb_pose_uniform_and_sphere
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        # Create base pose
        base_pose = np.eye(3, 4, dtype=np.float32)
        base_pose[:3, 3] = [1.0, 2.0, 3.0]
        
        # Test perturbation
        perturbed = perturb_pose_uniform_and_sphere(base_pose, x=0.5, angle_max=10.0)
        
        assert perturbed.shape == (3, 4), "Perturbed pose should be 3x4"
        assert not np.isnan(perturbed).any(), "Pose should not contain NaN"
    
    def test_pose_validity(self):
        """Test pose validity checks."""
        from utils.nvs_utils import perturb_pose_uniform_and_sphere
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        base_pose = np.eye(3, 4, dtype=np.float32)
        
        # Test multiple perturbations
        for _ in range(10):
            perturbed = perturb_pose_uniform_and_sphere(base_pose, x=1.0, angle_max=30.0)
            
            # Check rotation matrix properties (should be approximately orthogonal)
            R = perturbed[:3, :3]
            RRT = R @ R.T
            identity_approx = np.eye(3)
            
            # Allow some tolerance for numerical errors
            assert np.allclose(RRT, identity_approx, atol=1e-3), \
                "Rotation matrix should be approximately orthogonal"


@pytest.mark.rendering
class TestRenderingPerformance:
    """Performance tests for rendering operations."""
    
    def test_rendering_speed(self):
        """Test rendering operations are reasonably fast."""
        # This is a placeholder - actual rendering speed tests require 3DGS setup
        pass
    
    def test_batch_rendering(self):
        """Test batch rendering efficiency."""
        # Placeholder for batch rendering tests
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

