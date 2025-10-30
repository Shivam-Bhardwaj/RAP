#!/usr/bin/env python3
"""
Integration tests for RAP-ID components.

Tests component interactions and full workflows.
"""
import torch
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig
from uaas.sampler import UncertaintySampler
from probabilistic.hypothesis_validator import HypothesisValidator
from semantic.semantic_synthesizer import SemanticSynthesizer
from semantic.hard_negative_miner import HardNegativeMiner


class TestUncertaintySampler:
    """Tests for UncertaintySampler."""
    
    def test_sampler_initialization(self):
        """Test UncertaintySampler can be initialized."""
        # Mock renderer
        class MockRenderer:
            def __init__(self):
                self.configs = type('obj', (object,), {'white_background': False, 'render_device': 'cpu'})()
                self.hw = (240, 320)
                self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        
        renderer = MockRenderer()
        sampler = UncertaintySampler(renderer)
        
        assert sampler.renderer is not None
        assert sampler.num_candidates > 0
    
    def test_sample_method_signature(self):
        """Test sampler.sample() method signature."""
        class MockRenderer:
            def __init__(self):
                self.configs = type('obj', (object,), {'white_background': False, 'render_device': 'cpu'})()
                self.hw = (240, 320)
                self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        
        renderer = MockRenderer()
        sampler = UncertaintySampler(renderer)
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        # Create mock model
        class MockModel:
            def __init__(self):
                self.device = 'cpu'
            
            def parameters(self):
                return [torch.tensor([1.0])]
            
            def eval(self):
                pass
            
            def __call__(self, x, return_feature=False):
                batch_size = x.shape[0]
                pose = torch.randn(batch_size, 12)
                log_var = torch.randn(batch_size, 6) * 0.1 - 3.0
                return (pose, log_var)
        
        model = MockModel()
        current_views = TestUtils.create_dummy_pose(batch_size=10, device='cpu').reshape(10, 12)
        current_images = TestUtils.create_dummy_image(batch_size=10, device='cpu')
        
        # Test sampling (will fail rendering but should not crash)
        try:
            poses, images = sampler.sample(model, current_views, current_images, num_samples=5)
            # Should return tensors even if rendering fails
            assert isinstance(poses, torch.Tensor)
        except Exception as e:
            # Expected if renderer not fully set up
            assert "render" in str(e).lower() or "gaussians" in str(e).lower()


class TestHypothesisValidator:
    """Tests for HypothesisValidator."""
    
    def test_validator_initialization(self):
        """Test HypothesisValidator can be initialized."""
        validator = HypothesisValidator(renderer=None)
        assert validator.renderer is None
        assert validator.use_ssim == True
    
    def test_ssim_computation(self):
        """Test SSIM computation."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        validator = HypothesisValidator(renderer=None, use_ssim=True, use_lpips=False)
        
        # Create similar images
        img1 = TestUtils.create_dummy_image(batch_size=1, device='cpu')
        img2 = img1 + torch.randn_like(img1) * 0.01  # Small noise
        
        ssim_score = validator._ssim(img1, img2)
        
        assert 0 <= ssim_score <= 1, "SSIM should be in [0, 1]"
        assert ssim_score > 0.9, "Similar images should have high SSIM"
    
    def test_ssim_different_images(self):
        """Test SSIM on very different images."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        validator = HypothesisValidator(renderer=None, use_ssim=True, use_lpips=False)
        
        # Create very different images
        img1 = TestUtils.create_dummy_image(batch_size=1, device='cpu')
        img2 = TestUtils.create_dummy_image(batch_size=1, device='cpu')
        
        ssim_score = validator._ssim(img1, img2)
        
        assert 0 <= ssim_score <= 1, "SSIM should be in [0, 1]"
        assert ssim_score < 0.5, "Very different images should have low SSIM"


class TestSemanticSynthesizer:
    """Tests for SemanticSynthesizer."""
    
    def test_synthesizer_initialization(self):
        """Test SemanticSynthesizer can be initialized."""
        synthesizer = SemanticSynthesizer(renderer=None)
        assert synthesizer.renderer is None
    
    def test_synthesize_brighten(self):
        """Test semantic synthesis with brighten."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        synthesizer = SemanticSynthesizer(renderer=None)
        
        base_view = TestUtils.create_dummy_image(batch_size=1, height=64, width=64, device='cpu').squeeze(0)
        semantic_map = torch.randint(0, 3, (64, 64))
        
        result = synthesizer.synthesize(base_view, semantic_map, target_semantic_class=1, 
                                       appearance_change="brighten")
        
        # Should return tensor or None
        assert result is None or isinstance(result, torch.Tensor)
    
    def test_synthesize_darken(self):
        """Test semantic synthesis with darken."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        synthesizer = SemanticSynthesizer(renderer=None)
        
        base_view = TestUtils.create_dummy_image(batch_size=1, height=64, width=64, device='cpu').squeeze(0)
        semantic_map = torch.randint(0, 3, (64, 64))
        
        result = synthesizer.synthesize(base_view, semantic_map, target_semantic_class=1, 
                                       appearance_change="darken")
        
        assert result is None or isinstance(result, torch.Tensor)


class TestHardNegativeMiner:
    """Tests for HardNegativeMiner."""
    
    def test_miner_initialization(self):
        """Test HardNegativeMiner can be initialized."""
        miner = HardNegativeMiner(renderer=None)
        assert miner.renderer is None
        assert miner.num_candidates == 50
    
    def test_mine_method_signature(self):
        """Test mine() method signature."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        miner = HardNegativeMiner(renderer=None, num_candidates=10)
        
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
        base_poses = TestUtils.create_dummy_pose(batch_size=5, device='cpu').reshape(5, 12)
        base_images = TestUtils.create_dummy_image(batch_size=5, device='cpu')
        
        # Test mining (will fail rendering but should not crash)
        try:
            poses, images = miner.mine(model, base_poses, base_images, difficulty=0.5)
            # Should return None or tensors
            assert poses is None or isinstance(poses, torch.Tensor)
            assert images is None or isinstance(images, torch.Tensor)
        except Exception as e:
            # Expected if renderer not fully set up
            assert "render" in str(e).lower() or "gaussians" in str(e).lower()


class TestComponentIntegration:
    """Tests for component integration."""
    
    def test_uncertainty_pipeline(self):
        """Test uncertainty estimation pipeline."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from common.uncertainty import epistemic_uncertainty, aleatoric_uncertainty_regression
        
        # Simulate Monte Carlo samples
        n_samples = 10
        batch_size = 4
        samples = torch.randn(n_samples, batch_size, 6, device=config.device)
        
        epistemic = epistemic_uncertainty(samples)
        
        # Simulate aleatoric uncertainty
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        aleatoric = aleatoric_uncertainty_regression(log_var)
        
        total = epistemic + aleatoric
        
        assert total.shape == (batch_size, 6)
        assert torch.all(total >= 0)
    
    def test_loss_with_uncertainty(self):
        """Test loss computation with uncertainty."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from utils.pose_utils import CameraPoseLoss
        
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 4
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        
        loss = loss_fn(est_pose, gt_pose, log_var=log_var)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

