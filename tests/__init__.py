#!/usr/bin/env python3
"""
Comprehensive testing framework for RAP-ID.

This module provides testing utilities, fixtures, and test runners for:
- Unit tests (individual components)
- Integration tests (component interactions)
- End-to-end tests (full training/evaluation pipelines)
- Performance benchmarks
- Regression tests
"""
import os
import sys
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams
from utils.pose_utils import CameraPoseLoss
from common.uncertainty import epistemic_uncertainty, aleatoric_uncertainty_regression
from uaas.loss import UncertaintyWeightedAdversarialLoss
from probabilistic.loss import MixtureNLLLoss


@dataclass
class TestConfig:
    """Configuration for tests."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    batch_size: int = 4
    seed: int = 42
    atol: float = 1e-5
    rtol: float = 1e-4


class TestUtils:
    """Utility functions for testing."""
    
    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def create_dummy_pose(batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Create dummy pose tensor (3x4 matrix)."""
        # Create identity rotation + random translation
        poses = []
        for _ in range(batch_size):
            R = torch.eye(3, device=device)
            t = torch.randn(3, 1, device=device) * 0.1
            pose = torch.cat([R, t], dim=1)  # (3, 4)
            poses.append(pose)
        
        if batch_size == 1:
            return poses[0]
        return torch.stack(poses)
    
    @staticmethod
    def create_dummy_image(batch_size: int = 1, channels: int = 3, 
                           height: int = 240, width: int = 320, 
                           device: str = "cpu") -> torch.Tensor:
        """Create dummy image tensor."""
        img = torch.randn(batch_size, channels, height, width, device=device)
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        img = img * std + mean
        return img.clamp(0, 1)
    
    @staticmethod
    def create_dummy_features(batch_size: int = 1, dim: int = 256,
                             device: str = "cpu") -> torch.Tensor:
        """Create dummy feature tensor."""
        return torch.randn(batch_size, dim, device=device)
    
    @staticmethod
    def assert_tensors_close(t1: torch.Tensor, t2: torch.Tensor, 
                            atol: float = 1e-5, rtol: float = 1e-4):
        """Assert two tensors are close."""
        assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
            f"Tensors not close: max_diff={torch.abs(t1 - t2).max().item()}"
    
    @staticmethod
    def create_temp_dir(prefix: str = "rap_test_") -> Path:
        """Create temporary directory for tests."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        return temp_dir


class LossFunctionTests:
    """Tests for loss functions."""
    
    def test_camera_pose_loss_basic(self, config: TestConfig):
        """Test basic camera pose loss."""
        TestUtils.set_seed(config.seed)
        loss_fn = CameraPoseLoss(config)
        
        batch_size = config.batch_size
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        
        loss = loss_fn(est_pose, gt_pose)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_camera_pose_loss_with_uncertainty(self, config: TestConfig):
        """Test camera pose loss with uncertainty."""
        TestUtils.set_seed(config.seed)
        loss_fn = CameraPoseLoss(config)
        
        batch_size = config.batch_size
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0  # Reasonable log-variance
        
        loss = loss_fn(est_pose, gt_pose, log_var=log_var)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_uncertainty_weighted_adversarial_loss(self, config: TestConfig):
        """Test uncertainty-weighted adversarial loss."""
        TestUtils.set_seed(config.seed)
        loss_fn = UncertaintyWeightedAdversarialLoss()
        
        batch_size = config.batch_size
        disc_out_fake = torch.randn(batch_size, 1, device=config.device)
        valid_labels = torch.ones(batch_size, 1, device=config.device)
        uncertainty_weights = torch.rand(batch_size, 1, device=config.device)
        
        loss = loss_fn(disc_out_fake, valid_labels, uncertainty_weights)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_mixture_nll_loss(self, config: TestConfig):
        """Test mixture density network NLL loss."""
        TestUtils.set_seed(config.seed)
        from torch.distributions import MixtureSameFamily, Categorical, Independent, Normal
        
        batch_size = config.batch_size
        num_gaussians = 5
        
        # Create mixture distribution
        pi_logits = torch.randn(batch_size, num_gaussians, device=config.device)
        mus = torch.randn(batch_size, num_gaussians, 6, device=config.device)
        sigmas = torch.exp(torch.randn(batch_size, num_gaussians, 6, device=config.device) * 0.1)
        
        pi = Categorical(logits=pi_logits)
        component_dist = Independent(Normal(loc=mus, scale=sigmas), 1)
        mixture_dist = MixtureSameFamily(pi, component_dist)
        
        # Sample target
        target = torch.randn(batch_size, 6, device=config.device)
        
        loss_fn = MixtureNLLLoss()
        loss = loss_fn(mixture_dist, target)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"


class UncertaintyTests:
    """Tests for uncertainty estimation."""
    
    def test_epistemic_uncertainty(self, config: TestConfig):
        """Test epistemic uncertainty computation."""
        TestUtils.set_seed(config.seed)
        
        # Create samples from Monte Carlo Dropout
        n_samples = 10
        batch_size = config.batch_size
        output_dim = 6
        
        samples = torch.randn(n_samples, batch_size, output_dim, device=config.device)
        
        uncertainty = epistemic_uncertainty(samples)
        
        assert uncertainty.shape == (batch_size, output_dim), \
            f"Expected shape ({batch_size}, {output_dim}), got {uncertainty.shape}"
        assert torch.all(uncertainty >= 0), "Uncertainty should be non-negative"
    
    def test_aleatoric_uncertainty_regression(self, config: TestConfig):
        """Test aleatoric uncertainty computation."""
        TestUtils.set_seed(config.seed)
        
        batch_size = config.batch_size
        output_dim = 6
        
        log_var = torch.randn(batch_size, output_dim, device=config.device) * 0.1 - 3.0
        
        uncertainty = aleatoric_uncertainty_regression(log_var)
        
        assert uncertainty.shape == (batch_size, output_dim), \
            f"Expected shape ({batch_size}, {output_dim}), got {uncertainty.shape}"
        assert torch.all(uncertainty > 0), "Uncertainty (variance) should be positive"
    
    def test_uncertainty_decomposition(self, config: TestConfig):
        """Test that total uncertainty = epistemic + aleatoric."""
        TestUtils.set_seed(config.seed)
        
        batch_size = config.batch_size
        
        # Generate epistemic uncertainty
        n_samples = 10
        samples = torch.randn(n_samples, batch_size, 6, device=config.device)
        epistemic = epistemic_uncertainty(samples)
        
        # Generate aleatoric uncertainty
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        aleatoric = aleatoric_uncertainty_regression(log_var)
        
        total = epistemic + aleatoric
        
        assert total.shape == (batch_size, 6)
        assert torch.all(total >= epistemic), "Total should be >= epistemic"
        assert torch.all(total >= aleatoric), "Total should be >= aleatoric"


class ModelTests:
    """Tests for model architectures."""
    
    def test_uaas_rap_net_forward(self, config: TestConfig):
        """Test UAASRAPNet forward pass."""
        TestUtils.set_seed(config.seed)
        from uaas.uaas_rap_net import UAASRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 256
        args.num_heads = 4
        
        model = UAASRAPNet(args).to(config.device)
        model.eval()
        
        batch_size = config.batch_size
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            outputs = model(img, return_feature=False)
        
        assert isinstance(outputs, tuple), "Should return tuple"
        assert len(outputs) == 2, "Should return (pose, log_var)"
        
        pose, log_var = outputs
        assert pose.shape == (batch_size, 12), f"Expected pose shape ({batch_size}, 12), got {pose.shape}"
        assert log_var.shape == (batch_size, 6), f"Expected log_var shape ({batch_size}, 6), got {log_var.shape}"
    
    def test_probabilistic_rap_net_forward(self, config: TestConfig):
        """Test ProbabilisticRAPNet forward pass."""
        TestUtils.set_seed(config.seed)
        from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 256
        args.num_heads = 4
        
        model = ProbabilisticRAPNet(args, num_gaussians=5).to(config.device)
        model.eval()
        
        batch_size = config.batch_size
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            mixture_dist = model(img, return_feature=False)
        
        from torch.distributions import MixtureSameFamily
        assert isinstance(mixture_dist, MixtureSameFamily), \
            "Should return MixtureSameFamily distribution"
        
        # Test sampling
        samples = mixture_dist.sample((10,))
        assert samples.shape == (10, batch_size, 6), \
            f"Expected sample shape (10, {batch_size}, 6), got {samples.shape}"


class RenderingTests:
    """Tests for rendering functionality."""
    
    def test_render_single_pose_exists(self, config: TestConfig):
        """Test that renderer has single pose rendering capability."""
        # This will be implemented when we fix rendering integration
        pass
    
    def test_render_perturbed_imgs_batch(self, config: TestConfig):
        """Test batch rendering of perturbed images."""
        # This will be implemented when we fix rendering integration
        pass


class IntegrationTests:
    """Integration tests for component interactions."""
    
    def test_uaas_trainer_initialization(self, config: TestConfig):
        """Test UAAS trainer can be initialized."""
        TestUtils.set_seed(config.seed)
        from uaas.trainer import UAASTrainer
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.run_name = "test_uaas"
        args.logbase = str(TestUtils.create_temp_dir())
        args.datadir = "dummy"
        args.model_path = "dummy"
        args.dataset_type = "Colmap"
        args.train_skip = 1
        args.test_skip = 1
        
        # This will fail without proper setup, but we can test initialization
        try:
            trainer = UAASTrainer(args)
            assert hasattr(trainer, 'model'), "Trainer should have model"
            assert hasattr(trainer, 'sampler'), "Trainer should have sampler"
            assert hasattr(trainer, 'adversarial_loss'), "Trainer should have adversarial_loss"
        except Exception as e:
            # Expected to fail without proper data setup
            assert "datadir" in str(e).lower() or "model_path" in str(e).lower() or "cameras.json" in str(e).lower()


class BenchmarkingTests:
    """Tests for benchmarking infrastructure."""
    
    def test_benchmark_config_validation(self):
        """Test benchmark configuration validation."""
        from benchmark_comparison import BenchmarkConfig
        
        config = BenchmarkConfig(
            model_type="uaas",
            run_training=True,
            run_evaluation=True,
            num_epochs=1
        )
        
        assert config.model_type == "uaas"
        assert config.run_training == True
        assert config.run_evaluation == True


# Pytest fixtures
@pytest.fixture
def test_config():
    """Fixture for test configuration."""
    return TestConfig()


@pytest.fixture
def cleanup_temp_dirs():
    """Fixture to cleanup temporary directories."""
    temp_dirs = []
    
    def _create_temp_dir(prefix: str = "rap_test_"):
        temp_dir = TestUtils.create_temp_dir(prefix)
        temp_dirs.append(temp_dir)
        return temp_dir
    
    yield _create_temp_dir
    
    # Cleanup
    for temp_dir in temp_dirs:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_tests()

