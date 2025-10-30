#!/usr/bin/env python3
"""
Unit tests for loss functions.
"""
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig
from utils.pose_utils import CameraPoseLoss
from uaas.loss import UncertaintyWeightedAdversarialLoss
from probabilistic.loss import MixtureNLLLoss
from torch.distributions import MixtureSameFamily, Categorical, Independent, Normal


class TestCameraPoseLoss:
    """Tests for CameraPoseLoss."""
    
    def test_basic_loss(self):
        """Test basic camera pose loss."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        # Create a simple config-like object
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 4
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        
        loss = loss_fn(est_pose, gt_pose)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_loss_with_uncertainty(self):
        """Test camera pose loss with uncertainty."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 4
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        
        # Create reasonable log-variance (negative for small variance)
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        
        loss = loss_fn(est_pose, gt_pose, log_var=log_var)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_loss_gradient_flow(self):
        """Test that loss gradients flow properly."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 2
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        est_pose.requires_grad = True
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        
        loss = loss_fn(est_pose, gt_pose)
        loss.backward()
        
        assert est_pose.grad is not None, "Gradients should flow"
        assert not torch.isnan(est_pose.grad).any(), "Gradients should not be NaN"


class TestUncertaintyWeightedAdversarialLoss:
    """Tests for UncertaintyWeightedAdversarialLoss."""
    
    def test_basic_loss(self):
        """Test basic uncertainty-weighted adversarial loss."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        loss_fn = UncertaintyWeightedAdversarialLoss()
        
        batch_size = 4
        disc_out_fake = torch.randn(batch_size, 1, device=config.device)
        valid_labels = torch.ones(batch_size, 1, device=config.device)
        uncertainty_weights = torch.rand(batch_size, 1, device=config.device)
        
        loss = loss_fn(disc_out_fake, valid_labels, uncertainty_weights)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_loss_with_zero_uncertainty(self):
        """Test loss with zero uncertainty weights."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        loss_fn = UncertaintyWeightedAdversarialLoss()
        
        batch_size = 4
        disc_out_fake = torch.randn(batch_size, 1, device=config.device)
        valid_labels = torch.ones(batch_size, 1, device=config.device)
        uncertainty_weights = torch.zeros(batch_size, 1, device=config.device)
        
        loss = loss_fn(disc_out_fake, valid_labels, uncertainty_weights)
        
        # With zero weights, loss should be zero
        assert torch.allclose(loss, torch.tensor(0.0, device=config.device))


class TestMixtureNLLLoss:
    """Tests for MixtureNLLLoss."""
    
    def test_basic_loss(self):
        """Test basic mixture NLL loss."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        batch_size = 4
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
        assert not torch.isinf(loss), "Loss should not be Inf"
    
    def test_loss_gradient_flow(self):
        """Test that mixture loss gradients flow."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        batch_size = 2
        num_gaussians = 3
        
        # Create learnable parameters
        pi_logits = torch.randn(batch_size, num_gaussians, device=config.device, requires_grad=True)
        mus = torch.randn(batch_size, num_gaussians, 6, device=config.device, requires_grad=True)
        log_sigmas = torch.randn(batch_size, num_gaussians, 6, device=config.device, requires_grad=True)
        sigmas = torch.exp(log_sigmas)
        
        pi = Categorical(logits=pi_logits)
        component_dist = Independent(Normal(loc=mus, scale=sigmas), 1)
        mixture_dist = MixtureSameFamily(pi, component_dist)
        
        target = torch.randn(batch_size, 6, device=config.device)
        
        loss_fn = MixtureNLLLoss()
        loss = loss_fn(mixture_dist, target)
        loss.backward()
        
        assert pi_logits.grad is not None, "Gradients should flow to pi_logits"
        assert mus.grad is not None, "Gradients should flow to mus"
        assert log_sigmas.grad is not None, "Gradients should flow to log_sigmas"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

