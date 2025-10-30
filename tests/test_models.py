#!/usr/bin/env python3
"""
Unit tests for model architectures.
"""
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig


class TestUAASRAPNet:
    """Tests for UAASRAPNet."""
    
    def test_model_initialization(self):
        """Test UAASRAPNet can be initialized."""
        from uaas.uaas_rap_net import UAASRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.hidden_dim = 256
        args.num_heads = 4
        args.dropout = 0.1
        
        model = UAASRAPNet(args)
        assert model is not None
    
    def test_forward_pass(self):
        """Test UAASRAPNet forward pass."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from uaas.uaas_rap_net import UAASRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128  # Smaller for testing
        args.num_heads = 4
        args.dropout = 0.1
        
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
    
    def test_forward_with_features(self):
        """Test UAASRAPNet forward pass with features."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from uaas.uaas_rap_net import UAASRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128
        args.num_heads = 4
        args.dropout = 0.1
        
        model = UAASRAPNet(args).to(config.device)
        model.eval()
        
        batch_size = 2
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            outputs = model(img, return_feature=True)
        
        assert isinstance(outputs, tuple), "Should return tuple"
        assert len(outputs) == 2, "Should return (features_and_pose, log_var)"
        
        (features, pose), log_var = outputs
        assert pose.shape == (batch_size, 12), f"Expected pose shape ({batch_size}, 12), got {pose.shape}"
        assert log_var.shape == (batch_size, 6), f"Expected log_var shape ({batch_size}, 6), got {log_var.shape}"


class TestProbabilisticRAPNet:
    """Tests for ProbabilisticRAPNet."""
    
    def test_model_initialization(self):
        """Test ProbabilisticRAPNet can be initialized."""
        from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.hidden_dim = 256
        args.num_heads = 4
        
        model = ProbabilisticRAPNet(args, num_gaussians=5)
        assert model is not None
    
    def test_forward_pass(self):
        """Test ProbabilisticRAPNet forward pass."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
        from arguments.options import config_parser
        from torch.distributions import MixtureSameFamily
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128
        args.num_heads = 4
        
        model = ProbabilisticRAPNet(args, num_gaussians=5).to(config.device)
        model.eval()
        
        batch_size = config.batch_size
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            mixture_dist = model(img, return_feature=False)
        
        assert isinstance(mixture_dist, MixtureSameFamily), \
            "Should return MixtureSameFamily distribution"
        
        # Test sampling
        samples = mixture_dist.sample((10,))
        assert samples.shape == (10, batch_size, 6), \
            f"Expected sample shape (10, {batch_size}, 6), got {samples.shape}"
        
        # Test log probability
        target = torch.randn(batch_size, 6, device=config.device)
        log_prob = mixture_dist.log_prob(target)
        assert log_prob.shape == (batch_size,), \
            f"Expected log_prob shape ({batch_size},), got {log_prob.shape}"


class TestSemanticRAPNet:
    """Tests for SemanticRAPNet."""
    
    def test_model_initialization(self):
        """Test SemanticRAPNet can be initialized."""
        from semantic.semantic_rap_net import SemanticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.hidden_dim = 256
        args.num_heads = 4
        
        model = SemanticRAPNet(args, num_semantic_classes=19)
        assert model is not None
    
    def test_forward_pass(self):
        """Test SemanticRAPNet forward pass."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from semantic.semantic_rap_net import SemanticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128
        args.num_heads = 4
        
        model = SemanticRAPNet(args, num_semantic_classes=19).to(config.device)
        model.eval()
        
        batch_size = config.batch_size
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            outputs = model(img, return_feature=False)
        
        # Should work like baseline RAPNet
        assert isinstance(outputs, tuple), "Should return tuple"
        _, pose = outputs
        assert pose.shape == (batch_size, 12), f"Expected pose shape ({batch_size}, 12), got {pose.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

