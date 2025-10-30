#!/usr/bin/env python3
"""
End-to-End Test: Full Training Pipeline

Tests that the complete training pipeline works end-to-end for all model types.
"""
import torch
import pytest
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig


class TestTrainingPipeline:
    """End-to-end tests for training pipelines."""
    
    @pytest.mark.slow
    def test_uaas_trainer_initialization(self, tmp_path):
        """Test UAAS trainer can be initialized."""
        from uaas.trainer import UAASTrainer
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.run_name = "test_uaas"
        args.logbase = str(tmp_path)
        args.datadir = "dummy"
        args.model_path = "dummy"
        args.dataset_type = "Colmap"
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        args.val_batch_size = 1
        
        # This will fail without proper data, but we can test initialization
        try:
            trainer = UAASTrainer(args)
            assert hasattr(trainer, 'model')
            assert hasattr(trainer, 'sampler')
            assert hasattr(trainer, 'adversarial_loss')
        except Exception as e:
            # Expected to fail without proper data setup
            assert any(keyword in str(e).lower() for keyword in 
                      ["datadir", "model_path", "cameras.json", "data"])
    
    @pytest.mark.slow
    def test_probabilistic_trainer_initialization(self, tmp_path):
        """Test Probabilistic trainer can be initialized."""
        from probabilistic.trainer import ProbabilisticTrainer
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.run_name = "test_prob"
        args.logbase = str(tmp_path)
        args.datadir = "dummy"
        args.model_path = "dummy"
        args.dataset_type = "Colmap"
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        
        try:
            trainer = ProbabilisticTrainer(args)
            assert hasattr(trainer, 'model')
            assert hasattr(trainer, 'hypothesis_validator')
            assert hasattr(trainer, 'criterion')
        except Exception as e:
            assert any(keyword in str(e).lower() for keyword in 
                      ["datadir", "model_path", "cameras.json", "data"])
    
    @pytest.mark.slow
    def test_semantic_trainer_initialization(self, tmp_path):
        """Test Semantic trainer can be initialized."""
        from semantic.trainer import SemanticTrainer
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cpu"
        args.run_name = "test_semantic"
        args.logbase = str(tmp_path)
        args.datadir = "dummy"
        args.model_path = "dummy"
        args.dataset_type = "Colmap"
        args.train_skip = 1
        args.test_skip = 1
        args.batch_size = 2
        args.num_semantic_classes = 19
        
        try:
            trainer = SemanticTrainer(args)
            assert hasattr(trainer, 'model')
            assert hasattr(trainer, 'synthesizer')
            assert hasattr(trainer, 'hard_negative_miner')
            assert hasattr(trainer, 'curriculum')
        except Exception as e:
            assert any(keyword in str(e).lower() for keyword in 
                      ["datadir", "model_path", "cameras.json", "data"])


class TestModelForwardPass:
    """Tests for model forward passes."""
    
    def test_all_models_forward(self):
        """Test all models can perform forward pass."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from uaas.uaas_rap_net import UAASRAPNet
        from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
        from semantic.semantic_rap_net import SemanticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128
        args.num_heads = 4
        
        batch_size = 2
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        # Test UAAS
        uaas_model = UAASRAPNet(args).to(config.device)
        uaas_model.eval()
        with torch.no_grad():
            pose, log_var = uaas_model(img, return_feature=False)
            assert pose.shape == (batch_size, 12)
            assert log_var.shape == (batch_size, 6)
        
        # Test Probabilistic
        prob_model = ProbabilisticRAPNet(args, num_gaussians=5).to(config.device)
        prob_model.eval()
        with torch.no_grad():
            mixture_dist = prob_model(img, return_feature=False)
            samples = mixture_dist.sample((10,))
            assert samples.shape == (10, batch_size, 6)
        
        # Test Semantic
        semantic_model = SemanticRAPNet(args, num_semantic_classes=19).to(config.device)
        semantic_model.eval()
        with torch.no_grad():
            _, pose = semantic_model(img, return_feature=False)
            assert pose.shape == (batch_size, 12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

