#!/usr/bin/env python3
"""
Performance benchmarks for RAP-ID components.

Measures training speed, inference speed, and memory usage.
"""
import torch
import pytest
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig
import pytest_benchmark


class TestInferenceSpeed:
    """Benchmarks for inference speed."""
    
    def test_uaas_inference_speed(self, benchmark):
        """Benchmark UAAS model inference speed."""
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
        
        batch_size = 8
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            def run_inference():
                return model(img, return_feature=False)
            
            result = benchmark(run_inference)
            assert result is not None
    
    def test_probabilistic_inference_speed(self, benchmark):
        """Benchmark Probabilistic model inference speed."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = config.device
        args.hidden_dim = 128
        args.num_heads = 4
        
        model = ProbabilisticRAPNet(args, num_gaussians=5).to(config.device)
        model.eval()
        
        batch_size = 8
        img = TestUtils.create_dummy_image(batch_size, device=config.device)
        
        with torch.no_grad():
            def run_inference():
                return model(img, return_feature=False)
            
            result = benchmark(run_inference)
            assert result is not None


class TestMemoryUsage:
    """Benchmarks for memory usage."""
    
    def test_uaas_memory_usage(self):
        """Measure UAAS model memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from uaas.uaas_rap_net import UAASRAPNet
        from arguments.options import config_parser
        
        parser = config_parser()
        args = parser.parse_args([])
        args.device = "cuda"
        args.hidden_dim = 256
        args.num_heads = 4
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model = UAASRAPNet(args).to("cuda")
        
        model_memory = torch.cuda.memory_allocated() - initial_memory
        
        batch_size = 8
        img = TestUtils.create_dummy_image(batch_size, device="cuda")
        
        with torch.no_grad():
            _ = model(img, return_feature=False)
        
        peak_memory = torch.cuda.max_memory_allocated() - initial_memory
        
        print(f"\nUAAS Model Memory:")
        print(f"  Model size: {model_memory / 1e6:.2f} MB")
        print(f"  Peak memory: {peak_memory / 1e6:.2f} MB")
        
        assert model_memory > 0
        assert peak_memory >= model_memory


class TestLossComputationSpeed:
    """Benchmarks for loss computation speed."""
    
    def test_pose_loss_speed(self, benchmark):
        """Benchmark pose loss computation."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from utils.pose_utils import CameraPoseLoss
        
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 32
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        
        def compute_loss():
            return loss_fn(est_pose, gt_pose)
        
        result = benchmark(compute_loss)
        assert result is not None
    
    def test_uncertainty_loss_speed(self, benchmark):
        """Benchmark uncertainty-weighted loss computation."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        from utils.pose_utils import CameraPoseLoss
        
        class SimpleConfig:
            loss_learnable = False
            loss_norm = 2
            s_x = 1.0
            s_q = 1.0
        
        loss_fn = CameraPoseLoss(SimpleConfig())
        
        batch_size = 32
        est_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        gt_pose = TestUtils.create_dummy_pose(batch_size, config.device).reshape(batch_size, 12)
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        
        def compute_loss():
            return loss_fn(est_pose, gt_pose, log_var=log_var)
        
        result = benchmark(compute_loss)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])

