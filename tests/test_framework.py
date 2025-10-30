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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
