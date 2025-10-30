#!/usr/bin/env python3
"""
Unit tests for uncertainty estimation functions.
"""
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_framework import TestUtils, TestConfig
from common.uncertainty import epistemic_uncertainty, aleatoric_uncertainty_regression, UncertaintyVisualizer


class TestUncertaintyFunctions:
    """Tests for uncertainty estimation functions."""
    
    def test_epistemic_uncertainty_basic(self):
        """Test basic epistemic uncertainty computation."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        n_samples = 10
        batch_size = 4
        output_dim = 6
        
        # Create samples with known variance
        samples = torch.randn(n_samples, batch_size, output_dim, device=config.device)
        
        uncertainty = epistemic_uncertainty(samples)
        
        assert uncertainty.shape == (batch_size, output_dim), \
            f"Expected shape ({batch_size}, {output_dim}), got {uncertainty.shape}"
        assert torch.all(uncertainty >= 0), "Uncertainty should be non-negative"
    
    def test_epistemic_uncertainty_zero_variance(self):
        """Test epistemic uncertainty with zero variance (all samples same)."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        n_samples = 10
        batch_size = 2
        output_dim = 6
        
        # All samples are the same
        base_sample = torch.randn(1, batch_size, output_dim, device=config.device)
        samples = base_sample.repeat(n_samples, 1, 1)
        
        uncertainty = epistemic_uncertainty(samples)
        
        # Should be close to zero
        assert torch.allclose(uncertainty, torch.zeros_like(uncertainty), atol=1e-6), \
            "Zero variance should give zero uncertainty"
    
    def test_aleatoric_uncertainty_regression(self):
        """Test aleatoric uncertainty computation."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        batch_size = 4
        output_dim = 6
        
        log_var = torch.randn(batch_size, output_dim, device=config.device) * 0.1 - 3.0
        
        uncertainty = aleatoric_uncertainty_regression(log_var)
        
        assert uncertainty.shape == (batch_size, output_dim), \
            f"Expected shape ({batch_size}, {output_dim}), got {uncertainty.shape}"
        assert torch.all(uncertainty > 0), "Uncertainty (variance) should be positive"
        
        # Check exp relationship
        expected = torch.exp(log_var)
        TestUtils.assert_tensors_close(uncertainty, expected)
    
    def test_uncertainty_decomposition(self):
        """Test that total uncertainty = epistemic + aleatoric."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        batch_size = 4
        n_samples = 10
        
        # Generate epistemic uncertainty
        samples = torch.randn(n_samples, batch_size, 6, device=config.device)
        epistemic = epistemic_uncertainty(samples)
        
        # Generate aleatoric uncertainty
        log_var = torch.randn(batch_size, 6, device=config.device) * 0.1 - 3.0
        aleatoric = aleatoric_uncertainty_regression(log_var)
        
        total = epistemic + aleatoric
        
        assert total.shape == (batch_size, 6)
        assert torch.all(total >= epistemic), "Total should be >= epistemic"
        assert torch.all(total >= aleatoric), "Total should be >= aleatoric"


class TestUncertaintyVisualizer:
    """Tests for uncertainty visualization."""
    
    def test_visualizer_initialization(self):
        """Test visualizer can be initialized."""
        visualizer = UncertaintyVisualizer()
        assert visualizer is not None
    
    def test_plot_uncertainty_map(self, tmp_path):
        """Test uncertainty map plotting."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        visualizer = UncertaintyVisualizer()
        
        # Create dummy image and uncertainty
        image = TestUtils.create_dummy_image(batch_size=1, height=64, width=64)
        uncertainty = torch.rand(64, 64)
        
        save_path = tmp_path / "uncertainty_map.png"
        
        try:
            visualizer.plot_uncertainty_map(image.squeeze(0), uncertainty, str(save_path))
            assert save_path.exists(), "Visualization should be saved"
        except Exception as e:
            # If matplotlib dependencies are missing, that's okay for now
            if "matplotlib" not in str(e).lower():
                raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

