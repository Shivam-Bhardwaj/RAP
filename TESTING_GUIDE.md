# RAP-ID Testing & Benchmarking Framework

## Overview

This document describes the comprehensive testing and benchmarking framework for RAP-ID, designed to ensure publication-quality code and reproducible results.

## Testing Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_framework.py        # Testing utilities and fixtures
├── test_uncertainty.py      # Unit tests for uncertainty estimation
├── test_losses.py           # Unit tests for loss functions
├── test_models.py           # Unit tests for model architectures
├── test_integration.py      # Integration tests (to be created)
├── test_benchmarks.py        # Performance benchmarks (to be created)
└── test_rendering.py        # Rendering integration tests (to be created)
```

## Test Categories

### 1. Unit Tests

**Purpose:** Test individual components in isolation.

**Coverage:**
- Uncertainty estimation functions
- Loss functions (pose, adversarial, mixture)
- Model architectures (UAAS, Probabilistic, Semantic)
- Utility functions

**Run:**
```bash
python run_tests.py --unit
# or
pytest tests/test_uncertainty.py tests/test_losses.py tests/test_models.py -v
```

### 2. Integration Tests

**Purpose:** Test component interactions and workflows.

**Coverage:**
- Trainer initialization
- Training loop execution
- Component integration
- Data flow

**Run:**
```bash
python run_tests.py --integration
# or
pytest tests/test_integration.py -v
```

### 3. End-to-End Tests

**Purpose:** Test complete training/evaluation pipelines.

**Coverage:**
- Full training cycle
- Model checkpointing
- Evaluation metrics
- Inference pipeline

**Run:**
```bash
pytest tests/test_e2e.py -v
```

### 4. Performance Benchmarks

**Purpose:** Measure and track performance metrics.

**Coverage:**
- Training speed (iterations/sec)
- Inference speed (FPS)
- Memory usage
- GPU utilization

**Run:**
```bash
python run_tests.py --benchmarks
# or
pytest tests/test_benchmarks.py --benchmark-only
```

### 5. Regression Tests

**Purpose:** Prevent regressions in functionality.

**Coverage:**
- Known working configurations
- Previously fixed bugs
- Performance baselines

**Run:**
```bash
pytest tests/test_regression.py -v
```

## Test Execution

### Quick Test Run
```bash
python run_tests.py --unit
```

### Full Test Suite
```bash
python run_tests.py --all
```

### With Coverage
```bash
pytest tests/ --cov=. --cov-report=html
```

### Specific Test File
```bash
pytest tests/test_losses.py -v
```

### Specific Test Function
```bash
pytest tests/test_losses.py::TestCameraPoseLoss::test_basic_loss -v
```

## Test Utilities

### TestConfig
Provides standardized test configuration:
- Device selection (CPU/CUDA)
- Batch sizes
- Tolerance settings
- Random seeds

### TestUtils
Helper functions for test setup:
- `create_dummy_pose()` - Generate test poses
- `create_dummy_image()` - Generate test images
- `create_dummy_features()` - Generate test features
- `assert_tensors_close()` - Tensor comparison
- `create_temp_dir()` - Temporary directories

## Benchmarking

### Benchmark Categories

1. **Training Performance**
   - Time per iteration
   - Iterations per second
   - GPU memory usage
   - Convergence speed

2. **Inference Performance**
   - Latency (ms)
   - Throughput (FPS)
   - Batch processing speed
   - Memory footprint

3. **Accuracy Metrics**
   - Translation error
   - Rotation error
   - Success rates
   - Uncertainty calibration

4. **Component Performance**
   - Rendering speed
   - Uncertainty computation
   - Loss computation
   - Feature extraction

### Benchmark Execution

```bash
# Run all benchmarks
python run_tests.py --benchmarks

# Run specific benchmark
pytest tests/test_benchmarks.py::test_training_speed -v --benchmark-only

# Compare with baseline
python benchmark_comparison.py --models baseline uaas
```

## Continuous Integration

### GitHub Actions (TODO)

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=. --cov-report=xml
      - run: python run_tests.py --all
```

## Code Coverage

Target: **>80% coverage** for critical components

Current coverage can be checked with:
```bash
pytest tests/ --cov=. --cov-report=term-missing
```

## Test Data

### Synthetic Data
- Dummy poses (identity + small perturbations)
- Dummy images (random noise with ImageNet normalization)
- Dummy features (random tensors)

### Real Data (Optional)
- Small subset of actual datasets
- Mocked dataset loaders
- Cached test results

## Best Practices

1. **Isolation:** Each test should be independent
2. **Reproducibility:** Use fixed random seeds
3. **Clear Assertions:** Explicit error messages
4. **Fast Execution:** Unit tests should complete quickly
5. **Proper Cleanup:** Temporary files/directories

## Debugging Tests

```bash
# Run with verbose output
pytest tests/ -v -s

# Run with debugging
pytest tests/ --pdb

# Show print statements
pytest tests/ -s

# Run specific test with debugging
pytest tests/test_losses.py::TestCameraPoseLoss::test_basic_loss -v -s --pdb
```

## Writing New Tests

### Template

```python
import torch
import pytest
from tests.test_framework import TestUtils, TestConfig

class TestNewComponent:
    """Tests for new component."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        config = TestConfig()
        TestUtils.set_seed(config.seed)
        
        # Setup
        # ...
        
        # Execute
        # ...
        
        # Assert
        assert condition, "Error message"
```

## Future Enhancements

1. **Property-based Testing:** Use Hypothesis for property tests
2. **Fuzzing:** Random input generation
3. **Mutation Testing:** Verify test quality
4. **Visual Regression:** Image comparison tests
5. **Performance Profiling:** Detailed profiling integration

## References

- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Continuous Integration Best Practices](https://martinfowler.com/articles/continuousIntegration.html)

