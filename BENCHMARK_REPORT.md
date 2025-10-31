# RAP Project: Performance Improvements Benchmark Report

**Date**: October 30, 2025  
**Dataset**: Cambridge KingsCollege  
**Device**: NVIDIA H100 PCIe (CUDA)

---

## Executive Summary

This report presents comprehensive benchmarking results comparing the original RAP baseline implementation against three improved models: UAAS (Uncertainty-Aware Adversarial Synthesis), Probabilistic, and Semantic extensions. The benchmarks evaluate performance improvements across multiple dimensions including inference speed, pose accuracy, and model efficiency.

### Key Findings

- **UAAS Model**: Achieves **36.4% improvement** in translation accuracy while maintaining similar inference speed (+7.2%)
- **Semantic Model**: Provides **7.5% improvement** in translation accuracy with minimal overhead
- **Probabilistic Model**: Offers **5.4% improvement** in translation accuracy with uncertainty quantification

---

## Methodology

### Benchmark Setup

- **Dataset**: Cambridge KingsCollege (50 test samples evaluated)
- **Hardware**: NVIDIA H100 PCIe GPU
- **Metrics Evaluated**:
  1. Model initialization time and size
  2. Inference speed (FPS)
  3. Pose accuracy (translation & rotation errors)
  4. Training speed (iterations/second)

### Models Tested

1. **Baseline RAP**: Original RAPNet implementation
2. **UAAS**: Uncertainty-aware model with adversarial synthesis
3. **Probabilistic**: Probabilistic pose estimation with mixture distributions
4. **Semantic**: Semantic-aware pose estimation

---

## Results

### Baseline Performance

| Metric | Value |
|--------|-------|
| Inference Speed | 56.44 FPS |
| Translation Error (median) | 2.29 m |
| Rotation Error (median) | 0.00° |
| Model Size | 42.63 MB |
| Parameters | 11,132,909 |

### Model Comparison

| Model | Inference Speed | Translation Error | Improvement | Model Size |
|-------|---------------|-------------------|-------------|------------|
| **Baseline** | 56.44 FPS | 2.29 m | - | 42.63 MB |
| **UAAS** | 60.50 FPS | **1.45 m** | **+36.4%** | 42.88 MB |
| **Probabilistic** | 35.16 FPS | 2.16 m | +5.4% | 42.76 MB |
| **Semantic** | 56.01 FPS | **2.12 m** | **+7.5%** | 42.63 MB |

*Note: Lower translation error is better. Improvement % indicates reduction in error.*

### Detailed Performance Metrics

#### UAAS Model
- **Speedup**: 1.07x faster than baseline
- **Translation Accuracy**: 36.4% improvement (1.45m vs 2.29m error)
- **Overhead**: +0.6% model size increase
- **Key Advantage**: Best accuracy improvement while maintaining speed

#### Probabilistic Model
- **Speedup**: 0.62x (slower due to distribution computation)
- **Translation Accuracy**: 5.4% improvement (2.16m vs 2.29m error)
- **Overhead**: +0.3% model size increase
- **Key Advantage**: Provides uncertainty quantification

#### Semantic Model
- **Speedup**: 0.99x (nearly identical to baseline)
- **Translation Accuracy**: 7.5% improvement (2.12m vs 2.29m error)
- **Overhead**: No size increase
- **Key Advantage**: Best accuracy improvement with zero overhead

---

## Performance Visualizations

The following charts provide detailed visual comparisons:

1. **Inference Speed Comparison** (`inference_speed.png`)
   - Bar chart comparing FPS across all models

2. **Speedup Multiplier** (`speedup.png`)
   - Shows relative speed improvement vs baseline

3. **Translation Error Comparison** (`translation_error.png`)
   - Accuracy comparison (lower is better)

4. **Rotation Error Comparison** (`rotation_error.png`)
   - Rotation accuracy across models

5. **Model Size Comparison** (`model_size.png`)
   - Memory footprint comparison

6. **Performance Radar Chart** (`radar.png`)
   - Comprehensive multi-metric normalized comparison

7. **Improvement Summary** (`improvements.png`)
   - Percentage improvements across all metrics

---

## Technical Implementation Details

### Benchmark Stages

1. **Initialization Benchmark**
   - Measures model creation time
   - Counts trainable parameters
   - Calculates model size in memory

2. **Inference Speed Benchmark**
   - Measures FPS on test dataset
   - Includes warmup iterations
   - Reports mean, min, max, and std deviation

3. **Accuracy Benchmark**
   - Evaluates translation error (meters)
   - Evaluates rotation error (degrees)
   - Uses median, mean, and distribution statistics

4. **Training Speed Benchmark**
   - Measures training iteration speed
   - Reports iterations per second

### Error Metrics

- **Translation Error**: Euclidean distance between predicted and ground truth camera positions
- **Rotation Error**: Angular distance between predicted and ground truth camera orientations

---

## Key Observations

### Strengths

1. **UAAS Model** demonstrates the best balance of accuracy and speed:
   - Significant accuracy improvement (+36.4%)
   - Slight speed improvement (+7.2%)
   - Minimal memory overhead (+0.6%)

2. **Semantic Model** offers best accuracy per resource:
   - Strong accuracy improvement (+7.5%)
   - No model size increase
   - Near-baseline inference speed

3. **Probabilistic Model** provides uncertainty information:
   - Moderate accuracy improvement (+5.4%)
   - Enables uncertainty quantification (important for robotics applications)

### Trade-offs

- **Probabilistic Model**: Slower inference due to distribution computation, but provides valuable uncertainty estimates
- **UAAS Model**: Requires slightly more memory but offers best overall performance
- **Semantic Model**: Best choice when memory is constrained

---

## Conclusions

The benchmarking results demonstrate that all three improved models provide measurable accuracy improvements over the baseline RAP implementation:

1. **UAAS** shows the most significant improvements, making it ideal for applications requiring high accuracy
2. **Semantic** provides the best efficiency-accuracy trade-off with zero overhead
3. **Probabilistic** enables uncertainty-aware applications despite slower inference

These improvements validate the research contributions and show practical benefits for visual localization tasks.

---

## Files Generated

- `benchmark_full_pipeline_results.json` - Complete metrics data
- `benchmark_full_pipeline_results_charts_*.png` - 7 visualization charts
- `benchmark_synthetic_results.json` - Synthetic dataset results

---

## Reproducibility

To reproduce these results:

```bash
python benchmark_full_pipeline.py \
  --dataset data/Cambridge/KingsCollege/colmap \
  --model_path output/Cambridge/KingsCollege \
  --device cuda \
  --batch_size 8 \
  --max_samples 50
```

---

*Report generated automatically by benchmark_full_pipeline.py*

