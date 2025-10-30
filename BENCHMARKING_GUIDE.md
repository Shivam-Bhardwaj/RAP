# Benchmarking Guide for RAP-ID

This guide explains how to benchmark the different RAP-ID models.

## Overview

RAP-ID provides two benchmarking scripts:

1. **benchmark_speed.py** - Measures rendering speed and GPU memory usage
2. **benchmark_comprehensive.py** - Comprehensive pose accuracy and inference speed evaluation

## Benchmarking Rendering Speed

To benchmark 3DGS rendering speed:

```bash
python benchmark_speed.py --model_path /path/to/3dgs/model --iteration 30000
```

Options:
- `--model_path`: Path to 3DGS model directory
- `--iteration`: Checkpoint iteration to load (default: -1 for latest)
- `--num_iterations`: Number of images to benchmark (default: 100)
- `--output`: Output directory for results (default: model_path)

## Comprehensive Benchmarking

To benchmark pose accuracy and inference speed for trained models:

```bash
# Benchmark UAAS model
python benchmark_comprehensive.py \
    --model_type uaas \
    --config configs/7scenes.txt \
    --checkpoint_path /path/to/checkpoint.pth \
    --benchmark_speed

# Benchmark Probabilistic model
python benchmark_comprehensive.py \
    --model_type probabilistic \
    --config configs/7scenes.txt \
    --checkpoint_path /path/to/checkpoint.pth \
    --benchmark_speed

# Benchmark Semantic model
python benchmark_comprehensive.py \
    --model_type semantic \
    --config configs/7scenes.txt \
    --checkpoint_path /path/to/checkpoint.pth \
    --num_semantic_classes 19 \
    --benchmark_speed

# Benchmark baseline RAP
python benchmark_comprehensive.py \
    --model_type baseline \
    --config configs/7scenes.txt \
    --checkpoint_path /path/to/checkpoint.pth \
    --benchmark_speed
```

## Output

The comprehensive benchmark generates JSON files with:

- **Pose Accuracy Metrics:**
  - Median/Mean translation and rotation errors
  - Success rates (5cm/5deg and 2cm/2deg thresholds)
  - Min/Max errors

- **Uncertainty Metrics (UAAS only):**
  - Mean and standard deviation of uncertainty estimates

- **Inference Speed Metrics:**
  - FPS (frames per second)
  - Mean/std/min/max inference times

- **Model-Specific Metrics:**
  - Probabilistic: Number of hypotheses
  - Semantic: Semantic class statistics

## Comparing Results

To compare results across different models:

```bash
python -c "
import json
baseline = json.load(open('benchmark_baseline_7Scenes.json'))
uaas = json.load(open('benchmark_uaas_7Scenes.json'))
print('Baseline median translation:', baseline['accuracy']['median_translation'])
print('UAAS median translation:', uaas['accuracy']['median_translation'])
"
```

## Notes

- Ensure you have trained models before benchmarking pose accuracy
- Untrained models will produce poor results
- For probabilistic models, pose format conversion may need implementation
- GPU benchmarking requires CUDA-compatible hardware

