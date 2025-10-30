# Benchmarking Guide for RAP-ID

This guide explains how to benchmark the different RAP-ID models for both inference and training, including parallel benchmarking and comparison with the original RAP implementation.

## Overview

RAP-ID provides several benchmarking scripts:

1. **benchmark_speed.py** - Measures 3DGS rendering speed and GPU memory usage
2. **benchmark_comprehensive.py** - Comprehensive pose accuracy and inference speed evaluation
3. **benchmark_training.py** - Measures 3DGS training speed
4. **benchmark_training_rap.py** - Measures RAP model training performance
5. **benchmark_comparison.py** - Parallel benchmarking suite comparing all RAP-ID models
6. **benchmark_vs_original.py** - Compare RAP-ID against original RAP implementation

## Benchmarking 3DGS Rendering Speed

To benchmark 3DGS rendering speed:

```bash
python benchmark_speed.py --model_path /path/to/3dgs/model --iteration 30000
```

Options:
- `--model_path`: Path to 3DGS model directory
- `--iteration`: Checkpoint iteration to load (default: -1 for latest)
- `--num_iterations`: Number of images to benchmark (default: 100)
- `--output`: Output directory for results (default: model_path)

## Benchmarking 3DGS Training Speed

To benchmark 3DGS training performance:

```bash
python benchmark_training.py --model_path /path/to/3dgs/model --num_iterations 1000
```

This measures:
- Time per training iteration
- Iterations per second
- GPU memory usage
- Estimated time to convergence (30k iterations)

## Benchmarking RAP Model Training Performance

To benchmark RAP model training (including new models):

```bash
# Benchmark UAAS training
python benchmark_training_rap.py \
    --model_type uaas \
    --config configs/7scenes.txt \
    --benchmark_epochs 1 \
    --benchmark_batches 100

# Benchmark Probabilistic training
python benchmark_training_rap.py \
    --model_type probabilistic \
    --config configs/7scenes.txt \
    --benchmark_epochs 1

# Benchmark Semantic training
python benchmark_training_rap.py \
    --model_type semantic \
    --config configs/7scenes.txt \
    --num_semantic_classes 19 \
    --benchmark_epochs 1

# Benchmark baseline RAP training
python benchmark_training_rap.py \
    --model_type baseline \
    --config configs/7scenes.txt \
    --benchmark_epochs 1
```

Options:
- `--model_type`: Type of model ('baseline', 'uaas', 'probabilistic', 'semantic')
- `--benchmark_epochs`: Number of epochs to benchmark (default: 1)
- `--benchmark_batches`: Number of batches per epoch (default: all batches)
- `--output`: Output directory for results

## Comprehensive Inference Benchmarking

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

## Output Formats

### Training Benchmark Output

Training benchmarks generate JSON files with:

- **Training Performance Metrics:**
  - Total batches processed
  - Average/median batch time
  - Batches per second
  - Estimated time to convergence (30k iterations)

- **Memory Usage:**
  - Average and maximum GPU memory usage
  - Memory samples over time

- **Loss Statistics:**
  - Initial, final, and mean loss values
  - Loss standard deviation

- **Batch Time Statistics:**
  - Min/max/mean/median batch times
  - Percentiles (p95, p99)

### Inference Benchmark Output

Inference benchmarks generate JSON files with:

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

### Comparing Training Performance

To compare training performance across models:

```bash
python -c "
import json
baseline = json.load(open('training_benchmark_baseline.json'))
uaas = json.load(open('training_benchmark_uaas.json'))
print('Baseline avg batch time:', baseline['training']['avg_batch_time']*1000, 'ms')
print('UAAS avg batch time:', uaas['training']['avg_batch_time']*1000, 'ms')
print('Speedup:', baseline['training']['avg_batch_time'] / uaas['training']['avg_batch_time'], 'x')
"
```

### Comparing Inference Performance

To compare inference results:

```bash
python -c "
import json
baseline = json.load(open('benchmark_baseline_7Scenes.json'))
uaas = json.load(open('benchmark_uaas_7Scenes.json'))
print('Baseline median translation:', baseline['accuracy']['median_translation'])
print('UAAS median translation:', uaas['accuracy']['median_translation'])
print('UAAS mean uncertainty:', uaas['accuracy'].get('mean_uncertainty', 'N/A'))
"
```

## Performance Optimization Tips

1. **Training Speed:**
   - Enable `torch.compile` with `--compile_model` if supported
   - Use mixed precision training with `--amp`
   - Increase `num_workers` for data loading
   - Use `pin_memory=True` for faster GPU transfers

2. **Memory Usage:**
   - Reduce batch size if running out of memory
   - Use gradient checkpointing for large models
   - Monitor memory usage with training benchmarks

3. **Inference Speed:**
   - Use `torch.compile` for faster inference
   - Enable mixed precision with `--amp`
   - Batch inference when possible

## Notes

- Ensure you have trained models before benchmarking pose accuracy
- Untrained models will produce poor results
- For probabilistic models, pose format conversion may need implementation
- GPU benchmarking requires CUDA-compatible hardware
- Training benchmarks use a subset of data - results may vary with full training

