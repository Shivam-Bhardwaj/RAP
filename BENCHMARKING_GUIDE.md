# Benchmarking Guide

This guide explains how to run the benchmarking scripts to evaluate model performance.

## Available Benchmark Scripts

1. **`benchmark_full_pipeline.py`** - Comprehensive full pipeline benchmark
2. **`benchmark_synthetic.py`** - Benchmark on synthetic data

## Prerequisites

- GPU with CUDA (recommended) or CPU
- Trained model checkpoints (optional, for accuracy evaluation)
- Dataset in Colmap format

## 1. Full Pipeline Benchmark

Runs complete benchmarking comparing baseline RAP vs UAAS/Probabilistic/Semantic models.

### Basic Usage

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --device cuda \
    --batch_size 8 \
    --max_samples 50
```

### Arguments

- `--dataset`: Path to dataset (Colmap format)
- `--model_path`: Path to Gaussian Splatting model directory
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda` if available)
- `--batch_size`: Batch size for evaluation (default: 4)
- `--max_samples`: Limit number of test samples (optional, for faster testing)
- `--checkpoint_dir`: Directory with model checkpoints (optional)

### Example: Cambridge KingsCollege

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --device cuda \
    --batch_size 8 \
    --max_samples 50
```

### Example: 7Scenes Office

```bash
python benchmark_full_pipeline.py \
    --dataset data/7Scenes/office \
    --model_path output/7Scenes/office \
    --device cuda \
    --batch_size 4
```

### Output Files

After running, you'll get:

- `benchmark_full_pipeline_results.json` - Complete metrics data
- `benchmark_full_pipeline_results_charts_*.png` - 7 visualization charts:
  - `inference_speed.png` - FPS comparison
  - `speedup.png` - Speed improvement multiplier
  - `translation_error.png` - Translation accuracy
  - `rotation_error.png` - Rotation accuracy
  - `model_size.png` - Memory footprint
  - `radar.png` - Comprehensive performance radar
  - `improvements.png` - Improvement percentages

## 2. Synthetic Data Benchmark

Benchmark models on synthetic datasets for quick testing.

```bash
python benchmark_synthetic.py \
    --dataset data/synthetic_test_dataset \
    --model_path output/synthetic_test_dataset \
    --device cuda \
    --batch_size 4 \
    --max_samples 100
```

### Arguments

- `--dataset`: Path to synthetic dataset
- `--model_path`: Path to GS model directory
- `--device`: Device (`cuda` or `cpu`)
- `--batch_size`: Batch size (default: 4)
- `--max_samples`: Limit test samples (optional)

## Quick Testing (CPU-only)

For quick testing without GPU:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --device cpu \
    --batch_size 2 \
    --max_samples 10
```

Note: CPU is much slower - use `--max_samples` to limit evaluation time.

## Troubleshooting

### Missing GS Checkpoints

If you see warnings about missing Gaussian Splatting checkpoints:

- Training speed benchmarks will be skipped (this is expected)
- Accuracy and inference benchmarks will still run
- Model checkpoints are optional for benchmarking

### CUDA Out of Memory

Reduce batch size:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --device cuda \
    --batch_size 2  # Reduced from 8
```

### Slow Execution

- Use `--max_samples` to limit evaluation
- Use `--device cpu` if GPU is unavailable (much slower)
- Reduce `--batch_size` if memory constrained

## Viewing Results

### JSON Results

```bash
python -c "import json; f=open('benchmark_full_pipeline_results.json'); d=json.load(f); print(json.dumps(d, indent=2))"
```

### Charts

Charts are saved as PNG files. Open them directly:

```bash
# View all charts
ls -lh benchmark_full_pipeline_results_charts*.png

# Or open in image viewer
xdg-open benchmark_full_pipeline_results_charts_radar.png
```

## Expected Runtime

- **Full benchmark (50 samples, GPU)**: ~5-10 minutes
- **Full benchmark (50 samples, CPU)**: ~30-60 minutes
- **Synthetic benchmark (100 samples, GPU)**: ~2-5 minutes

## Reproducing Results from README

To reproduce the exact results shown in the README:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --device cuda \
    --batch_size 4 \
    --max_samples 50
```

This will generate the same results and charts shown in the README case study.
