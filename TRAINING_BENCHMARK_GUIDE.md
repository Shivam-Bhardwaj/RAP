# Training Performance Benchmark Guide

## Overview

The `benchmark_training.py` script measures **actual training loop performance**, which is where the real optimizations show their impact. This is different from rendering benchmarks - it measures the full training iteration time including forward pass, backward pass, and optimizer steps.

## Quick Start

### Benchmark Current (Optimized) Version

```bash
cd /home/ubuntu/RAP
source venv/bin/activate
python benchmark_training.py \
  -s data/Cambridge/KingsCollege/colmap \
  -m output/Cambridge/KingsCollege \
  --num_iterations 1000 \
  --output training_benchmark_optimized
```

### Benchmark Baseline (Original) Version

```bash
cd /home/ubuntu/RAP_original
source venv/bin/activate
python /home/ubuntu/RAP/benchmark_training.py \
  -s /home/ubuntu/RAP/data/Cambridge/KingsCollege/colmap \
  -m /home/ubuntu/RAP/output/Cambridge/KingsCollege \
  --num_iterations 1000 \
  --output training_benchmark_baseline
```

### Compare Results

```bash
cd /home/ubuntu/RAP
python benchmark_training.py --compare \
  /home/ubuntu/RAP_original/training_benchmark_baseline/training_benchmark_results.json \
  training_benchmark_optimized/training_benchmark_results.json
```

## What It Measures

- **Time per iteration**: Average milliseconds per training iteration
- **Iterations per second**: Throughput metric
- **Time to convergence**: Estimated time for 30k iterations
- **Memory usage**: GPU memory allocation patterns
- **Iteration time distribution**: Min, max, median, p95, p99

## Expected Results

Based on your optimizations:
- **Batch RVS rendering**: 2-5x improvement expected
- **Non-blocking transfers**: 3.96x improvement expected
- **Data loading**: 2-4x improvement expected
- **Overall training**: 30-50% faster expected

## Output Files

- `training_benchmark_results.json`: Complete benchmark data
- Comparison output: Shows speedup, time saved, iterations per second

## Usage Tips

1. **Warmup**: First few iterations are slower due to CUDA initialization
2. **Iterations**: Use at least 1000 iterations for stable results
3. **Comparison**: Always compare on same hardware and same model checkpoint
4. **Memory**: Check memory usage to ensure no memory leaks

## Example Output

```
TRAINING TIME COMPARISON
======================================================================

Training Performance:
  Baseline:  45.23 ms/iter (22.11 iter/s)
  Optimized: 28.45 ms/iter (35.15 iter/s)
  Speedup:   1.59x (+59.1%)

Estimated Time to Convergence (30k iterations):
  Baseline:  37.69 hours
  Optimized: 23.71 hours
  Time Saved: 13.98 hours (37.1% faster)
```

