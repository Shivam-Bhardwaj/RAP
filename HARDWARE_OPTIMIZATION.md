# Hardware Optimization Guide

## Current Hardware
- **GPU**: NVIDIA H100 PCIe (85GB VRAM, 114 SMs)
- **CPU**: 18 cores
- **RAM**: 136GB

## Current Bottlenecks

### 1. Batch Size (MAJOR BOTTLENECK)
- **Current**: 4 samples/batch
- **Problem**: H100 can handle 32-64+ samples easily
- **Impact**: Only ~20% GPU utilization
- **Solution**: Increase to 32-64 based on memory

### 2. Data Loading (MAJOR BOTTLENECK)
- **Current**: `num_workers=0` (single-threaded)
- **Problem**: CPU idle while GPU waits for data
- **Impact**: GPU starved, slow training
- **Solution**: Use 12-16 workers for parallel data loading

### 3. Memory Transfer
- **Current**: No `pin_memory`
- **Problem**: Slower CPU→GPU transfer
- **Solution**: Enable `pin_memory=True`

### 4. Mixed Precision
- **Current**: Disabled by default in some places
- **Problem**: Not using Tensor Cores on H100
- **Solution**: Enable AMP (Automatic Mixed Precision)

## Optimized Configuration

### Quick Start (Optimized Pipeline)
```bash
# Source optimized version
source pipeline_steps_optimized.sh

# Check optimized config
show_config_opt

# Run with optimized settings
step_one_opt    # GS training
step_two_opt    # RAP training (32 batch, 16 workers, AMP)
step_three_opt  # Benchmark
step_four_opt   # Robustness test
```

### Manual Optimization

#### 1. Increase Batch Sizes
```bash
export BATCH_SIZE=32        # Training (was 4)
export VAL_BATCH_SIZE=32    # Validation (was 4)
```

For even more aggressive (if memory allows):
```bash
export BATCH_SIZE=64        # Training
export VAL_BATCH_SIZE=64    # Validation
```

#### 2. Enable Data Loading Parallelism
The code has been updated to automatically use:
- `num_workers=16` (parallel data loading)
- `pin_memory=True` (faster GPU transfer)
- `persistent_workers=True` (reuse worker processes)

#### 3. Enable Mixed Precision
```bash
export AMP=true
# Then in training scripts, use --amp flag
```

#### 4. Monitor GPU Utilization
While training, monitor with:
```bash
watch -n 1 nvidia-smi
```

Target metrics:
- **GPU Utilization**: >90%
- **GPU Memory**: 60-80% (not 100%, need headroom)
- **Temperature**: <85°C

## Expected Speedups

| Optimization | Expected Speedup |
|-------------|------------------|
| Batch size 4→32 | **4-6x faster** training |
| num_workers 0→16 | **2-3x faster** data loading |
| pin_memory | **10-20% faster** data transfer |
| AMP enabled | **1.5-2x faster** training |
| **Combined** | **10-15x faster** overall |

## Code Changes Made

1. **benchmark_full_pipeline.py**: 
   - Automatic `num_workers` based on CPU cores
   - `pin_memory=True` enabled
   - `persistent_workers=True` for efficiency

2. **test_dynamic_scene_robustness.py**:
   - Same optimizations applied

3. **pipeline_steps_optimized.sh**:
   - Pre-configured with optimal settings
   - Easy-to-use commands

## Fine-Tuning

### If Out of Memory (OOM)
Reduce batch size incrementally:
```bash
export BATCH_SIZE=16  # Try 16
export BATCH_SIZE=8   # Or 8 if still OOM
```

### If CPU Overloaded
Reduce workers:
```bash
export NUM_WORKERS=8   # Instead of 16
```

### Optimal Settings for H100
For this specific hardware:
```bash
export BATCH_SIZE=32
export VAL_BATCH_SIZE=32
export NUM_WORKERS=16
export AMP=true
```

These should give you:
- **GPU**: 90-95% utilization
- **GPU Memory**: 60-70% usage
- **Training Speed**: 10-15x faster than defaults

## Verification

After running with optimized settings, check:
```bash
# During training, in another terminal:
nvidia-smi

# Should show:
# - GPU Utilization: >90%
# - Memory Usage: 50-70GB (out of 85GB)
# - Power: High (not idle)
```

## Notes

- Batch size 32 is conservative for H100 - you can likely go higher
- num_workers=16 is optimal for 18 CPU cores
- Monitor GPU memory - if you see <50% usage, increase batch size
- If you see OOM errors, reduce batch size

