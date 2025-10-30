# Performance Comparison Guide

## Current Optimized Performance

Your optimized RAP implementation achieves:
- **Test Set Rendering**: 173.63 FPS (~5.76 ms/image)
- **Train Set Rendering**: 173.19 FPS (~5.77 ms/image)
- **Standard Test (800x800)**: 140.60 FPS (~7.11 ms/image)
- **GPU Memory**: 9.14 GB allocated
- **Model**: 161,643 Gaussians on NVIDIA H100 PCIe

## How to Measure Speed Boost

### Quick Comparison

If you have baseline results from the original repo:

```bash
python compare_performance.py --baseline /path/to/baseline_results.json
```

### Full Automated Comparison

Run the automated comparison tool:

```bash
cd /home/ubuntu/RAP
python compare_performance.py
```

This will:
1. Clone the original [ai4ce/RAP](https://github.com/ai4ce/RAP) repository
2. Set up its environment  
3. Run benchmark on the original version
4. Compare with your optimized version
5. Generate a detailed comparison report

### Manual Benchmark

If you want to benchmark manually:

**Original Repo:**
```bash
cd /home/ubuntu/RAP_original
source venv/bin/activate
python benchmark_speed.py -s data/Cambridge/KingsCollege/colmap \
  -m output/Cambridge/KingsCollege --iteration 30000 \
  --output baseline_output
```

**Then Compare:**
```bash
cd /home/ubuntu/RAP
python compare_performance.py --baseline /home/ubuntu/RAP_original/baseline_output/benchmark_results.json
```

## Expected Speedup

Based on your optimization commits:
- ✅ Batch RVS rendering for 2-5x performance improvement
- ✅ Comprehensive performance optimizations

**Expected**: 2-5x speedup compared to original repo

## Benchmark Results Files

- Optimized results: `output/Cambridge/KingsCollege/benchmark_results.json`
- Comparison report: `performance_comparison.txt` (generated after comparison)
