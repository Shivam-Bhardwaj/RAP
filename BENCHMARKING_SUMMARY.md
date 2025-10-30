# Benchmarking Summary: RAP-ID vs Original RAP

This document summarizes the benchmarking infrastructure for comparing RAP-ID extensions against the baseline and original RAP implementation.

## Benchmarking Infrastructure

### 1. Parallel Benchmarking Suite (`benchmark_comparison.py`)

**Purpose:** Run comprehensive benchmarks for all RAP-ID models in parallel, ideal for powerful machines.

**Features:**
- Parallel execution of training and evaluation benchmarks
- Automatic comparison generation
- JSON and text report outputs
- Support for all model types: baseline, UAAS, Probabilistic, Semantic

**Usage:**
```bash
python benchmark_comparison.py \
    --config configs/7scenes.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs \
    --models baseline uaas probabilistic semantic \
    --parallel \
    --max_workers 4 \
    --benchmark_epochs 1 \
    --output ./benchmark_results
```

**Output:**
- `results_<model_type>.json` - Individual model results
- `comparison_report.txt` - Human-readable comparison
- `benchmark_summary.json` - Complete results summary

### 2. Original RAP Comparison (`benchmark_vs_original.py`)

**Purpose:** Compare RAP-ID results against the original [ai4ce/RAP](https://github.com/ai4ce/RAP) implementation.

**Features:**
- Clone/setup original RAP repository
- Parse original RAP results
- Generate comparison reports
- Calculate improvement metrics

**Usage:**
```bash
# Setup original repo
python benchmark_vs_original.py --clone_original

# Compare results
python benchmark_vs_original.py \
    --compare \
    --rap_id_results ./benchmark_results/benchmark_summary.json \
    --original_results /path/to/original/results.json \
    --output ./comparison
```

## Expected Improvements

Based on the theoretical foundations and implementation, we expect:

### UAAS (Uncertainty-Aware Adversarial Synthesis)

**Training Performance:**
- Slight overhead: ~5-10% slower due to uncertainty computation
- Better convergence: Uncertainty-guided sampling improves training efficiency

**Evaluation Accuracy:**
- 5-15% reduction in translation error
- 5-15% reduction in rotation error
- Higher success rates (5cm, 5deg): +2-5 percentage points
- Calibrated uncertainty estimates enable better failure detection

**Inference Speed:**
- Minimal overhead: ~2-5% slower due to uncertainty head
- Uncertainty estimates available for downstream tasks

### Probabilistic (Multi-Hypothesis)

**Training Performance:**
- Moderate overhead: ~10-20% slower due to MDN forward pass
- Better handling of ambiguous scenes

**Evaluation Accuracy:**
- 10-20% reduction in error for ambiguous scenes
- Higher success rates when multiple hypotheses are validated
- Improved robustness to scene ambiguity

**Inference Speed:**
- Hypothesis validation adds overhead (when enabled)
- Can be configured for speed vs accuracy tradeoff

### Semantic (Semantic-Adversarial)

**Training Performance:**
- Overhead depends on semantic segmentation model
- Curriculum learning improves convergence

**Evaluation Accuracy:**
- 5-10% reduction in error for semantically challenging scenes
- Better robustness to appearance changes
- Improved generalization to new environments

**Inference Speed:**
- Minimal overhead if semantic features are optional
- Full semantic pipeline adds ~5-10% overhead

## Benchmarking Workflow

### Step 1: Prepare Data and Models

1. Train 3DGS model on your dataset:
```bash
python gs.py -s /path/to/colmap/data -m /path/to/output
```

2. Ensure you have trained RAP-ID models (or use untrained for training benchmarks)

### Step 2: Run RAP-ID Benchmarks

```bash
# Comprehensive parallel benchmark
python benchmark_comparison.py \
    --config configs/7scenes.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs \
    --models baseline uaas probabilistic semantic \
    --parallel \
    --output ./benchmark_results
```

### Step 3: Benchmark Original RAP (Optional)

1. Clone original repo:
```bash
python benchmark_vs_original.py --clone_original
```

2. Run original RAP evaluation:
```bash
cd ~/RAP_original
python rap.py -c configs/7scenes.txt -m /path/to/3dgs
# Save results to JSON format
```

3. Compare:
```bash
python benchmark_vs_original.py \
    --compare \
    --rap_id_results ./benchmark_results/benchmark_summary.json \
    --original_results /path/to/original/results.json \
    --output ./comparison
```

## Benchmark Metrics

### Training Metrics

- **Batch Time:** Average time per training batch (ms)
- **Throughput:** Batches per second
- **Memory Usage:** GPU memory allocated (GB)
- **Estimated 30k Iter Time:** Estimated time for full training (hours)

### Evaluation Metrics

- **Translation Error:** Median and mean translation error (meters)
- **Rotation Error:** Median and mean rotation error (degrees)
- **Success Rates:**
  - 5cm, 5deg: Percentage of poses within 5cm translation and 5deg rotation
  - 2cm, 2deg: Percentage of poses within 2cm translation and 2deg rotation

### Inference Metrics

- **FPS:** Frames per second
- **Latency:** Mean inference time (ms)
- **Throughput:** Images processed per second

### Model-Specific Metrics

- **UAAS:** Mean uncertainty, uncertainty calibration
- **Probabilistic:** Number of hypotheses, hypothesis diversity
- **Semantic:** Semantic class coverage, curriculum difficulty

## Interpreting Results

### Positive Improvements

- **Translation/Rotation Error Reduction:** Lower is better
- **Success Rate Improvement:** Higher is better
- **Training Throughput:** Higher is better (but accuracy more important)
- **Inference Speed:** Higher FPS is better (with comparable accuracy)

### Trade-offs

- **Speed vs Accuracy:** Some extensions trade speed for accuracy
- **Memory vs Performance:** Larger models use more memory
- **Training Time vs Accuracy:** Longer training may improve accuracy

### Statistical Significance

For rigorous comparison:
- Run multiple trials (use different seeds)
- Report mean ± standard deviation
- Perform statistical tests (t-test, Mann-Whitney U)
- Consider confidence intervals

## Troubleshooting

### Out of Memory

- Reduce batch size: `--batch_size 4`
- Reduce workers: `--max_workers 2`
- Disable model compilation: `--compile_model False`

### Benchmarks Hang

- Check GPU availability: `nvidia-smi`
- Reduce parallel workers: `--max_workers 1`
- Use `--training_only` or `--evaluation_only` to isolate issues

### Missing Checkpoints

- Benchmarks will warn if checkpoints not found
- Training benchmarks can run without checkpoints
- Evaluation benchmarks require trained models

### Original RAP Comparison Issues

- Ensure original repo is properly cloned with submodules
- Check that original RAP results are in correct format
- Manually verify original RAP results before comparison

## Next Steps

1. **Run Initial Benchmarks:** Use parallel suite on your hardware
2. **Train Models:** Train all extensions on your dataset
3. **Full Evaluation:** Run comprehensive benchmarks with trained models
4. **Compare:** Generate comparison reports
5. **Iterate:** Use results to guide further improvements

## References

- Original RAP: https://github.com/ai4ce/RAP
- RAP-ID Repository: https://github.com/Shivam-Bhardwaj/RAP
- Technical Documentation: See `TECHNICAL_DOCUMENTATION.md` for mathematical details

