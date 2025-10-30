# Next Steps: RAP-ID Development Roadmap

## Current Status

✅ **Completed:**
- All three extensions implemented (UAAS, Probabilistic, Semantic)
- Comprehensive benchmarking infrastructure
- Mathematical documentation (PhD-level)
- Parallel benchmarking suite
- Comparison tools vs original RAP

## Immediate Next Steps

### 1. **Implement Placeholder Functions**

Several modules have `NotImplementedError` stubs that need implementation:

**Priority: Critical (for training)**
- `RAP/uaas/sampler.py` - `UncertaintySampler.sample()` - Generates training samples in high-uncertainty regions
- `RAP/probabilistic/hypothesis_validator.py` - `HypothesisValidator.validate()` - Validates pose hypotheses via rendering
- `RAP/semantic/semantic_synthesizer.py` - `SemanticSynthesizer.synthesize()` - Semantic-aware scene manipulation
- `RAP/semantic/hard_negative_miner.py` - `HardNegativeMiner.mine()` - Adversarial hard negative mining

**Priority: Optional (for visualization)**
- `RAP/common/uncertainty.py` - `UncertaintyVisualizer.plot_uncertainty_map()` - Uncertainty visualization

### 2. **Train Models**

Train the baseline and extended models on your dataset:

```bash
# Train baseline (using original rap.py)
python rap.py -c configs/aachen.txt -m /path/to/3dgs/model -n baseline_exp

# Train UAAS model
python train.py -c configs/aachen.txt -m /path/to/3dgs/model --trainer_type uaas -n uaas_exp

# Train Probabilistic model
python train.py -c configs/aachen.txt -m /path/to/3dgs/model --trainer_type probabilistic -n prob_exp

# Train Semantic model
python train.py -c configs/aachen.txt -m /path/to/3dgs/model --trainer_type semantic -n semantic_exp --num_semantic_classes 19
```

**Note:** You'll need to implement the placeholder functions first, or modify trainers to skip those components initially.

### 3. **Run Benchmarks**

Once models are trained:

```bash
# Run comprehensive parallel benchmarks
python benchmark_comparison.py \
    --config configs/aachen.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs/model \
    --models baseline uaas probabilistic semantic \
    --parallel \
    --checkpoint_path /path/to/checkpoint.pth \
    --output ./benchmark_results
```

### 4. **Compare Against Original RAP**

```bash
# Clone original repo
python benchmark_vs_original.py --clone_original

# Run original RAP evaluation (manual)
cd ~/RAP_original
python rap.py -c configs/aachen.txt -m /path/to/3dgs/model

# Compare results
python benchmark_vs_original.py \
    --compare \
    --rap_id_results ./benchmark_results/benchmark_summary.json \
    --original_results /path/to/original/results.json
```

## Implementation Priority

### Phase 1: Make Training Work (Critical)
1. **UncertaintySampler** - Implement view sampling based on uncertainty
   - Sample candidate poses near training data
   - Compute uncertainty for each candidate
   - Select top-K highest uncertainty poses
   - Render using 3DGS renderer

2. **SemanticSynthesizer** - Basic semantic manipulation
   - Load semantic segmentation (or use pretrained model)
   - Apply appearance changes to semantic regions
   - Render modified scenes

3. **HardNegativeMiner** - Basic adversarial mining
   - Generate pose perturbations
   - Render scenes with perturbations
   - Select samples that maximize prediction error

### Phase 2: Full Functionality (Important)
1. **HypothesisValidator** - Hypothesis ranking
   - Render each hypothesis pose
   - Compute similarity (SSIM, LPIPS)
   - Rank hypotheses

2. **UncertaintyVisualizer** - Visualization tools
   - Heatmap overlay on images
   - Uncertainty distribution plots

### Phase 3: Optimization (Nice to Have)
1. Performance optimizations
2. Memory efficiency improvements
3. Batch processing optimizations

## Quick Start Guide

### Option A: Minimal Implementation (Get Training Working)

1. **Modify trainers to skip placeholder functions temporarily:**
   - Comment out calls to `UncertaintySampler`, `SemanticSynthesizer`, etc.
   - Use basic data augmentation instead
   - Train models to verify infrastructure works

2. **Train baseline:**
   ```bash
   python rap.py -c configs/aachen.txt -m /path/to/3dgs -n baseline_test
   ```

3. **Train extensions (with simplified versions):**
   ```bash
   python train.py -c configs/aachen.txt -m /path/to/3dgs --trainer_type uaas -n uaas_test
   ```

### Option B: Full Implementation (Recommended)

1. **Implement placeholder functions one by one:**
   - Start with `UncertaintySampler` (most critical)
   - Then `SemanticSynthesizer`
   - Then `HypothesisValidator`
   - Finally `HardNegativeMiner`

2. **Test each implementation:**
   - Unit tests for each function
   - Integration tests with trainers
   - Visual inspection of outputs

3. **Full training pipeline:**
   - Train all models
   - Run benchmarks
   - Generate comparison reports

## Testing Strategy

1. **Unit Tests:**
   - Test each implemented function independently
   - Mock dependencies (renderer, etc.)

2. **Integration Tests:**
   - Test trainer initialization
   - Test training loop (few iterations)
   - Test evaluation

3. **Benchmarking:**
   - Compare against baseline
   - Verify improvements match expectations
   - Check for regressions

## Expected Timeline

- **Phase 1 (Critical):** 1-2 days
  - Implement placeholder functions
  - Get training working

- **Phase 2 (Full):** 3-5 days
  - Complete implementations
  - Full training pipeline

- **Phase 3 (Benchmarking):** 1-2 days
  - Train all models
  - Run benchmarks
  - Generate reports

- **Phase 4 (Optimization):** Ongoing
  - Performance improvements
  - Code refactoring
  - Documentation updates

## Questions to Consider

1. **Do you have a dataset ready?** (7Scenes, Cambridge, custom)
2. **Do you have 3DGS models trained?** (Required for RAP training)
3. **Which extension to prioritize?** (UAAS, Probabilistic, or Semantic)
4. **Hardware constraints?** (GPU memory, multi-GPU setup)

## Resources

- **Original RAP Paper:** https://ai4ce.github.io/RAP/static/RAP_Paper.pdf
- **Original RAP Repo:** https://github.com/ai4ce/RAP
- **Technical Documentation:** `TECHNICAL_DOCUMENTATION.md`
- **Benchmarking Guide:** `BENCHMARKING_GUIDE.md`
