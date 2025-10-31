# Complete Pipeline Run Guide

Step-by-step guide to run the entire pipeline from scratch.

## Prerequisites Check

```bash
# Check dataset exists
ls data/Cambridge/KingsCollege/colmap/

# Check config exists
ls configs/7scenes.txt

# Verify GPU available
nvidia-smi
```

---

## Step 1: Train Gaussian Splatting Model

This is required before training RAP models.

```bash
python gs.py \
    -s data/Cambridge/KingsCollege/colmap \
    -m output/Cambridge/KingsCollege \
    --iterations 30000 \
    --eval
```

**What this does:**
- Creates 3D Gaussian Splatting representation
- Saves checkpoints to `output/Cambridge/KingsCollege/model/ckpts_point_cloud/`
- Takes ~30min - 2 hours depending on GPU

**Verify it worked:**
```bash
ls output/Cambridge/KingsCollege/model/ckpts_point_cloud/
# Should see checkpoint directories
```

---

## Step 2: Train All Models

Train baseline and all enhanced models:

```bash
./train_all_models.sh \
    data/Cambridge/KingsCollege/colmap \
    output/Cambridge/KingsCollege \
    configs/7scenes.txt \
    100
```

**What this does:**
- Trains Baseline RAP model
- Trains UAAS model
- Trains Probabilistic model
- Trains Semantic model
- Saves checkpoints to `output/Cambridge/KingsCollege/logs/*/checkpoints/`
- Takes ~4-10 hours total depending on GPU and epochs

**Monitor training:**
```bash
# In another terminal, watch logs
tail -f output/Cambridge/KingsCollege/logs/*/train.log

# Or check GPU usage
watch -n 1 nvidia-smi
```

**Verify checkpoints created:**
```bash
ls output/Cambridge/KingsCollege/logs/*/checkpoints/
```

---

## Step 3: Run Comprehensive Benchmark

After all models are trained, run benchmark:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda \
    --batch_size 8
```

**What this does:**
- Loads trained checkpoints
- Benchmarks all models on test set
- Measures: inference speed, accuracy (translation & rotation), model size
- Generates 7 visualization charts
- Saves results to `benchmark_full_pipeline_results.json`
- Takes ~5-10 minutes

**Output files:**
- `benchmark_full_pipeline_results.json` - Complete metrics
- `benchmark_full_pipeline_results_charts_*.png` - 7 charts

---

## Step 4: Analyze Results

Analyze and view improvements:

```bash
python analyze_results.py
```

**What this shows:**
- Comprehensive comparison table
- Improvement percentages for each model
- Summary of best performing models
- Recommendations

---

## Quick Start (All Steps)

Copy-paste this entire workflow:

```bash
# Step 1: Train GS (30min - 2 hours)
python gs.py \
    -s data/Cambridge/KingsCollege/colmap \
    -m output/Cambridge/KingsCollege \
    --iterations 30000 \
    --eval

# Verify GS checkpoint
ls output/Cambridge/KingsCollege/model/ckpts_point_cloud/

# Step 2: Train all models (4-10 hours)
./train_all_models.sh \
    data/Cambridge/KingsCollege/colmap \
    output/Cambridge/KingsCollege \
    configs/7scenes.txt \
    100

# Verify checkpoints
ls output/Cambridge/KingsCollege/logs/*/checkpoints/

# Step 3: Benchmark (5-10 minutes)
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda \
    --batch_size 8

# Step 4: Analyze
python analyze_results.py
```

---

## Troubleshooting

### GS Training Fails

```bash
# Check dataset structure
ls data/Cambridge/KingsCollege/colmap/
# Need: images/, sparse/, cameras.json
```

### Model Training Fails

```bash
# Check GS checkpoint exists
ls output/Cambridge/KingsCollege/model/ckpts_point_cloud/

# Check GPU memory
nvidia-smi

# Reduce batch size if OOM
# Edit train_all_models.sh or run manually with --batch_size 1
```

### Benchmark Fails

```bash
# Check checkpoints exist
ls output/Cambridge/KingsCollege/logs/*/checkpoints/

# Check checkpoint format
# Should have files like: full_checkpoint.pth or checkpoint_epoch_*.pth
```

---

## Expected Timeline

- **GS Training**: 30min - 2 hours
- **Model Training**: 4-10 hours (100 epochs, 4 models)
- **Benchmarking**: 5-10 minutes
- **Analysis**: Instant

**Total**: ~5-12 hours for complete pipeline

---

## After Completion

1. **Review results**:
   ```bash
   cat benchmark_full_pipeline_results.json | python -m json.tool | less
   ```

2. **View charts**:
   ```bash
   ls -lh benchmark_full_pipeline_results_charts*.png
   ```

3. **Update README** with new results

4. **Update papers** with documented gains

---

## Tips

- **Use screen/tmux** for long training sessions
- **Monitor GPU usage** with `watch nvidia-smi`
- **Save good checkpoints** - training takes hours
- **Log everything** - helps with debugging
- **Run on subset first** - use `--max_samples 10` for testing

