# Fresh Start Guide

This guide helps you start fresh and properly document performance gains.

## Cleanup Process

### Automated Cleanup

Run the cleanup script to remove all runtime files:

```bash
./cleanup_runtime_files.sh
```

This removes:
- ✓ Training checkpoints (`output/*/logs/`)
- ✓ GS model checkpoints (`output/*/model/ckpts_point_cloud/`)
- ✓ Wandb logs (`wandb/`)
- ✓ Python cache files (`__pycache__/`, `*.pyc`)

**Preserves:**
- ✓ All datasets in `data/`
- ✓ All source code
- ✓ Configs (`configs/`)
- ✓ Scripts and documentation

---

## Fresh Start Workflow

### Step 1: Verify Dataset

Ensure your dataset is ready:

```bash
# Check dataset structure
ls data/Cambridge/KingsCollege/colmap/
# Should see: images/, sparse/, cameras.json, etc.
```

### Step 2: Train Gaussian Splatting

Train GS model for your dataset:

```bash
python gs.py \
    -s data/Cambridge/KingsCollege/colmap \
    -m output/Cambridge/KingsCollege \
    --iterations 30000 \
    --eval

# Verify checkpoint created
ls output/Cambridge/KingsCollege/model/ckpts_point_cloud/
```

### Step 3: Train All Models

Train all models systematically:

```bash
./train_all_models.sh \
    data/Cambridge/KingsCollege/colmap \
    output/Cambridge/KingsCollege \
    configs/7scenes.txt \
    100  # epochs
```

### Step 4: Run Benchmark

Run comprehensive benchmark with trained checkpoints:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda \
    --batch_size 8
```

### Step 5: Analyze Results

Analyze and document improvements:

```bash
python analyze_results.py
```

### Step 6: Document Gains

Update documentation with new results.

---

## Complete Run Example

```bash
# 1. Cleanup (if needed)
./cleanup_runtime_files.sh

# 2. Train GS
python gs.py -s data/Cambridge/KingsCollege/colmap \
             -m output/Cambridge/KingsCollege \
             --iterations 30000 --eval

# 3. Train all models
./train_all_models.sh data/Cambridge/KingsCollege/colmap \
                      output/Cambridge/KingsCollege \
                      configs/7scenes.txt 100

# 4. Benchmark
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda

# 5. Analyze
python analyze_results.py
```

