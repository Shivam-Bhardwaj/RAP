# Complete Training and Analysis Guide

This guide covers how to train all models and perform comprehensive analysis.

## Overview

To get proper checkpoints and perform analysis, you need to:

1. **Train Gaussian Splatting (GS) models** for your dataset
2. **Train RAP baseline** model
3. **Train enhanced models** (UAAS, Probabilistic, Semantic)
4. **Run comprehensive benchmarking** with trained checkpoints
5. **Analyze results** across all metrics

---

## Step 1: Train Gaussian Splatting Models

First, train Gaussian Splatting models for your dataset. This is required for RAP training.

### Basic GS Training

```bash
python gs.py -s /path/to/colmap/data -m /path/to/output
```

### Example: Cambridge KingsCollege

```bash
python gs.py \
    -s data/Cambridge/KingsCollege/colmap \
    -m output/Cambridge/KingsCollege \
    --iterations 30000 \
    --eval
```

### Important GS Parameters

- `--iterations`: Number of training iterations (default: 30000)
- `--eval`: Enable evaluation during training
- `--white_background`: For synthetic data
- `--antialiasing`: Enable antialiasing
- `--use_masks`: Use masks if available

**Check GS checkpoint exists:**
```bash
ls output/Cambridge/KingsCollege/model/ckpts_point_cloud/
```

---

## Step 2: Train Baseline RAP Model

Train the original RAP baseline for comparison.

### Basic Training

```bash
python rap.py \
    -c configs/7scenes.txt \
    -m output/Cambridge/KingsCollege \
    --run_name baseline_kingscollege \
    --datadir data/Cambridge/KingsCollege/colmap
```

### Key Arguments

- `-c`: Config file
- `-m`: Path to GS model directory
- `--run_name`: Experiment name (checkpoints saved under this)
- `--datadir`: Path to dataset
- `--epochs`: Number of training epochs (default: varies by config)
- `--batch_size`: Batch size (default: 1)
- `--lr`: Learning rate

### Checkpoint Location

After training, checkpoints are saved to:
```
output/Cambridge/KingsCollege/logs/baseline_kingscollege/checkpoints/
```

Look for files like:
- `full_checkpoint.pth` - Complete checkpoint
- `checkpoint_epoch_X.pth` - Per-epoch checkpoints

---

## Step 3: Train Enhanced Models

### UAAS Model Training

```bash
python train.py \
    --trainer_type uaas \
    --run_name uaas_kingscollege \
    --datadir data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --config configs/7scenes.txt \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4
```

### Probabilistic Model Training

```bash
python train.py \
    --trainer_type probabilistic \
    --run_name probabilistic_kingscollege \
    --datadir data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --config configs/7scenes.txt \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --num_gaussians 5
```

### Semantic Model Training

```bash
python train.py \
    --trainer_type semantic \
    --run_name semantic_kingscollege \
    --datadir data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --config configs/7scenes.txt \
    --epochs 100 \
    --batch_size 1 \
    --lr 1e-4 \
    --num_semantic_classes 19
```

---

## Step 4: Automated Training Script

Use the provided script to train all models systematically:

```bash
./train_all_models.sh \
    data/Cambridge/KingsCollege/colmap \
    output/Cambridge/KingsCollege \
    configs/7scenes.txt \
    100  # epochs
```

Or customize:
```bash
./train_all_models.sh [dataset] [model_path] [config] [epochs] [batch_size] [lr]
```

---

## Step 5: Run Comprehensive Benchmarking

After training, run benchmarks with checkpoints:

```bash
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda \
    --batch_size 8
```

The benchmark will automatically find checkpoints in:
- `output/Cambridge/KingsCollege/logs/baseline_kingscollege/checkpoints/`
- `output/Cambridge/KingsCollege/logs/uaas_kingscollege/checkpoints/`
- `output/Cambridge/KingsCollege/logs/probabilistic_kingscollege/checkpoints/`
- `output/Cambridge/KingsCollege/logs/semantic_kingscollege/checkpoints/`

---

## Step 6: Analysis

### Quick Analysis

```bash
python analyze_results.py benchmark_full_pipeline_results.json
```

This will show:
- Comprehensive comparison of all models
- Improvements/worsenings across metrics
- Summary and recommendations

### Manual Analysis

Results are stored in `benchmark_full_pipeline_results.json`:
```python
import json
with open('benchmark_full_pipeline_results.json') as f:
    data = json.load(f)
    
# Access results
baseline = data['results']['baseline']
uaas = data['results']['uaas']
improvements = data['improvements']
```

---

## Training Tips

### 1. Monitor Training

Use wandb or tensorboard:
```bash
# If using wandb
wandb login
# Training will automatically log to wandb
```

### 2. Early Stopping

Monitor validation loss and stop if overfitting:
- Check validation loss in logs
- Use early stopping if loss plateaus

### 3. Learning Rate Scheduling

Adjust learning rate:
```bash
--lr 1e-4 --lr_decay 0.9 --lr_decay_steps 1000
```

### 4. Batch Size

Adjust based on GPU memory:
- 24GB GPU: batch_size=1 or 2
- Smaller GPUs: batch_size=1

### 5. Checkpoint Frequency

Checkpoints saved every epoch automatically.

---

## Expected Training Times

- **GS Training**: 30min - 2 hours (depending on dataset size)
- **RAP Baseline**: 1-4 hours (depending on epochs)
- **UAAS/Probabilistic/Semantic**: 2-6 hours each

---

## Troubleshooting

### No Checkpoints Found

Ensure training completed successfully:
```bash
ls output/Cambridge/KingsCollege/logs/*/checkpoints/
```

### CUDA Out of Memory

Reduce batch size:
```bash
--batch_size 1
```

### Slow Training

- Use GPU acceleration
- Reduce dataset size for testing
- Use `--train_skip` to subsample training data

---

## Complete Workflow Example

```bash
# 1. Train GS (if not already done)
python gs.py -s data/Cambridge/KingsCollege/colmap \
             -m output/Cambridge/KingsCollege \
             --iterations 30000

# 2. Train all models
./train_all_models.sh data/Cambridge/KingsCollege/colmap \
                      output/Cambridge/KingsCollege \
                      configs/7scenes.txt \
                      100

# 3. Run benchmark
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda

# 4. Analyze results
python analyze_results.py
```

---

## Next Steps After Training

1. **Run full benchmark** with trained checkpoints
2. **Compare results** across all metrics
3. **Generate visualizations** (automatic)
4. **Update README** with new results
5. **Write analysis report** based on findings

