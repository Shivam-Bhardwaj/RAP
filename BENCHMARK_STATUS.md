# RAP Benchmarking Status Report

## Current System Status

**Date:** Generated via benchmarking script
**Project:** /home/curious/projects/RAP

### Prerequisites Check Results

✅ **Python**: 3.12.3 (meets requirement of 3.11+)
❌ **PyTorch**: Not installed
❌ **Dependencies**: wandb, numpy, opencv-python, kornia, lpips not installed
❌ **Data Directory**: Not found
❌ **Checkpoints**: Not found
❌ **3DGS Models**: Not found

## Available Benchmarking Tools

### Automated Benchmarking Script (`run_benchmark.py`)

Created a comprehensive benchmarking tool that:
- ✅ Checks all prerequisites automatically
- ✅ Scans for available datasets and configurations
- ✅ Detects trained checkpoints
- ✅ Guides through benchmark execution
- ✅ Provides clear error messages and next steps

**Usage:**
```bash
python3 run_benchmark.py
```

## Available Configurations

Found **25 configuration files** ready for benchmarking:

- **7Scenes** (7 scenes): chess, fire, heads, office, pumpkin, kitchen, stairs
- **7Scenes_sfm** (7 scenes): Same with COLMAP SfM format
- **Cambridge** (5 scenes): shop, hospital, college, church, court
- **MARS** (4 sequences): 11, 15, 37, 41
- **Aachen**: aachen_sub
- **St. George's**: new_church

## Next Steps to Run Benchmark

### Step 1: Install Dependencies
```bash
cd /home/curious/projects/RAP
pip install -r requirements.txt
```

### Step 2: Prepare Dataset
Download and setup a dataset (e.g., 7Scenes or Cambridge)

### Step 3: Train 3DGS Model
```bash
python gs.py -s data/7Scenes/chess -m output/7Scenes/chess
```

### Step 4: Train RAPNet
```bash
python rap.py -c configs/7Scenes/chess.txt -m output/7Scenes/chess
```

### Step 5: Run Benchmark
```bash
python3 run_benchmark.py
```

## Expected Performance Targets

- **Indoor (7Scenes)**: <1cm / <0.3°
- **Outdoor (Cambridge)**: <20cm / <0.5°
- **Driving (MARS)**: <10cm / <0.2°

