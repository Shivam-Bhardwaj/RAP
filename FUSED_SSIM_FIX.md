# Fixing fused_ssim Import Error

## Issue
`ModuleNotFoundError: No module named 'fused_ssim'`

## Solution

The `fused_ssim` module is a submodule that needs to be initialized and built.

### Step 1: Initialize Git Submodules

```bash
cd /home/ubuntu/RAP
git submodule update --init --recursive
```

### Step 2: Build fused_ssim

```bash
cd submodules/fused-ssim
python setup.py build_ext --inplace
cd ../..
```

### Step 3: Add to PYTHONPATH

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/submodules/fused-ssim"
```

### Step 4: Test

```bash
python -c "from fused_ssim import fused_ssim; print('✓ Working')"
```

### Step 5: Run GS Training

```bash
python gs.py \
    --source_path data/Cambridge/KingsCollege/colmap \
    --model_path data/Cambridge/KingsCollege/colmap/model \
    --images data/Cambridge/KingsCollege/colmap/images \
    --resolution 1 \
    --iterations 30000 \
    --eval
```

## Alternative: Use Setup Script

If the above doesn't work, you can modify gs.py to use an alternative SSIM implementation temporarily.

