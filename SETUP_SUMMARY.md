# RAP Environment Setup Summary

## ✅ Completed Setup Steps

1. **Virtual Environment**: Created and activated (`venv/`)
2. **Python**: 3.12.3 ✓
3. **PyTorch**: 2.5.1+cu121 installed ✓
4. **Standard Dependencies**: All installed ✓
   - configargparse, efficientnet_pytorch, matplotlib, opencv-python
   - pandas, plyfile, scikit-learn, tqdm, wandb
   - kornia, lpips, imageio, einops, huggingface_hub
   - safetensors, kapture, kapture_localization, roma

## ⚠️ Pending Steps (Require CUDA Driver)

### 1. Fix NVIDIA Driver
The driver kernel module needs to be loaded:

```bash
# Try loading the module
sudo modprobe nvidia

# Or reboot the system
sudo reboot
```

After reboot, verify:
```bash
nvidia-smi  # Should show GPU information
```

### 2. Install CUDA Submodules (After Driver Works)

Once `nvidia-smi` works and CUDA is available to PyTorch:

```bash
cd /home/curious/projects/RAP
source venv/bin/activate
source activate_cuda.sh

# Install submodules
pip install -e submodules/fused-ssim
pip install -e submodules/gsplat
```

## Quick Start Commands

### Activate Environment
```bash
cd /home/curious/projects/RAP
source venv/bin/activate
source activate_cuda.sh  # Sets CUDA environment variables
```

### Verify Setup
```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Check dependencies
python -c "import kornia, lpips, wandb; print('All dependencies OK')"
```

### Check CUDA Status
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

## Current System Status

- **GPU**: NVIDIA GeForce RTX 4070 SUPER (detected)
- **CUDA Toolkit**: 12.0 (installed)
- **NVIDIA Driver**: 535.274.02 (installed but not loaded)
- **PyTorch**: 2.5.1+cu121 (installed)
- **Environment**: Ready (except for CUDA submodules)

## Next Steps for Performance Improvement

Once CUDA is working:

1. **Verify CUDA**: `python -c "import torch; print(torch.cuda.is_available())"`
2. **Install submodules**: `pip install -e submodules/gsplat submodules/fused-ssim`
3. **Prepare dataset**: Download and setup a benchmark dataset
4. **Run benchmark**: `python3 run_benchmark.py`
5. **Analyze performance**: Review metrics and identify bottlenecks

