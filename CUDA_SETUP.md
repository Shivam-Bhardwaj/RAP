# CUDA Setup Guide for RAP

## Current Status

✅ **GPU Hardware**: NVIDIA GeForce RTX 4070 SUPER detected
✅ **CUDA Toolkit**: Version 12.0 installed (`/usr/bin/nvcc`)
✅ **NVIDIA Driver**: Version 535.274.02 installed (packages)
✅ **CUDA Libraries**: Installed (`/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146`)
✅ **PyTorch**: Version 2.5.1+cu121 installed (CUDA 12.1 compatible)

⚠️ **Issue**: NVIDIA driver kernel module not loaded/communicating with GPU

## Quick Fix

The NVIDIA driver is installed but the kernel module needs to be loaded. Try:

```bash
# Option 1: Load the module
sudo modprobe nvidia

# Option 2: Check if it's loaded
lsmod | grep nvidia

# Option 3: Verify GPU communication
nvidia-smi
```

If `nvidia-smi` still fails, you may need to:
1. **Reboot the system** (most common solution)
2. Check if secure boot is enabled (may need to disable)
3. Verify driver installation: `dpkg -l | grep nvidia-driver`

## After Driver is Working

Once `nvidia-smi` works, activate the environment:

```bash
cd /home/curious/projects/RAP
source venv/bin/activate
source activate_cuda.sh  # Sets CUDA environment variables

# Verify CUDA is available
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Environment Variables

The `activate_cuda.sh` script sets:
- `CUDA_HOME`: CUDA installation directory
- `PATH`: Includes CUDA bin directory
- `LD_LIBRARY_PATH`: Includes CUDA lib64 directory
- `CUDA_VISIBLE_DEVICES`: Set to 0 (single GPU)

## Performance Optimization

For best performance with RTX 4070 SUPER:
- Ensure CUDA 12.x is properly configured
- Use PyTorch with CUDA 12.1+ (already installed)
- Enable mixed precision training (AMP) in your configs
- Use `torch.compile()` for faster inference (already enabled in configs)

## Next Steps

1. **Fix driver issue** (reboot or modprobe)
2. **Verify CUDA**: Run `nvidia-smi` successfully
3. **Test PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
4. **Install remaining dependencies**: `pip install -r requirements.txt`
5. **Install submodules**: `pip install -e submodules/gsplat`

