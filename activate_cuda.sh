#!/bin/bash
# Activate CUDA environment for RAP
cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Run setup_env.sh first."
    exit 1
fi

# Set CUDA environment variables
# Find CUDA installation
CUDA_PATHS=(
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda"
)

CUDA_HOME_FOUND=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
        CUDA_HOME_FOUND="$path"
        break
    fi
done

if [ -z "$CUDA_HOME_FOUND" ]; then
    # Try to find from nvcc
    NVCC_PATH=$(which nvcc)
    if [ -n "$NVCC_PATH" ]; then
        CUDA_HOME_FOUND=$(dirname $(dirname "$NVCC_PATH"))
    fi
fi

if [ -n "$CUDA_HOME_FOUND" ]; then
    export CUDA_HOME="$CUDA_HOME_FOUND"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Set CUDA library paths
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0

echo "CUDA environment activated!"
echo "CUDA_HOME: ${CUDA_HOME:-/usr}"
echo ""

# Check CUDA availability
python -c "import torch; cuda_avail = torch.cuda.is_available(); print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {cuda_avail}'); print(f'CUDA Version: {torch.version.cuda if cuda_avail else \"N/A\"}'); print(f'CUDA Devices: {torch.cuda.device_count() if cuda_avail else 0}')" 2>/dev/null

if ! python -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    echo ""
    echo "⚠ WARNING: CUDA is not available to PyTorch"
    echo "This usually means the NVIDIA driver kernel module is not loaded."
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Try loading the module: sudo modprobe nvidia"
    echo "2. Check driver status: nvidia-smi"
    echo "3. If driver fails, you may need to reboot the system"
    echo "4. Verify GPU is visible: lspci | grep -i nvidia"
    echo ""
fi

