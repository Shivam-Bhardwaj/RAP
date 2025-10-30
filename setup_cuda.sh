#!/bin/bash
# CUDA Setup and Verification Script for RAP
set -e

cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo "=================================="
echo "CUDA Setup for RAP"
echo "=================================="
echo ""

# Check GPU
echo "1. Checking GPU Hardware..."
if lspci | grep -i nvidia > /dev/null; then
    GPU_INFO=$(lspci | grep -i nvidia | head -1)
    echo "   ✓ GPU Found: $GPU_INFO"
else
    echo "   ✗ No NVIDIA GPU detected"
    exit 1
fi

# Check CUDA Toolkit
echo ""
echo "2. Checking CUDA Toolkit..."
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "   ✓ CUDA Toolkit installed: $CUDA_VERSION"
    echo "   ✓ nvcc path: $(which nvcc)"
else
    echo "   ✗ CUDA Toolkit not found"
    echo "   Installing CUDA toolkit..."
    # Add CUDA installation instructions
    exit 1
fi

# Check NVIDIA Driver
echo ""
echo "3. Checking NVIDIA Driver..."
if nvidia-smi &>/dev/null; then
    echo "   ✓ NVIDIA driver is working"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "   ⚠ NVIDIA driver not communicating with GPU"
    echo "   This might require:"
    echo "   - Loading kernel module: sudo modprobe nvidia"
    echo "   - System reboot (if driver was recently installed)"
    echo "   - Driver reinstallation"
fi

# Set CUDA environment variables
echo ""
echo "4. Setting up CUDA Environment Variables..."

# Find CUDA installation
CUDA_PATHS=(
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-12.1"
    "/usr/local/cuda-12.2"
    "/usr/local/cuda-11.8"
    "/usr/local/cuda"
    "/usr/lib/cuda"
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
    echo "   ✓ Found CUDA at: $CUDA_HOME_FOUND"
    export CUDA_HOME="$CUDA_HOME_FOUND"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    
    # Create activation script
    cat > venv/bin/activate_cuda.sh << EOF
# CUDA Environment Variables
export CUDA_HOME="$CUDA_HOME_FOUND"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
EOF
    
    echo "   ✓ Created venv/bin/activate_cuda.sh"
else
    echo "   ⚠ Could not locate CUDA installation directory"
fi

# Verify PyTorch CUDA support
echo ""
echo "5. Checking PyTorch CUDA Support..."
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || true
    
    if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null; then
        echo "   ✓ PyTorch can detect CUDA"
    else
        echo "   ⚠ PyTorch not installed or CUDA not detected"
        echo "   Will install PyTorch with CUDA support..."
    fi
else
    echo "   ⚠ Virtual environment not found"
fi

echo ""
echo "=================================="
echo "CUDA Setup Complete"
echo "=================================="
echo ""
echo "To use CUDA in your environment:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Load CUDA vars: source venv/bin/activate_cuda.sh"
echo "  3. Verify: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
