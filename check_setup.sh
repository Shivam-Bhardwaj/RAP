#!/bin/bash
# Quick setup verification script
cd "$(dirname "$0")"

echo "=================================="
echo "RAP Environment Status Check"
echo "=================================="
echo ""

# Check venv
if [ -d "venv" ]; then
    echo "✓ Virtual environment exists"
    source venv/bin/activate
    
    # Check Python
    PYTHON_VERSION=$(python --version 2>&1)
    echo "✓ $PYTHON_VERSION"
    
    # Check PyTorch
    if python -c "import torch" 2>/dev/null; then
        PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
        CUDA_AVAIL=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        echo "✓ PyTorch: $PYTORCH_VERSION"
        echo "  CUDA Available: $CUDA_AVAIL"
    else
        echo "✗ PyTorch not installed"
    fi
    
    # Check key dependencies
    echo ""
    echo "Checking dependencies..."
    # Map package names to their import names
    declare -A DEP_MAP=(
        ["kornia"]="kornia"
        ["lpips"]="lpips"
        ["wandb"]="wandb"
        ["opencv-python"]="cv2"
        ["numpy"]="numpy"
    )
    for dep in "kornia" "lpips" "wandb" "opencv-python" "numpy"; do
        import_name="${DEP_MAP[$dep]}"
        if python -c "import ${import_name}" 2>/dev/null; then
            echo "  ✓ $dep"
        else
            echo "  ✗ $dep"
        fi
    done
    
    # Check submodules
    echo ""
    echo "Checking submodules..."
    if python -c "import fused_ssim" 2>/dev/null; then
        echo "  ✓ fused-ssim"
    else
        echo "  ✗ fused-ssim (requires CUDA)"
    fi
    
    if python -c "import gsplat" 2>/dev/null; then
        echo "  ✓ gsplat"
    else
        echo "  ✗ gsplat (requires CUDA)"
    fi
    
    # Check CUDA
    echo ""
    echo "CUDA Status:"
    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            echo "  ✓ NVIDIA driver working"
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
        else
            echo "  ✗ NVIDIA driver not communicating (may need reboot)"
        fi
    else
        echo "  ✗ nvidia-smi not found"
    fi
    
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}')
        echo "  ✓ CUDA Toolkit: $CUDA_VER"
    else
        echo "  ✗ CUDA Toolkit not found"
    fi
    
else
    echo "✗ Virtual environment not found"
    echo "Run: bash setup_env_alt.sh"
fi

echo ""
echo "=================================="

