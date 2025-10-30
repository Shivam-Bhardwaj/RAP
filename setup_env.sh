#!/bin/bash
# RAP Environment Setup Script
set -e

cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo "=================================="
echo "RAP Environment Setup"
echo "=================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Download get-pip.py if pip is not available
if ! python3 -m pip --version &>/dev/null; then
    echo "pip not found. Bootstrapping pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python3 /tmp/get-pip.py --user
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    # Try venv first
    if python3 -m venv venv 2>/dev/null; then
        echo "✓ Virtual environment created using venv"
    else
        # Try virtualenv
        if command -v virtualenv &>/dev/null; then
            virtualenv venv
            echo "✓ Virtual environment created using virtualenv"
        else
            echo "Installing virtualenv..."
            python3 -m pip install --user virtualenv
            python3 -m virtualenv venv
            echo "✓ Virtual environment created"
        fi
    fi
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch first (if CUDA available, install CUDA version)
echo ""
echo "Checking for CUDA..."
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "CUDA found: $CUDA_VERSION"
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not found. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install requirements
echo ""
echo "Installing project dependencies..."
echo "This may take a while (especially compiling CUDA extensions)..."
pip install -r requirements.txt

# Install submodules
echo ""
echo "Installing submodules..."
if [ -d "submodules/gsplat" ]; then
    echo "Installing gsplat..."
    pip install -e submodules/gsplat
fi

if [ -d "submodules/fused-ssim" ]; then
    echo "Installing fused-ssim..."
    pip install -e submodules/fused-ssim
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To verify installation:"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"
echo ""

