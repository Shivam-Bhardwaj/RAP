#!/bin/bash
# Install PyTorch with CUDA 12.x support
set -e

cd "$(dirname "$0")"

echo "Installing PyTorch with CUDA 12.x support..."

source venv/bin/activate

# Uninstall current PyTorch if exists
echo "Removing old PyTorch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.0)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'CUDA Devices: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "✓ PyTorch installation complete!"

