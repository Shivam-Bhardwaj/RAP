#!/bin/bash
# Quick CUDA Setup Script - Alternative using apt
set -e

echo "=================================="
echo "CUDA Setup via apt (Ubuntu)"
echo "=================================="

# Update package list
sudo apt-get update

# Install NVIDIA driver
echo "Installing NVIDIA driver..."
sudo apt-get install -y nvidia-driver-535  # RTX 4070 SUPER compatible

# Install CUDA toolkit via apt
echo "Installing CUDA toolkit..."
sudo apt-get install -y nvidia-cuda-toolkit

# Verify installation
echo ""
echo "Verifying installation..."
if command -v nvcc &>/dev/null; then
    nvcc --version
else
    echo "⚠ nvcc not found in PATH. Adding CUDA to PATH..."
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    
    if ! grep -q "CUDA" ~/.bashrc; then
        echo '' >> ~/.bashrc
        echo '# CUDA' >> ~/.bashrc
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    source ~/.bashrc
    nvcc --version
fi

echo ""
echo "✓ CUDA installation complete!"
echo ""
echo "⚠ Note: You may need to reboot for the NVIDIA driver to load."
echo "   After reboot, verify with: nvidia-smi"

