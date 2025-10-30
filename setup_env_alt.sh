#!/bin/bash
# RAP Environment Setup Script - Alternative Approach
set -e

cd "$(dirname "$0")"
PROJECT_DIR=$(pwd)

echo "=================================="
echo "RAP Environment Setup"
echo "=================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Check if we can use system package manager
if command -v apt-get &>/dev/null; then
    echo "Detected apt package manager"
    echo "Please run with sudo to install system packages:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y python3-pip python3-venv python3-dev"
    echo ""
    echo "Or try installing packages with --break-system-packages flag"
    echo ""
fi

# Try to install pip using get-pip.py with --break-system-packages
echo "Attempting to bootstrap pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --user --break-system-packages 2>&1 | tail -10

# Set PATH
export PATH="$HOME/.local/bin:$PATH"

# Check if pip is now available
if command -v pip3 &>/dev/null || python3 -m pip --version &>/dev/null; then
    echo "✓ pip is now available"
    
    # Install virtualenv
    echo "Installing virtualenv..."
    pip3 install --user --break-system-packages virtualenv 2>&1 | tail -5
    
    # Create virtual environment
    echo "Creating virtual environment..."
    if command -v virtualenv &>/dev/null; then
        virtualenv venv --python=python3
    else
        python3 -m virtualenv venv --python=python3
    fi
    
    # Activate and setup venv
    source venv/bin/activate
    
    echo "Upgrading pip in venv..."
    pip install --upgrade pip setuptools wheel
    
    echo ""
    echo "=================================="
    echo "Virtual environment ready!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate: source venv/bin/activate"
    echo "2. Install PyTorch: pip install torch torchvision torchaudio"
    echo "3. Install requirements: pip install -r requirements.txt"
    echo ""
else
    echo ""
    echo "Could not install pip automatically."
    echo "Please install python3-pip manually:"
    echo "  sudo apt-get install python3-pip python3-venv"
    echo ""
    echo "Or contact your system administrator."
    exit 1
fi

