# Environment Setup Guide

## Virtual Environment

This project uses a Python virtual environment to manage dependencies and avoid conflicts.

### Initial Setup

1. Create virtual environment:
```bash
python3 -m venv venv
```

2. Activate virtual environment:
```bash
source venv/bin/activate
# OR use the helper script:
source activate_env.sh
```

3. Install PyTorch first (required for submodules):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

4. Install all requirements:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

5. Install submodules:
```bash
cd submodules/fused-ssim && pip install -e . && cd ../..
cd submodules/gsplat && pip install -e . && cd ../..
```

### Usage

**Always activate the virtual environment before running any scripts:**

```bash
source venv/bin/activate
# Now run your commands
python gs.py --source_path ...
python train.py --trainer_type uaas ...
```

### Deactivate

To exit the virtual environment:
```bash
deactivate
```

### Troubleshooting

- **ModuleNotFoundError**: Make sure the virtual environment is activated
- **fused_ssim not found**: Ensure you installed submodules and added them to PYTHONPATH (handled automatically in gs.py)
- **Path issues**: Always use absolute paths or paths relative to the project root when passing arguments

