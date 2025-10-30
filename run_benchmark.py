#!/usr/bin/env python3
"""
Benchmarking script for RAP visual localization system.
This script checks prerequisites and runs a benchmark on a selected dataset.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("=" * 60)
    print("Checking Prerequisites for RAP Benchmarking")
    print("=" * 60)
    
    checks = {}
    
    # Check Python version
    python_version = sys.version_info
    checks['python'] = python_version.major >= 3 and python_version.minor >= 11
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}: {'OK' if checks['python'] else 'FAILED (Need Python 3.11+)'}")
    
    # Check PyTorch
    try:
        import torch
        checks['pytorch'] = True
        checks['cuda_available'] = torch.cuda.is_available()
        checks['cuda_devices'] = torch.cuda.device_count() if checks['cuda_available'] else 0
        print(f"✓ PyTorch {torch.__version__}: OK")
        if checks['cuda_available']:
            print(f"  ✓ CUDA available: {checks['cuda_devices']} device(s)")
            for i in range(checks['cuda_devices']):
                props = torch.cuda.get_device_properties(i)
                print(f"    - GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print(f"  ⚠ CUDA not available (will use CPU - very slow)")
    except ImportError:
        checks['pytorch'] = False
        checks['cuda_available'] = False
        print(f"✗ PyTorch: NOT INSTALLED")
    
    # Check other dependencies
    deps = ['wandb', 'numpy', 'opencv-python', 'kornia', 'lpips']
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
            checks[dep] = True
            print(f"✓ {dep}: OK")
        except ImportError:
            checks[dep] = False
            print(f"✗ {dep}: NOT INSTALLED")
    
    # Check if data directory exists
    data_dir = Path("data")
    checks['data_dir'] = data_dir.exists()
    if checks['data_dir']:
        print(f"✓ Data directory exists: {data_dir}")
        # List available datasets
        datasets = []
        for item in data_dir.iterdir():
            if item.is_dir():
                datasets.append(item.name)
        if datasets:
            print(f"  Available datasets: {', '.join(datasets)}")
        else:
            print(f"  ⚠ No dataset subdirectories found")
    else:
        print(f"✗ Data directory not found: {data_dir}")
    
    # Check if checkpoints exist
    checkpoint_dirs = ['logs', 'ckpts']
    checks['checkpoints'] = False
    checkpoint_paths = []
    for ckpt_dir in checkpoint_dirs:
        path = Path(ckpt_dir)
        if path.exists():
            # Look for .pth files
            pth_files = list(path.rglob("*.pth"))
            if pth_files:
                checks['checkpoints'] = True
                checkpoint_paths.extend(pth_files)
                print(f"✓ Found checkpoints in {ckpt_dir}: {len(pth_files)} file(s)")
    
    if not checks['checkpoints']:
        print(f"✗ No checkpoints found in {checkpoint_dirs}")
    
    # Check if 3DGS models exist
    output_dir = Path("output")
    checks['gs_models'] = output_dir.exists() if output_dir else False
    if checks['gs_models']:
        gs_scenes = [d.name for d in output_dir.iterdir() if d.is_dir()]
        if gs_scenes:
            print(f"✓ Found 3DGS models: {len(gs_scenes)} scene(s)")
            print(f"  Scenes: {', '.join(gs_scenes[:5])}" + ("..." if len(gs_scenes) > 5 else ""))
        else:
            print(f"  ⚠ No 3DGS model directories found")
    else:
        print(f"✗ 3DGS models directory not found: {output_dir}")
    
    print("=" * 60)
    return checks, checkpoint_paths


def find_available_configs():
    """Find available configuration files."""
    configs_dir = Path("configs")
    if not configs_dir.exists():
        return []
    
    configs = []
    for config_file in configs_dir.rglob("*.txt"):
        configs.append(config_file)
    
    return configs


def select_dataset(configs):
    """Select a dataset to benchmark."""
    print("\nAvailable Configuration Files:")
    print("-" * 60)
    
    # Group by dataset
    datasets = {}
    for config in configs:
        parts = config.parts
        if len(parts) > 2:
            dataset = parts[1]
            scene = config.stem
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append((scene, config))
        else:
            scene = config.stem
            datasets['root'] = datasets.get('root', [])
            datasets['root'].append((scene, config))
    
    # Display options
    options = []
    idx = 1
    for dataset, scenes in sorted(datasets.items()):
        print(f"\n{dataset.upper()}:")
        for scene, config in sorted(scenes):
            print(f"  [{idx}] {scene}")
            options.append((idx, scene, config, dataset))
            idx += 1
    
    return options


def run_benchmark(config_path, checkpoint_path=None, gs_model_path=None):
    """Run benchmark evaluation."""
    print("\n" + "=" * 60)
    print("Running Benchmark")
    print("=" * 60)
    
    # Parse config to get dataset info
    config_data = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    config_data[key.strip()] = value.strip()
    
    run_name = config_data.get('run_name', config_path.stem)
    dataset_type = config_data.get('dataset_type', 'Colmap')
    datadir = config_data.get('datadir', '')
    
    print(f"Config: {config_path}")
    print(f"Run Name: {run_name}")
    print(f"Dataset Type: {dataset_type}")
    print(f"Data Directory: {datadir}")
    
    # Check if data directory exists
    if datadir and not Path(datadir).exists():
        print(f"\n✗ ERROR: Data directory not found: {datadir}")
        print("  Please ensure the dataset is properly set up.")
        return False
    
    # Determine checkpoint path
    if not checkpoint_path:
        # Try to find checkpoint based on config
        logbase = config_data.get('logbase', 'logs')
        checkpoint_patterns = [
            f"{logbase}/{run_name}/full_checkpoint.pth",
            f"{logbase}/{run_name}.pth",
            f"logs/{run_name}.pth",
        ]
        for pattern in checkpoint_patterns:
            if Path(pattern).exists():
                checkpoint_path = pattern
                break
    
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"\n✗ ERROR: Checkpoint not found")
        print(f"  Expected locations:")
        for pattern in checkpoint_patterns:
            print(f"    - {pattern}")
        print(f"\n  Please train a model first using:")
        print(f"    python rap.py -c {config_path} -m <3dgs_model_path>")
        return False
    
    print(f"Checkpoint: {checkpoint_path}")
    
    # Determine 3DGS model path
    if not gs_model_path:
        # Try common locations
        possible_paths = [
            f"output/{run_name}",
            f"output/{config_path.parent.name}/{run_name}",
            datadir.replace('data/', 'output/') if datadir.startswith('data/') else None,
        ]
        for path in possible_paths:
            if path and Path(path).exists():
                camera_file = Path(path) / "cameras.json"
                if camera_file.exists():
                    gs_model_path = path
                    break
    
    if not gs_model_path or not Path(gs_model_path).exists():
        print(f"\n✗ ERROR: 3DGS model not found")
        print(f"  Please train a 3DGS model first using:")
        print(f"    python gs.py -s <colmap_data> -m {gs_model_path}")
        return False
    
    print(f"3DGS Model: {gs_model_path}")
    
    # Run evaluation
    print("\n" + "-" * 60)
    print("Starting Evaluation...")
    print("-" * 60)
    
    cmd = [
        sys.executable, "eval.py",
        "-c", str(config_path),
        "-m", str(gs_model_path),
        "-p", str(checkpoint_path),
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("Benchmark Completed Successfully!")
        print("=" * 60)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Benchmark failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ ERROR: eval.py not found. Make sure you're in the RAP project directory.")
        return False


def main():
    """Main benchmarking function."""
    print("\n")
    print("=" * 60)
    print("RAP Benchmarking Tool")
    print("=" * 60)
    
    # Check prerequisites
    checks, checkpoint_paths = check_prerequisites()
    
    # Check critical dependencies
    if not checks.get('pytorch', False):
        print("\n✗ CRITICAL: PyTorch is not installed.")
        print("  Please install dependencies first:")
        print("    pip install -r requirements.txt")
        return 1
    
    # Find available configs
    configs = find_available_configs()
    if not configs:
        print("\n✗ No configuration files found in configs/ directory")
        return 1
    
    # Select dataset
    options = select_dataset(configs)
    if not options:
        print("\n✗ No valid configurations found")
        return 1
    
    # For now, auto-select first available config that has data and checkpoint
    # In interactive mode, you could ask user to select
    selected_config = None
    selected_checkpoint = None
    
    # Try to find a config with available data and checkpoint
    for idx, scene, config_path, dataset in options:
        config_data = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config_data[key.strip()] = value.strip()
        
        datadir = config_data.get('datadir', '')
        run_name = config_data.get('run_name', scene)
        
        # Check if data exists
        if datadir and Path(datadir).exists():
            # Check if checkpoint exists
            logbase = config_data.get('logbase', 'logs')
            checkpoint_candidates = [
                f"{logbase}/{run_name}/full_checkpoint.pth",
                f"{logbase}/{run_name}.pth",
            ]
            
            for ckpt_path in checkpoint_candidates:
                if Path(ckpt_path).exists():
                    selected_config = config_path
                    selected_checkpoint = ckpt_path
                    print(f"\n✓ Found ready-to-run benchmark:")
                    print(f"  Dataset: {dataset}/{scene}")
                    print(f"  Config: {selected_config}")
                    print(f"  Checkpoint: {selected_checkpoint}")
                    break
            
            if selected_config:
                break
    
    if not selected_config:
        print("\n" + "=" * 60)
        print("No benchmark ready to run automatically.")
        print("=" * 60)
        print("\nTo run a benchmark, you need:")
        print("1. A trained 3DGS model in output/<scene>/")
        print("2. A trained RAPNet checkpoint in logs/ or ckpts/")
        print("3. Dataset in data/<dataset>/<scene>/")
        print("\nExample workflow:")
        print("  1. Train 3DGS: python gs.py -s data/7Scenes/chess -m output/7Scenes/chess")
        print("  2. Train RAPNet: python rap.py -c configs/7Scenes/chess.txt -m output/7Scenes/chess")
        print("  3. Evaluate: python eval.py -c configs/7Scenes/chess.txt -m output/7Scenes/chess -p logs/chess.pth")
        return 0
    
    # Run benchmark
    print("\n")
    response = input("Run benchmark now? [y/N]: ").strip().lower()
    if response == 'y':
        success = run_benchmark(selected_config, selected_checkpoint)
        return 0 if success else 1
    else:
        print("\nBenchmark cancelled.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

