#!/usr/bin/env python3
"""
Compare performance between original RAP repo and optimized version.
This script helps measure the speed boost from optimizations.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import shutil

def clone_original_repo(clone_dir="/home/ubuntu/RAP_original"):
    """Clone the original RAP repository for comparison."""
    clone_path = Path(clone_dir)
    
    if clone_path.exists():
        print(f"✓ Original repo already exists at {clone_dir}")
        return clone_dir
    
    print(f"Cloning original RAP repo to {clone_dir}...")
    try:
        subprocess.run(
            ["git", "clone", "--recursive", "https://github.com/ai4ce/RAP.git", clone_dir],
            check=True,
            capture_output=True
        )
        print(f"✓ Successfully cloned original repo")
        return clone_dir
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone original repo: {e}")
        print(f"Error output: {e.stderr.decode()}")
        return None


def setup_original_repo(repo_path):
    """Set up the original repo with virtual environment and dependencies."""
    repo_path = Path(repo_path)
    venv_path = repo_path / "venv"
    
    if venv_path.exists():
        print(f"✓ Virtual environment already exists at {venv_path}")
        return True
    
    print(f"Setting up virtual environment for original repo...")
    try:
        # Create venv
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            check=True,
            cwd=repo_path
        )
        
        # Install PyTorch first (required for CUDA extensions)
        pip_path = venv_path / "bin" / "pip"
        print("Installing PyTorch first (required for CUDA extensions)...")
        subprocess.run(
            [str(pip_path), "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True,
            cwd=repo_path
        )
        
        # Install requirements
        print("Installing dependencies (this may take a while)...")
        subprocess.run(
            [str(pip_path), "install", "-r", "requirements.txt"],
            check=True,
            cwd=repo_path
        )
        
        print("✓ Original repo setup complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to set up original repo: {e}")
        return False


def run_benchmark_on_repo(repo_path, model_path, colmap_path, output_name="baseline"):
    """Run benchmark on a specific repo version."""
    repo_path = Path(repo_path)
    venv_python = repo_path / "venv" / "bin" / "python"
    benchmark_script = repo_path / "benchmark_speed.py"
    
    if not benchmark_script.exists():
        print(f"⚠ Benchmark script not found in {repo_path}")
        print("  Using current benchmark script...")
        benchmark_script = Path("/home/ubuntu/RAP/benchmark_speed.py")
    
    # Check if original repo has benchmark script, if not copy it
    if not (repo_path / "benchmark_speed.py").exists():
        print(f"Copying benchmark script to {repo_path}...")
        shutil.copy(benchmark_script, repo_path / "benchmark_speed.py")
    
    output_dir = repo_path / "benchmark_output" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning benchmark on {output_name}...")
    print(f"Model: {model_path}")
    print(f"COLMAP: {colmap_path}")
    
    cmd = [
        str(venv_python),
        "benchmark_speed.py",
        "-s", str(colmap_path),
        "-m", str(model_path),
        "--iteration", "30000",
        "--num_iterations", "50",
        "--output", str(output_dir)
    ]
    
    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"
    env["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            env=env,
            capture_output=True,
            text=True,
            check=True
        )
        
        results_file = output_dir / "benchmark_results.json"
        if results_file.exists():
            print(f"✓ Benchmark completed: {results_file}")
            return str(results_file)
        else:
            print(f"✗ Benchmark results file not found")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Benchmark failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def compare_benchmarks(baseline_file, optimized_file):
    """Compare benchmark results between baseline and optimized versions."""
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)
    
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_file, 'r') as f:
        optimized = json.load(f)
    
    print("\n📊 RENDERING SPEED COMPARISON:")
    print("-" * 70)
    
    comparisons = []
    
    # Test set comparison
    if 'test_set' in baseline and 'test_set' in optimized:
        base = baseline['test_set']
        opt = optimized['test_set']
        base_fps = base.get('fps', 0)
        opt_fps = opt.get('fps', 0)
        speedup = opt_fps / base_fps if base_fps > 0 else 0
        improvement = (speedup - 1.0) * 100
        
        print(f"\nTest Set Rendering:")
        print(f"  Original:      {base_fps:.2f} FPS ({base.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Optimized:     {opt_fps:.2f} FPS ({opt.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Speedup:       {speedup:.2f}x ({improvement:+.1f}%)")
        
        comparisons.append(('Test Set', speedup, improvement))
    
    # Train set comparison
    if 'train_set' in baseline and 'train_set' in optimized:
        base = baseline['train_set']
        opt = optimized['train_set']
        base_fps = base.get('fps', 0)
        opt_fps = opt.get('fps', 0)
        speedup = opt_fps / base_fps if base_fps > 0 else 0
        improvement = (speedup - 1.0) * 100
        
        print(f"\nTrain Set Rendering:")
        print(f"  Original:      {base_fps:.2f} FPS ({base.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Optimized:     {opt_fps:.2f} FPS ({opt.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Speedup:       {speedup:.2f}x ({improvement:+.1f}%)")
        
        comparisons.append(('Train Set', speedup, improvement))
    
    # Standard test comparison
    if 'standard_test' in baseline and 'standard_test' in optimized:
        base = baseline['standard_test']
        opt = optimized['standard_test']
        base_fps = base.get('fps', 0)
        opt_fps = opt.get('fps', 0)
        speedup = opt_fps / base_fps if base_fps > 0 else 0
        improvement = (speedup - 1.0) * 100
        
        print(f"\nStandard Speed Test (800x800):")
        print(f"  Original:      {base_fps:.2f} FPS ({base.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Optimized:     {opt_fps:.2f} FPS ({opt.get('time_per_image', 0)*1000:.2f} ms/image)")
        print(f"  Speedup:       {speedup:.2f}x ({improvement:+.1f}%)")
        
        comparisons.append(('Standard Test', speedup, improvement))
    
    # GPU memory comparison
    if 'gpu_memory' in baseline and 'gpu_memory' in optimized:
        base_mem = baseline['gpu_memory'].get('allocated_gb', 0)
        opt_mem = optimized['gpu_memory'].get('allocated_gb', 0)
        mem_change = opt_mem - base_mem
        mem_change_pct = (mem_change / base_mem * 100) if base_mem > 0 else 0
        
        print(f"\nGPU Memory Usage:")
        print(f"  Original:      {base_mem:.2f} GB")
        print(f"  Optimized:     {opt_mem:.2f} GB")
        print(f"  Change:         {mem_change:+.2f} GB ({mem_change_pct:+.1f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if comparisons:
        avg_speedup = sum(s[1] for s in comparisons) / len(comparisons)
        max_speedup = max(s[1] for s in comparisons)
        best_test = max(comparisons, key=lambda x: x[1])
        
        print(f"\n✨ Average Speedup: {avg_speedup:.2f}x")
        print(f"🚀 Maximum Speedup: {max_speedup:.2f}x ({best_test[0]})")
        print(f"\nAll improvements:")
        for test_name, speedup, improvement in comparisons:
            print(f"  • {test_name:20s}: {speedup:.2f}x ({improvement:+.1f}%)")
    
    print("\n" + "=" * 70)
    
    return comparisons


def main():
    """Main comparison workflow."""
    import argparse as ap
    
    parser = ap.ArgumentParser(description="Compare RAP performance between original and optimized versions")
    parser.add_argument("--baseline", type=str, help="Path to baseline benchmark results JSON file")
    parser.add_argument("--skip-clone", action="store_true", help="Skip cloning original repo")
    args_cli = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("RAP PERFORMANCE COMPARISON TOOL")
    print("=" * 70)
    
    # Paths
    optimized_repo = Path("/home/ubuntu/RAP")
    original_repo = Path("/home/ubuntu/RAP_original")
    model_path = Path("/home/ubuntu/RAP/output/Cambridge/KingsCollege")
    colmap_path = Path("/home/ubuntu/RAP/data/Cambridge/KingsCollege/colmap")
    
    # Check if optimized results already exist
    optimized_results = optimized_repo / "output" / "Cambridge" / "KingsCollege" / "benchmark_results.json"
    
    if not optimized_results.exists():
        print(f"\n⚠ Optimized benchmark results not found at {optimized_results}")
        print("Running benchmark on optimized version...")
        optimized_results = run_benchmark_on_repo(
            optimized_repo,
            model_path,
            colmap_path,
            "optimized"
        )
        if not optimized_results:
            print("✗ Failed to generate optimized benchmark")
            return 1
    
    print(f"\n✓ Using optimized results: {optimized_results}")
    
    # Check if baseline file is provided
    if args_cli.baseline:
        baseline_file = args_cli.baseline
        if not Path(baseline_file).exists():
            print(f"✗ Baseline file not found: {baseline_file}")
            return 1
        print(f"✓ Using provided baseline: {baseline_file}")
    else:
        # Clone and setup original repo
        if args_cli.skip_clone:
            print("\n⚠ Skipping clone (--skip-clone flag set)")
            print("Please provide baseline results with --baseline flag")
            return 1
            
        original_path = clone_original_repo(str(original_repo))
        if not original_path:
            print("\n⚠ Could not clone original repo.")
            print("Please provide baseline results with --baseline flag")
            return 1
        
        if not setup_original_repo(original_path):
            print("\n⚠ Could not set up original repo.")
            print("You can manually run the benchmark on the original repo:")
            print(f"  1. cd {original_path}")
            print(f"  2. source venv/bin/activate")
            print(f"  3. python benchmark_speed.py -s {colmap_path} -m {model_path} --iteration 30000 --output baseline_output")
            print("\nThen run this script again with --baseline flag:")
            print(f"   python compare_performance.py --baseline baseline_output/benchmark_results.json")
            return 1
        
        # Run benchmark on original repo
        print("\n" + "=" * 70)
        print("RUNNING BASELINE BENCHMARK ON ORIGINAL REPO")
        print("=" * 70)
        
        baseline_file = run_benchmark_on_repo(
            original_path,
            model_path,
            colmap_path,
            "baseline"
        )
        
        if not baseline_file:
            print("\n⚠ Could not run baseline benchmark")
            return 1
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARING RESULTS")
    print("=" * 70)
    
    comparisons = compare_benchmarks(baseline_file, str(optimized_results))
    
    # Save comparison report
    comparison_report = optimized_repo / "performance_comparison.txt"
    with open(comparison_report, 'w') as f:
        f.write("RAP Performance Comparison Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Baseline: {baseline_file}\n")
        f.write(f"Optimized: {optimized_results}\n\n")
        if comparisons:
            f.write("Speedup Summary:\n")
            for test_name, speedup, improvement in comparisons:
                f.write(f"  {test_name}: {speedup:.2f}x ({improvement:+.1f}%)\n")
    
    print(f"\n✓ Comparison report saved to: {comparison_report}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

