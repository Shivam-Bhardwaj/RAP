#!/usr/bin/env python3
"""
Benchmark script to measure rendering speed and performance metrics.
This script helps measure the speed boost from optimization efforts.
"""

import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, get_combined_args
import arguments.args_init as args_init
from models.gs.gaussian_model import GaussianModel
from utils.scene import Scene
from render import test_rendering_speed, render_set
from utils.general_utils import fix_seed

# Import new models
from RAP.uaas.uaas_rap_net import UAASRAPNet
from RAP.probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from RAP.semantic.semantic_rap_net import SemanticRAPNet


def benchmark_rendering_speed(args, num_warmup=5, num_iterations=100):
    """Benchmark rendering speed with detailed metrics."""
    print("\n" + "=" * 60)
    print("RENDERING SPEED BENCHMARK")
    print("=" * 60)
    
    device = torch.device(args.render_device)
    fix_seed()
    
    # Load model
    print(f"\nLoading model from iteration {args.iteration}...")
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float, device=device)
    gaussians.set_eval(True)
    
    # Use a subset of cameras for benchmarking
    test_cameras = scene.test_cameras[:min(num_iterations, len(scene.test_cameras))]
    train_cameras = scene.train_cameras[:min(num_iterations, len(scene.train_cameras))]
    
    print(f"Benchmarking with {len(test_cameras)} test cameras and {len(train_cameras)} train cameras")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Gaussians: {gaussians._xyz.shape[0]:,}")
    
    results = {}
    
    # Warmup
    print(f"\nWarming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for i in range(num_warmup):
            _ = gaussians.render(test_cameras[0], args, background)["render"]
    
    torch.cuda.synchronize()
    
    # Benchmark test set rendering
    print(f"\nBenchmarking test set rendering ({len(test_cameras)} images)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for idx, view in enumerate(tqdm(test_cameras, desc="Rendering test")):
        with torch.no_grad():
            _ = gaussians.render(view, args, background)["render"]
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    test_render_time = end_time - start_time
    test_fps = len(test_cameras) / test_render_time
    test_time_per_image = test_render_time / len(test_cameras)
    
    results['test_set'] = {
        'num_images': len(test_cameras),
        'total_time': test_render_time,
        'fps': test_fps,
        'time_per_image': test_time_per_image
    }
    
    print(f"  Test set: {test_fps:.2f} FPS ({test_time_per_image*1000:.2f} ms/image)")
    
    # Benchmark train set rendering
    print(f"\nBenchmarking train set rendering ({len(train_cameras)} images)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for idx, view in enumerate(tqdm(train_cameras, desc="Rendering train")):
        with torch.no_grad():
            _ = gaussians.render(view, args, background)["render"]
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    train_render_time = end_time - start_time
    train_fps = len(train_cameras) / train_render_time
    train_time_per_image = train_render_time / len(train_cameras)
    
    results['train_set'] = {
        'num_images': len(train_cameras),
        'total_time': train_render_time,
        'fps': train_fps,
        'time_per_image': train_time_per_image
    }
    
    print(f"  Train set: {train_fps:.2f} FPS ({train_time_per_image*1000:.2f} ms/image)")
    
    # Use the built-in speed test (800x800 resolution)
    print(f"\nRunning standard speed test (800x800 resolution)...")
    avg_speed = test_rendering_speed(scene.train_cameras, gaussians, args, background, use_cache=False)
    results['standard_test'] = {
        'time_per_image': avg_speed,
        'fps': 1.0 / avg_speed if avg_speed > 0 else 0
    }
    print(f"  Standard test: {1.0/avg_speed:.2f} FPS ({avg_speed*1000:.2f} ms/image)")
    
    # GPU memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        results['gpu_memory'] = {
            'allocated_gb': memory_allocated,
            'reserved_gb': memory_reserved
        }
        print(f"\nGPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")
    
    # Model statistics
    results['model_stats'] = {
        'num_gaussians': int(gaussians._xyz.shape[0]),
        'sh_degree': args.sh_degree if hasattr(args, 'sh_degree') else None
    }
    
    return results


def benchmark_pose_accuracy(args, model, data_loader):
    """Benchmark pose accuracy."""
    print(f"Benchmarking pose accuracy for {model.__class__.__name__}...")
    # This function would call the existing eval_model utility
    # from utils.eval_utils import eval_model
    # and would need to be adapted for each model type (e.g. probabilistic)
    pass


def save_benchmark_results(results, output_path):
    """Save benchmark results to JSON file."""
    output_file = Path(output_path) / "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


def compare_results(results_file1, results_file2):
    """Compare two benchmark result files."""
    with open(results_file1, 'r') as f:
        baseline = json.load(f)
    
    with open(results_file2, 'r') as f:
        optimized = json.load(f)
    
    print("\n" + "=" * 60)
    print("SPEED COMPARISON")
    print("=" * 60)
    
    for key in ['test_set', 'train_set', 'standard_test']:
        if key in baseline and key in optimized:
            print(f"\n{key.upper().replace('_', ' ')}:")
            base_fps = baseline[key].get('fps', 1.0 / baseline[key].get('time_per_image', 1.0))
            opt_fps = optimized[key].get('fps', 1.0 / optimized[key].get('time_per_image', 1.0))
            
            speedup = opt_fps / base_fps if base_fps > 0 else 0
            improvement = (speedup - 1.0) * 100
            
            print(f"  Baseline:  {base_fps:.2f} FPS")
            print(f"  Optimized: {opt_fps:.2f} FPS")
            print(f"  Speedup:   {speedup:.2f}x ({improvement:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark rendering speed and pose accuracy")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--iteration", default=-1, type=int, help="Checkpoint iteration to load")
    parser.add_argument("--num_iterations", default=100, type=int, help="Number of images to benchmark")
    parser.add_argument("--compare", nargs=2, metavar=('BASELINE', 'OPTIMIZED'), 
                       help="Compare two benchmark result files")
    parser.add_argument("--output", default=None, help="Output directory for results")
    parser.add_argument("--benchmark_pose", action="store_true", help="Benchmark pose accuracy")
    parser.add_argument("--model_type", type=str, default="baseline", 
                        choices=["baseline", "uaas", "probabilistic", "semantic"],
                        help="Type of model to benchmark.")

    args = get_combined_args(parser)
    args = args_init.argument_init(args)
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = benchmark_rendering_speed(args, num_iterations=args.num_iterations)
    
    # Optionally benchmark pose accuracy
    if args.benchmark_pose:
        model_map = {
            "baseline": None, # Should load the original RAPNet
            "uaas": UAASRAPNet(args),
            "probabilistic": ProbabilisticRAPNet(args),
            "semantic": SemanticRAPNet(args, num_semantic_classes=19) # Example number of classes
        }
        model = model_map[args.model_type]
        if model:
            # Here you would load the trained model weights
            # model.load_state_dict(...)
            # And then run the benchmark
            # benchmark_pose_accuracy(args, model, data_loader)
            print(f"Pose accuracy benchmarking for {args.model_type} is a placeholder.")
        else:
            print("Running baseline pose accuracy benchmark (placeholder).")


    # Add metadata
    results['metadata'] = {
        'model_path': args.model_path,
        'iteration': args.iteration,
        'device': str(args.render_device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    save_benchmark_results(results, output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Test set FPS:    {results['test_set']['fps']:.2f}")
    print(f"Train set FPS:   {results['train_set']['fps']:.2f}")
    print(f"Standard test:   {results['standard_test']['fps']:.2f} FPS")
    if 'gpu_memory' in results:
        print(f"GPU Memory:      {results['gpu_memory']['allocated_gb']:.2f} GB")
    print(f"Gaussians:       {results['model_stats']['num_gaussians']:,}")


if __name__ == "__main__":
    main()

