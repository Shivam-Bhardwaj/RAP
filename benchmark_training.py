#!/usr/bin/env python3
"""
Training time benchmark to measure actual training speed improvements.
This measures the real gains from optimizations during training loop.
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
from utils.general_utils import fix_seed


def benchmark_training_time(args, num_iterations=1000, warmup_iterations=10):
    """Benchmark actual training loop performance."""
    print("\n" + "=" * 70)
    print("TRAINING TIME BENCHMARK")
    print("=" * 70)
    
    device = torch.device(args.render_device)
    fix_seed()
    
    print(f"\nBenchmarking training for {num_iterations} iterations...")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Load model and scene
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    gaussians.training_setup(args)
    
    print(f"Train cameras: {len(scene.train_cameras)}")
    print(f"Gaussians: {gaussians._xyz.shape[0]:,}")
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float, device=device)
    
    # Setup optimizer
    optimizer = gaussians.optimizer
    
    results = {}
    
    # Warmup iterations
    print(f"\nWarming up ({warmup_iterations} iterations)...")
    gaussians.training_setup(args)
    for i in range(warmup_iterations):
        viewpoint_cam = scene.train_cameras[i % len(scene.train_cameras)]
        image = gaussians.render(viewpoint_cam, args, background)["render"]
        gt_image = viewpoint_cam.original_image.to(device)
        loss = (image - gt_image).abs().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"\nBenchmarking training loop ({num_iterations} iterations)...")
    
    iteration_times = []
    memory_usage = []
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for iteration in tqdm(range(num_iterations), desc="Training"):
        iter_start = time.time()
        
        # Select random camera
        viewpoint_cam = scene.train_cameras[iteration % len(scene.train_cameras)]
        
        # Render
        with torch.cuda.amp.autocast(enabled=args.fp16 if hasattr(args, 'fp16') else False):
            image = gaussians.render(viewpoint_cam, args, background)["render"]
            gt_image = viewpoint_cam.original_image.to(device)
            
            # Loss
            loss = (image - gt_image).abs().mean()
            
            # Backward
            loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Track memory
        if torch.cuda.is_available() and iteration % 10 == 0:
            memory_usage.append({
                'iteration': iteration,
                'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                'reserved_gb': torch.cuda.memory_reserved(0) / 1e9
            })
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_iter = total_time / num_iterations
    median_time_per_iter = sorted(iteration_times)[num_iterations // 2]
    
    results['training'] = {
        'num_iterations': num_iterations,
        'total_time': total_time,
        'avg_time_per_iter': avg_time_per_iter,
        'median_time_per_iter': median_time_per_iter,
        'iterations_per_second': num_iterations / total_time,
        'time_per_iter_ms': avg_time_per_iter * 1000
    }
    
    results['iteration_times'] = {
        'min': min(iteration_times),
        'max': max(iteration_times),
        'median': median_time_per_iter,
        'p95': sorted(iteration_times)[int(num_iterations * 0.95)],
        'p99': sorted(iteration_times)[int(num_iterations * 0.99)]
    }
    
    if memory_usage:
        avg_memory = sum(m['allocated_gb'] for m in memory_usage) / len(memory_usage)
        max_memory = max(m['allocated_gb'] for m in memory_usage)
        results['memory'] = {
            'avg_allocated_gb': avg_memory,
            'max_allocated_gb': max_memory,
            'samples': memory_usage
        }
    
    results['model_stats'] = {
        'num_gaussians': int(gaussians._xyz.shape[0]),
        'num_train_cameras': len(scene.train_cameras)
    }
    
    print(f"\n{'Results':^70}")
    print("-" * 70)
    print(f"Total Time:          {total_time:.2f} s")
    print(f"Time per Iteration:  {avg_time_per_iter*1000:.2f} ms")
    print(f"Iterations per Sec:  {num_iterations/total_time:.2f}")
    print(f"Median Time:         {median_time_per_iter*1000:.2f} ms")
    if memory_usage:
        print(f"Avg GPU Memory:      {avg_memory:.2f} GB")
        print(f"Max GPU Memory:      {max_memory:.2f} GB")
    
    return results


def save_training_results(results, output_path):
    """Save training benchmark results."""
    output_file = Path(output_path) / "training_benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")


def compare_training_results(baseline_file, optimized_file):
    """Compare two training benchmark results."""
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_file, 'r') as f:
        optimized = json.load(f)
    
    print("\n" + "=" * 70)
    print("TRAINING TIME COMPARISON")
    print("=" * 70)
    
    base_training = baseline['training']
    opt_training = optimized['training']
    
    base_time = base_training['avg_time_per_iter']
    opt_time = opt_training['avg_time_per_iter']
    speedup = base_time / opt_time if opt_time > 0 else 0
    improvement = (speedup - 1.0) * 100
    
    base_ips = base_training['iterations_per_second']
    opt_ips = opt_training['iterations_per_second']
    
    print(f"\nTraining Performance:")
    print(f"  Baseline:  {base_time*1000:.2f} ms/iter ({base_ips:.2f} iter/s)")
    print(f"  Optimized: {opt_time*1000:.2f} ms/iter ({opt_ips:.2f} iter/s)")
    print(f"  Speedup:   {speedup:.2f}x ({improvement:+.1f}%)")
    
    # Time to convergence estimate (30k iterations)
    base_convergence = (base_time * 30000) / 3600  # hours
    opt_convergence = (opt_time * 30000) / 3600
    time_saved = base_convergence - opt_convergence
    
    print(f"\nEstimated Time to Convergence (30k iterations):")
    print(f"  Baseline:  {base_convergence:.2f} hours")
    print(f"  Optimized: {opt_convergence:.2f} hours")
    print(f"  Time Saved: {time_saved:.2f} hours ({time_saved/base_convergence*100:.1f}% faster)")
    
    return {
        'speedup': speedup,
        'improvement_pct': improvement,
        'time_saved_hours': time_saved
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark training time")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--num_iterations", default=1000, type=int, 
                       help="Number of training iterations to benchmark")
    parser.add_argument("--warmup_iterations", default=10, type=int,
                       help="Number of warmup iterations")
    parser.add_argument("--compare", nargs=2, metavar=('BASELINE', 'OPTIMIZED'),
                       help="Compare two training benchmark result files")
    parser.add_argument("--output", default=None, help="Output directory for results")
    
    args = get_combined_args(parser)
    args = args_init.argument_init(args)
    
    if args.compare:
        compare_training_results(args.compare[0], args.compare[1])
        return
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model_path) if args.model_path else Path(".")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = benchmark_training_time(
        args, 
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations
    )
    
    # Add metadata
    results['metadata'] = {
        'model_path': str(args.model_path) if args.model_path else None,
        'num_iterations': args.num_iterations,
        'device': str(args.render_device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save results
    save_training_results(results, output_path)
    
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Avg Time per Iteration: {results['training']['time_per_iter_ms']:.2f} ms")
    print(f"Iterations per Second:   {results['training']['iterations_per_second']:.2f}")
    print(f"Estimated 30k iter time: {(results['training']['avg_time_per_iter'] * 30000 / 3600):.2f} hours")


if __name__ == "__main__":
    main()

