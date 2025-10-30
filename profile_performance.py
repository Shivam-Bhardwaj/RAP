#!/usr/bin/env python3
"""
RAP Training Profiler
Profiles training loop to measure performance improvements
"""
import os
import sys
import time
import json
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.profiler import PerformanceProfiler
from arguments.options import config_parser
from arguments import ModelParams, OptimizationParams, get_combined_args
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from dataset_loaders.cambridge_scenes import Cambridge
from models.apr.rapnet import RAPNet
from utils.cameras import CamParams
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

torch.set_float32_matmul_precision('high')


def profile_training_step(model, profiler, args, device, batch_size=8, num_iterations=50):
    """Profile a single training step"""
    
    # Create dummy data
    rap_hw = (240, 427)  # Example resolution
    imgs_normed = torch.randn(batch_size, 3, rap_hw[0], rap_hw[1], device=device)
    poses = torch.randn(batch_size, 3, 4, device=device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device, enabled=args.amp)
    
    # Warmup
    for _ in range(5):
        with autocast(device, enabled=args.amp, dtype=args.amp_dtype):
            _, poses_pred = model(imgs_normed)
            loss = torch.nn.functional.mse_loss(poses_pred, poses.flatten(1))
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Profile iterations
    profiler.reset()
    print(f"\nProfiling {num_iterations} training iterations...")
    
    for i in range(num_iterations):
        with profiler.profile("data_prep"):
            imgs_batch = imgs_normed
            poses_batch = poses.flatten(1)
        
        with profiler.profile("forward_pass"):
            with autocast(device, enabled=args.amp, dtype=args.amp_dtype):
                _, poses_pred = model(imgs_batch)
        
        with profiler.profile("loss_computation"):
            loss = torch.nn.functional.mse_loss(poses_pred, poses_batch)
        
        with profiler.profile("backward_pass"):
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if i % 10 == 0:
            print(f"  Iteration {i}/{num_iterations}...")
    
    profiler.print_summary()


def profile_data_loading(dataset, profiler, batch_size=8, num_batches=50):
    """Profile data loading performance"""
    print(f"\nProfiling data loading ({num_batches} batches)...")
    
    profiler.reset()
    
    # Test different num_workers
    for num_workers in [1, 2, 4, 8]:
        print(f"\n  Testing num_workers={num_workers}...")
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        times = []
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            start = time.perf_counter()
            data, poses, _, _ = batch
            if torch.cuda.is_available():
                data = data.to('cuda', non_blocking=True)
                poses = poses.to('cuda', non_blocking=True)
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        if times:
            mean_time = np.mean(times)
            total_time = np.sum(times)
            print(f"    Mean time per batch: {mean_time*1000:.2f} ms")
            print(f"    Total time: {total_time:.2f} s")
            print(f"    Throughput: {num_batches/total_time:.2f} batches/s")


def profile_model_inference(model, profiler, device, batch_sizes=[1, 4, 8, 16], num_iterations=100):
    """Profile model inference at different batch sizes"""
    print(f"\nProfiling model inference...")
    
    rap_hw = (240, 427)
    
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        imgs = torch.randn(batch_size, 3, rap_hw[0], rap_hw[1], device=device)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(imgs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile
        profiler.reset()
        times = []
        
        for i in range(num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad(), autocast(device, enabled=True):
                _ = model(imgs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = batch_size / mean_time
        
        print(f"    Mean latency: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"    Throughput: {throughput:.2f} samples/s")


def compare_transfer_methods(data, device, num_iterations=100):
    """Compare blocking vs non-blocking transfers"""
    print(f"\nComparing transfer methods ({num_iterations} iterations)...")
    
    # Blocking
    times_blocking = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        data_gpu = data.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times_blocking.append(end - start)
        del data_gpu
    
    # Non-blocking
    times_non_blocking = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        data_gpu = data.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times_non_blocking.append(end - start)
        del data_gpu
    
    mean_blocking = np.mean(times_blocking)
    mean_non_blocking = np.mean(times_non_blocking)
    speedup = mean_blocking / mean_non_blocking if mean_non_blocking > 0 else 1.0
    
    print(f"  Blocking transfer:     {mean_blocking*1000:.4f} ms")
    print(f"  Non-blocking transfer: {mean_non_blocking*1000:.4f} ms")
    print(f"  Speedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description='Profile RAP training performance')
    parser.add_argument('-c', '--config', type=str, help='Config file path')
    parser.add_argument('-m', '--model_path', type=str, help='Path to 3DGS model')
    parser.add_argument('-d', '--datadir', type=str, help='Data directory')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations to profile')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--export', type=str, help='Export results to JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    
    profiler = PerformanceProfiler(device=device)
    
    # Initialize model
    print("\n" + "="*70)
    print("Initializing Model")
    print("="*70)
    
    # Parse config if provided
    if args.config:
        config_parser_obj = config_parser()
        model_params = ModelParams(config_parser_obj)
        opt_params = OptimizationParams(config_parser_obj)
        config_args = get_combined_args(config_parser_obj)
        config_args.config = args.config
        
        # Read config file
        with open(args.config, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    setattr(config_args, key, value)
        
        model_params.extract(config_args)
        opt_params.extract(config_args)
        args = config_args
    
    # Create model
    model = RAPNet(args).to(device)
    if hasattr(args, 'compile_model') and args.compile_model:
        model = torch.compile(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")
    
    # Profile components
    print("\n" + "="*70)
    print("Component Profiling")
    print("="*70)
    
    # 1. Model inference
    profile_model_inference(model, profiler, device, batch_sizes=[1, 4, 8, 16])
    
    # 2. Training step
    profile_training_step(model, profiler, args, device, batch_size=args.batch_size, num_iterations=args.iterations)
    
    # 3. Transfer comparison
    if torch.cuda.is_available():
        dummy_data = torch.randn(args.batch_size, 3, 240, 427)
        compare_transfer_methods(dummy_data, device)
    
    # Export results
    if args.export:
        profiler.export_json(args.export)
    
    print("\n" + "="*70)
    print("Profiling Complete!")
    print("="*70)
    print("\nTo measure improvements:")
    print("1. Run this script before optimizations")
    print("2. Apply optimizations")
    print("3. Run this script again")
    print("4. Compare the results")


if __name__ == "__main__":
    main()

