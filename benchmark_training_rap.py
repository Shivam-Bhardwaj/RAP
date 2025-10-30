#!/usr/bin/env python3
"""
Training performance benchmark for RAP-ID models.

This script benchmarks training speed, memory usage, and convergence metrics
for baseline RAP, UAAS, Probabilistic, and Semantic models.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
from utils.cameras import CamParams
from utils.general_utils import fix_seed

# Import trainers
from rap import BaseTrainer, RVSWithDiscriminatorTrainer
from uaas.trainer import UAASTrainer
from probabilistic.trainer import ProbabilisticTrainer
from semantic.trainer import SemanticTrainer


def benchmark_training_performance(trainer, num_epochs: int = 1, num_batches: Optional[int] = None):
    """
    Benchmark training performance for a trainer.
    
    Args:
        trainer: Trainer instance (BaseTrainer or subclass)
        num_epochs: Number of epochs to benchmark
        num_batches: Optional limit on number of batches per epoch
        
    Returns:
        Dictionary with training performance metrics
    """
    print(f"\nBenchmarking training for {trainer.__class__.__name__}...")
    print(f"Dataset size: {trainer.dset_size}")
    print(f"Batches per epoch: {trainer.n_iters}")
    
    device = trainer.args.device
    
    # Warmup
    print("\nWarming up...")
    trainer.model.train()
    warmup_batches = min(5, trainer.n_iters)
    for i in range(warmup_batches):
        batch_idx = np.random.randint(0, trainer.dset_size - trainer.args.batch_size)
        batch_indices = np.arange(batch_idx, batch_idx + trainer.args.batch_size)
        
        imgs_batch = trainer.imgs_normed[batch_indices].to(device, non_blocking=True)
        poses_batch = trainer.poses[batch_indices].reshape(trainer.args.batch_size, 12).to(device, torch.float, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=trainer.args.amp, dtype=trainer.args.amp_dtype):
            if isinstance(trainer, UAASTrainer):
                (features_target, features_rendered), (poses_predicted, log_var_predicted) = (
                    trainer.model(torch.cat([imgs_batch, imgs_batch]), return_feature=True))
            elif isinstance(trainer, ProbabilisticTrainer):
                _ = trainer.model(imgs_batch)
            elif isinstance(trainer, SemanticTrainer):
                _ = trainer.model(imgs_batch)
            else:
                _, poses_predicted = trainer.model(imgs_batch, return_feature=False)
        
        trainer.optimizer_model.zero_grad()
        # Simple forward pass for warmup
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Actual benchmark
    epoch_times = []
    batch_times = []
    memory_usage = []
    loss_values = []
    
    total_batches = 0
    max_batches = num_batches if num_batches else trainer.n_iters
    
    print(f"\nBenchmarking {num_epochs} epoch(s), {max_batches} batches per epoch...")
    
    for epoch in range(num_epochs):
        trainer.model.train()
        if trainer.args.freeze_batch_norm:
            trainer.model = trainer.freeze_bn_layer_train(trainer.model)
        
        epoch_start = time.time()
        selected_indexes = np.random.choice(trainer.dset_size, size=[trainer.dset_size], replace=False)
        
        i_batch = 0
        epoch_batches = 0
        
        for batch_idx in tqdm(range(min(max_batches, trainer.n_iters)), desc=f"Epoch {epoch+1}"):
            if i_batch + trainer.args.batch_size > trainer.dset_size:
                break
            
            batch_indices = selected_indexes[i_batch:i_batch + trainer.args.batch_size]
            i_batch += trainer.args.batch_size
            
            batch_start = time.time()
            
            # Prepare batch
            imgs_normed_batch = trainer.imgs_normed[batch_indices].to(device, non_blocking=True)
            poses_batch = trainer.poses[batch_indices].reshape(trainer.args.batch_size, 12).to(device, torch.float, non_blocking=True)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Forward pass
            forward_start = time.time()
            with torch.cuda.amp.autocast(enabled=trainer.args.amp, dtype=trainer.args.amp_dtype):
                # Model-specific forward pass
                if isinstance(trainer, UAASTrainer):
                    imgs_rendered_batch = trainer.imgs_rendered[batch_indices].to(device, non_blocking=True)
                    (features_target, features_rendered), (poses_predicted, log_var_predicted) = (
                        trainer.model(torch.cat([imgs_normed_batch, imgs_rendered_batch]), return_feature=True))
                    # Simplified loss for benchmarking
                    loss = trainer.pose_loss(poses_predicted, torch.cat([poses_batch, poses_batch]))
                elif isinstance(trainer, ProbabilisticTrainer):
                    mixture_dist = trainer.model(imgs_normed_batch)
                    loss = trainer.criterion(mixture_dist, poses_batch)
                elif isinstance(trainer, SemanticTrainer):
                    _, poses_predicted = trainer.model(imgs_normed_batch, return_feature=False)
                    loss = trainer.pose_loss(poses_predicted, poses_batch)
                else:
                    # Base trainer
                    _, poses_predicted = trainer.model(imgs_normed_batch, return_feature=False)
                    loss = trainer.pose_loss(poses_predicted, poses_batch)
            
            forward_time = time.time() - forward_start
            
            # Backward pass
            backward_start = time.time()
            trainer.optimizer_model.zero_grad(set_to_none=True)
            trainer.scaler_model.scale(loss).backward()
            trainer.scaler_model.step(trainer.optimizer_model)
            trainer.scaler_model.update()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            backward_time = time.time() - backward_start
            batch_time = time.time() - batch_start
            
            batch_times.append(batch_time)
            loss_values.append(loss.item())
            total_batches += 1
            
            # Track memory
            if torch.cuda.is_available() and epoch_batches % 10 == 0:
                memory_usage.append({
                    'epoch': epoch,
                    'batch': epoch_batches,
                    'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
                    'reserved_gb': torch.cuda.memory_reserved(0) / 1e9
                })
            
            epoch_batches += 1
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
    
    # Compile results
    results = {
        'training': {
            'num_epochs': num_epochs,
            'total_batches': total_batches,
            'total_time': sum(epoch_times),
            'avg_epoch_time': np.mean(epoch_times),
            'avg_batch_time': np.mean(batch_times),
            'median_batch_time': np.median(batch_times),
            'batches_per_second': total_batches / sum(epoch_times) if sum(epoch_times) > 0 else 0,
            'estimated_30k_iter_time_hours': (np.mean(batch_times) * 30000) / 3600,
        },
        'batch_times': {
            'min': np.min(batch_times),
            'max': np.max(batch_times),
            'mean': np.mean(batch_times),
            'median': np.median(batch_times),
            'std': np.std(batch_times),
            'p95': np.percentile(batch_times, 95),
            'p99': np.percentile(batch_times, 99),
        },
        'loss': {
            'initial': loss_values[0] if loss_values else None,
            'final': loss_values[-1] if loss_values else None,
            'mean': np.mean(loss_values),
            'std': np.std(loss_values),
        }
    }
    
    if memory_usage:
        avg_memory = sum(m['allocated_gb'] for m in memory_usage) / len(memory_usage)
        max_memory = max(m['allocated_gb'] for m in memory_usage)
        results['memory'] = {
            'avg_allocated_gb': avg_memory,
            'max_allocated_gb': max_memory,
            'samples': memory_usage[:10]  # Store first 10 samples
        }
    
    return results


def create_trainer(model_type: str, args):
    """
    Create appropriate trainer instance.
    
    Args:
        model_type: Type of model ('baseline', 'uaas', 'probabilistic', 'semantic')
        args: Configuration arguments
        
    Returns:
        Trainer instance
    """
    if model_type == 'baseline':
        from rap import BaseTrainer
        return BaseTrainer(args)
    elif model_type == 'uaas':
        return UAASTrainer(args)
    elif model_type == 'probabilistic':
        return ProbabilisticTrainer(args)
    elif model_type == 'semantic':
        return SemanticTrainer(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_training_benchmark(args):
    """
    Run training benchmark for specified model type.
    
    Args:
        args: Configuration arguments with benchmark parameters
    """
    print("=" * 70)
    print(f"RAP-ID Training Performance Benchmark: {args.model_type.upper()}")
    print("=" * 70)
    
    device = torch.device(args.device)
    fix_seed()
    
    # Create trainer
    print(f"\nInitializing {args.model_type} trainer...")
    trainer = create_trainer(args.model_type, args)
    
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Benchmark training
    num_epochs = getattr(args, 'benchmark_epochs', 1)
    num_batches = getattr(args, 'benchmark_batches', None)
    
    results = benchmark_training_performance(trainer, num_epochs=num_epochs, num_batches=num_batches)
    
    # Print results
    print("\n" + "=" * 70)
    print("TRAINING BENCHMARK RESULTS")
    print("=" * 70)
    
    training = results['training']
    print(f"\nTraining Performance:")
    print(f"  Total Batches:        {training['total_batches']}")
    print(f"  Total Time:           {training['total_time']:.2f} s")
    print(f"  Avg Batch Time:       {training['avg_batch_time']*1000:.2f} ms")
    print(f"  Median Batch Time:    {training['median_batch_time']*1000:.2f} ms")
    print(f"  Batches per Second:   {training['batches_per_second']:.2f}")
    print(f"  Estimated 30k iter:  {training['estimated_30k_iter_time_hours']:.2f} hours")
    
    if 'memory' in results:
        print(f"\nMemory Usage:")
        print(f"  Avg GPU Memory:       {results['memory']['avg_allocated_gb']:.2f} GB")
        print(f"  Max GPU Memory:       {results['memory']['max_allocated_gb']:.2f} GB")
    
    print(f"\nLoss Statistics:")
    print(f"  Initial Loss:         {results['loss']['initial']:.4f}")
    print(f"  Final Loss:           {results['loss']['final']:.4f}")
    print(f"  Mean Loss:            {results['loss']['mean']:.4f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(trainer.logdir) if hasattr(trainer, 'logdir') else Path('.')
    
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"training_benchmark_{args.model_type}.json"
    
    results['metadata'] = {
        'model_type': args.model_type,
        'dataset_type': args.dataset_type,
        'batch_size': args.batch_size,
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'amp_enabled': args.amp,
        'compile_enabled': args.compile_model,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark training performance for RAP-ID models")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["baseline", "uaas", "probabilistic", "semantic"],
                       help="Type of model to benchmark")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for benchmark results")
    parser.add_argument("--benchmark_epochs", type=int, default=1,
                       help="Number of epochs to benchmark")
    parser.add_argument("--benchmark_batches", type=int, default=None,
                       help="Number of batches per epoch to benchmark (None = all)")
    parser.add_argument("--num_semantic_classes", type=int, default=19,
                       help="Number of semantic classes for semantic model")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    run_training_benchmark(args)


if __name__ == "__main__":
    main()

