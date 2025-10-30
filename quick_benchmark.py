#!/usr/bin/env python3
"""
Quick Performance Benchmark Script
Simplified version for quick performance checks
"""
import time
import torch
import numpy as np
from contextlib import contextmanager

@contextmanager
def timer(name, device='cuda'):
    """Simple timer context manager"""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{name}: {elapsed*1000:.2f} ms")


def benchmark_transfers(device='cuda', iterations=100):
    """Benchmark CPU-GPU transfers"""
    print("\n" + "="*60)
    print("CPU-GPU Transfer Benchmark")
    print("="*60)
    
    data = torch.randn(8, 3, 240, 427)
    
    # Blocking
    times_blocking = []
    for _ in range(iterations):
        start = time.perf_counter()
        data_gpu = data.to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_blocking.append(time.perf_counter() - start)
        del data_gpu
    
    # Non-blocking
    times_non_blocking = []
    for _ in range(iterations):
        start = time.perf_counter()
        data_gpu = data.to(device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_non_blocking.append(time.perf_counter() - start)
        del data_gpu
    
    mean_blocking = np.mean(times_blocking) * 1000
    mean_non_blocking = np.mean(times_non_blocking) * 1000
    speedup = mean_blocking / mean_non_blocking
    
    print(f"\nBlocking transfer:     {mean_blocking:.4f} ms")
    print(f"Non-blocking transfer: {mean_non_blocking:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {(speedup-1)*100:.1f}%")
    

def benchmark_stack_operations(device='cuda', num_tensors=100):
    """Benchmark stacking operations"""
    print("\n" + "="*60)
    print("Stacking Operations Benchmark")
    print("="*60)
    
    tensors = [torch.randn(3, 240, 427, device=device) for _ in range(num_tensors)]
    
    # Method 1: Move each to CPU, then stack
    start = time.perf_counter()
    cpu_tensors = [t.cpu() for t in tensors]
    stacked1 = torch.stack(cpu_tensors)
    time1 = time.perf_counter() - start
    
    # Method 2: Stack on GPU, then move once
    start = time.perf_counter()
    stacked2 = torch.stack(tensors)
    stacked2 = stacked2.cpu()
    time2 = time.perf_counter() - start
    
    print(f"\nStack after CPU transfer: {time1*1000:.2f} ms")
    print(f"Stack then CPU transfer:  {time2*1000:.2f} ms")
    print(f"Speedup: {time1/time2:.2f}x")
    print(f"Improvement: {(time1/time2-1)*100:.1f}%")


def benchmark_batch_processing(device='cuda', batch_size=8, num_iterations=100):
    """Benchmark batch processing"""
    print("\n" + "="*60)
    print("Batch Processing Benchmark")
    print("="*60)
    
    data = torch.randn(batch_size, 3, 240, 427, device=device)
    
    # Process one by one
    times_sequential = []
    for i in range(num_iterations):
        start = time.perf_counter()
        for j in range(batch_size):
            _ = data[j:j+1].sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_sequential.append(time.perf_counter() - start)
    
    # Process as batch
    times_batch = []
    for i in range(num_iterations):
        start = time.perf_counter()
        _ = data.sum()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_batch.append(time.perf_counter() - start)
    
    mean_sequential = np.mean(times_sequential) * 1000
    mean_batch = np.mean(times_batch) * 1000
    speedup = mean_sequential / mean_batch
    
    print(f"\nSequential processing: {mean_sequential:.2f} ms")
    print(f"Batch processing:      {mean_batch:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Improvement: {(speedup-1)*100:.1f}%")


if __name__ == "__main__":
    print("RAP Performance Benchmarks")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    benchmark_transfers(device)
    benchmark_stack_operations(device)
    benchmark_batch_processing(device)
    
    print("\n" + "="*60)
    print("Benchmarks Complete!")
    print("="*60)

