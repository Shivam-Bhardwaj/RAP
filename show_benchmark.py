#!/usr/bin/env python3
"""
Quick summary script to display benchmark results in a readable format.
"""

import json
import sys
from pathlib import Path

def print_benchmark_summary(results_file):
    """Print a formatted summary of benchmark results."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "=" * 70)
    print("RENDERING PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 70)
    
    # Metadata
    if 'metadata' in results:
        meta = results['metadata']
        print(f"\nConfiguration:")
        print(f"  Model Path:      {meta.get('model_path', 'N/A')}")
        print(f"  Checkpoint:      Iteration {meta.get('iteration', 'N/A')}")
        print(f"  Device:          {meta.get('gpu_name', meta.get('device', 'N/A'))}")
        print(f"  Timestamp:       {meta.get('timestamp', 'N/A')}")
    
    # Model stats
    if 'model_stats' in results:
        stats = results['model_stats']
        print(f"\nModel Statistics:")
        print(f"  Number of Gaussians: {stats.get('num_gaussians', 0):,}")
        if stats.get('sh_degree'):
            print(f"  SH Degree:           {stats['sh_degree']}")
    
    # Performance metrics
    print(f"\n{'Performance Metrics':^70}")
    print("-" * 70)
    
    if 'test_set' in results:
        test = results['test_set']
        print(f"Test Set Rendering:")
        print(f"  Images:         {test.get('num_images', 0)}")
        print(f"  FPS:            {test.get('fps', 0):.2f}")
        print(f"  Time/Image:     {test.get('time_per_image', 0)*1000:.2f} ms")
        print(f"  Total Time:     {test.get('total_time', 0):.3f} s")
    
    if 'train_set' in results:
        train = results['train_set']
        print(f"\nTrain Set Rendering:")
        print(f"  Images:         {train.get('num_images', 0)}")
        print(f"  FPS:            {train.get('fps', 0):.2f}")
        print(f"  Time/Image:     {train.get('time_per_image', 0)*1000:.2f} ms")
        print(f"  Total Time:     {train.get('total_time', 0):.3f} s")
    
    if 'standard_test' in results:
        std = results['standard_test']
        print(f"\nStandard Speed Test (800x800):")
        print(f"  FPS:            {std.get('fps', 0):.2f}")
        print(f"  Time/Image:     {std.get('time_per_image', 0)*1000:.2f} ms")
    
    # GPU memory
    if 'gpu_memory' in results:
        mem = results['gpu_memory']
        print(f"\nGPU Memory Usage:")
        print(f"  Allocated:      {mem.get('allocated_gb', 0):.2f} GB")
        print(f"  Reserved:       {mem.get('reserved_gb', 0):.2f} GB")
    
    print("\n" + "=" * 70)
    
    # Key metrics highlight
    print("\n📊 KEY METRICS:")
    if 'test_set' in results:
        print(f"   • Test Rendering Speed:  {results['test_set'].get('fps', 0):.2f} FPS")
    if 'standard_test' in results:
        print(f"   • Standard Test Speed:   {results['standard_test'].get('fps', 0):.2f} FPS")
    if 'gpu_memory' in results:
        print(f"   • GPU Memory Usage:      {results['gpu_memory'].get('allocated_gb', 0):.2f} GB")
    
    print("\n💡 To compare with optimized version:")
    print(f"   python benchmark_speed.py --compare baseline.json optimized.json")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_benchmark.py <benchmark_results.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    if not Path(results_file).exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)
    
    print_benchmark_summary(results_file)

