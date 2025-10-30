#!/usr/bin/env python3
"""
Performance Profiler for RAP
Profiles training and inference to identify bottlenecks
"""
import time
import torch
import numpy as np
from contextlib import contextmanager
from collections import defaultdict

class PerformanceProfiler:
    def __init__(self, device='cuda', enable_cuda_sync=True):
        self.device = device
        self.enable_cuda_sync = enable_cuda_sync and torch.cuda.is_available()
        self.timings = defaultdict(list)
        self.memory_stats = []
        self.enabled = True
        
    @contextmanager
    def profile(self, name):
        """Context manager for timing operations"""
        if not self.enabled:
            yield
            return
            
        if self.enable_cuda_sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        start_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        start_mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        
        yield
        
        if self.enable_cuda_sync:
            torch.cuda.synchronize()
        end = time.perf_counter()
        end_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        end_mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        
        elapsed = end - start
        mem_used = end_mem - start_mem
        mem_reserved = end_mem_reserved - start_mem_reserved
        
        self.timings[name].append(elapsed)
        if torch.cuda.is_available():
            self.memory_stats.append({
                'name': name,
                'time': elapsed,
                'memory_allocated': mem_used,
                'memory_reserved': mem_reserved,
                'peak_memory': torch.cuda.max_memory_allocated() / 1024**3
            })
    
    def profile_function(self, func, name=None):
        """Decorator for profiling functions"""
        if name is None:
            name = func.__name__
            
        def wrapper(*args, **kwargs):
            with self.profile(name):
                return func(*args, **kwargs)
        return wrapper
    
    def get_summary(self):
        """Get profiling summary"""
        summary = {}
        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'total': np.sum(times),
                    'count': len(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'median': np.median(times)
                }
        return summary
    
    def print_summary(self, sort_by='total'):
        """Print profiling summary"""
        print("\n" + "="*70)
        print("Performance Profiling Summary")
        print("="*70)
        
        summary = self.get_summary()
        if not summary:
            print("No profiling data collected.")
            return
        
        # Sort by specified metric
        if sort_by == 'total':
            sorted_items = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        elif sort_by == 'mean':
            sorted_items = sorted(summary.items(), key=lambda x: x[1]['mean'], reverse=True)
        else:
            sorted_items = list(summary.items())
        
        print(f"\n{'Operation':<35} {'Mean (s)':<12} {'Total (s)':<12} {'Calls':<8} {'%':<8}")
        print("-"*70)
        
        total_time = sum(s['total'] for s in summary.values())
        
        for name, stats in sorted_items:
            percentage = (stats['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<35} {stats['mean']:<12.4f} {stats['total']:<12.4f} {stats['count']:<8} {percentage:<7.1f}%")
        
        print("-"*70)
        print(f"{'TOTAL':<35} {'':<12} {total_time:<12.4f} {'':<8} {'100.0%':<8}")
        
        # Memory summary
        if self.memory_stats and torch.cuda.is_available():
            print("\n" + "="*70)
            print("Memory Usage Summary")
            print("="*70)
            peak_mem = max(s['peak_memory'] for s in self.memory_stats)
            current_mem = torch.cuda.memory_allocated() / 1024**3
            reserved_mem = torch.cuda.memory_reserved() / 1024**3
            print(f"Peak GPU Memory Allocated: {peak_mem:.2f} GB")
            print(f"Current GPU Memory Allocated: {current_mem:.2f} GB")
            print(f"Current GPU Memory Reserved: {reserved_mem:.2f} GB")
            
        print("="*70 + "\n")
    
    def export_json(self, filepath):
        """Export profiling data to JSON"""
        import json
        data = {
            'summary': self.get_summary(),
            'memory_stats': self.memory_stats,
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Profiling data exported to {filepath}")
    
    def reset(self):
        """Reset profiling data"""
        self.timings.clear()
        self.memory_stats.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

# Global profiler instance
_profiler = None

def get_profiler():
    """Get or create global profiler instance"""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler

def profile(name):
    """Convenience function for profiling"""
    return get_profiler().profile(name)

def print_profile_summary(sort_by='total'):
    """Print profiling summary"""
    get_profiler().print_summary(sort_by=sort_by)

def reset_profile():
    """Reset profiling data"""
    get_profiler().reset()

