#!/usr/bin/env python3
"""
Performance Comparison Script
Measures speedup from optimizations
"""
import json
import sys
from pathlib import Path

def compare_profiles(before_file, after_file):
    """Compare two profiling JSON files"""
    with open(before_file) as f:
        before = json.load(f)
    with open(after_file) as f:
        after = json.load(f)
    
    print("="*70)
    print("Performance Comparison: Before vs After Optimizations")
    print("="*70)
    
    before_summary = before.get('summary', {})
    after_summary = after.get('summary', {})
    
    print(f"\n{'Operation':<35} {'Before (s)':<12} {'After (s)':<12} {'Speedup':<10} {'Improvement':<12}")
    print("-"*70)
    
    total_before = sum(s['total'] for s in before_summary.values())
    total_after = sum(s['total'] for s in after_summary.values())
    overall_speedup = total_before / total_after if total_after > 0 else 1.0
    
    for key in sorted(set(before_summary.keys()) | set(after_summary.keys())):
        if key in before_summary and key in after_summary:
            before_mean = before_summary[key]['mean']
            after_mean = after_summary[key]['mean']
            speedup = before_mean / after_mean if after_mean > 0 else 1.0
            improvement = (speedup - 1) * 100
            
            print(f"{key:<35} {before_mean:<12.4f} {after_mean:<12.4f} {speedup:<10.2f}x {improvement:<11.1f}%")
    
    print("-"*70)
    print(f"{'TOTAL':<35} {total_before:<12.4f} {total_after:<12.4f} {overall_speedup:<10.2f}x {(overall_speedup-1)*100:<11.1f}%")
    print("="*70)
    
    # Memory comparison
    if 'memory_stats' in before and 'memory_stats' in after:
        before_peak = max(s.get('peak_memory', 0) for s in before['memory_stats']) if before['memory_stats'] else 0
        after_peak = max(s.get('peak_memory', 0) for s in after['memory_stats']) if after['memory_stats'] else 0
        
        print(f"\nPeak GPU Memory:")
        print(f"  Before: {before_peak:.2f} GB")
        print(f"  After:  {after_peak:.2f} GB")
        print(f"  Change: {(after_peak - before_peak):+.2f} GB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_profiles.py <before.json> <after.json>")
        sys.exit(1)
    
    compare_profiles(sys.argv[1], sys.argv[2])

