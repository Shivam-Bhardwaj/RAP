#!/usr/bin/env python3
"""
Comprehensive parallel benchmarking suite for RAP-ID.

This script benchmarks all RAP-ID extensions (UAAS, Probabilistic, Semantic)
against the baseline RAP model, running training and evaluation benchmarks
in parallel where possible.

Usage:
    python benchmark_comparison.py -c configs/config.txt -m /path/to/3dgs --parallel
"""
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import multiprocessing

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
import arguments.args_init as args_init
from utils.general_utils import fix_seed


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    model_type: str
    checkpoint_path: Optional[str] = None
    run_training: bool = True
    run_evaluation: bool = True
    num_epochs: int = 1
    num_batches: Optional[int] = None


@dataclass
class BenchmarkResults:
    """Results from a benchmark run."""
    model_type: str
    training_results: Optional[Dict] = None
    evaluation_results: Optional[Dict] = None
    inference_speed: Optional[Dict] = None
    timestamp: str = ""
    error: Optional[str] = None


def run_training_benchmark_worker(args_dict: dict) -> Dict:
    """
    Worker function for running training benchmark in parallel.
    
    Args:
        args_dict: Dictionary of arguments to pass to benchmark_training_rap.py
        
    Returns:
        Dictionary with training benchmark results
    """
    import subprocess
    import json
    from pathlib import Path
    
    model_type = args_dict['model_type']
    output_dir = args_dict.get('output', '/tmp')
    
    # Build command
    cmd = [
        sys.executable, 'benchmark_training_rap.py',
        '--model_type', model_type,
        '--output', output_dir,
        '--benchmark_epochs', str(args_dict.get('benchmark_epochs', 1)),
    ]
    
    # Add other arguments
    if 'config' in args_dict:
        cmd.extend(['-c', args_dict['config']])
    if 'datadir' in args_dict:
        cmd.extend(['-d', args_dict['datadir']])
    if 'model_path' in args_dict:
        cmd.extend(['-m', args_dict['model_path']])
    if 'benchmark_batches' in args_dict and args_dict['benchmark_batches']:
        cmd.extend(['--benchmark_batches', str(args_dict['benchmark_batches'])])
    
    # Run benchmark
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return {
                'model_type': model_type,
                'error': f"Training benchmark failed: {result.stderr}",
                'stdout': result.stdout
            }
        
        # Load results
        results_file = Path(output_dir) / f"training_benchmark_{model_type}.json"
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        else:
            return {
                'model_type': model_type,
                'error': f"Results file not found: {results_file}",
                'stdout': result.stdout
            }
    except subprocess.TimeoutExpired:
        return {
            'model_type': model_type,
            'error': "Training benchmark timed out after 1 hour"
        }
    except Exception as e:
        return {
            'model_type': model_type,
            'error': f"Training benchmark exception: {str(e)}"
        }


def run_evaluation_benchmark_worker(args_dict: dict) -> Dict:
    """
    Worker function for running evaluation benchmark in parallel.
    
    Args:
        args_dict: Dictionary of arguments to pass to benchmark_comprehensive.py
        
    Returns:
        Dictionary with evaluation benchmark results
    """
    import subprocess
    import json
    from pathlib import Path
    
    model_type = args_dict['model_type']
    output_dir = args_dict.get('output', '/tmp')
    
    # Build command
    cmd = [
        sys.executable, 'benchmark_comprehensive.py',
        '--model_type', model_type,
        '--output', output_dir,
        '--benchmark_speed',
    ]
    
    # Add other arguments
    if 'config' in args_dict:
        cmd.extend(['-c', args_dict['config']])
    if 'datadir' in args_dict:
        cmd.extend(['-d', args_dict['datadir']])
    if 'model_path' in args_dict:
        cmd.extend(['-m', args_dict['model_path']])
    if 'checkpoint_path' in args_dict and args_dict['checkpoint_path']:
        cmd.extend(['--checkpoint_path', args_dict['checkpoint_path']])
    
    # Run benchmark
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            return {
                'model_type': model_type,
                'error': f"Evaluation benchmark failed: {result.stderr}",
                'stdout': result.stdout
            }
        
        # Load results
        results_file = Path(output_dir) / f"benchmark_{model_type}_*.json"
        import glob
        matching_files = glob.glob(str(results_file))
        if matching_files:
            with open(matching_files[0]) as f:
                return json.load(f)
        else:
            return {
                'model_type': model_type,
                'error': f"Results file not found: {results_file}",
                'stdout': result.stdout
            }
    except subprocess.TimeoutExpired:
        return {
            'model_type': model_type,
            'error': "Evaluation benchmark timed out after 30 minutes"
        }
    except Exception as e:
        return {
            'model_type': model_type,
            'error': f"Evaluation benchmark exception: {str(e)}"
        }


def compare_results(baseline_results: Dict, extended_results: Dict) -> Dict:
    """
    Compare extended model results against baseline.
    
    Args:
        baseline_results: Results from baseline RAP model
        extended_results: Results from extended model (UAAS/Probabilistic/Semantic)
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'model_type': extended_results.get('model_type', 'unknown'),
        'comparisons': {}
    }
    
    # Compare training metrics
    if 'training' in baseline_results and 'training' in extended_results:
        base_train = baseline_results['training']
        ext_train = extended_results['training']
        
        comparison['comparisons']['training'] = {
            'batch_time_improvement_pct': (
                (base_train.get('avg_batch_time', 0) - ext_train.get('avg_batch_time', 0)) /
                base_train.get('avg_batch_time', 1e-10) * 100
            ),
            'throughput_improvement_pct': (
                (ext_train.get('batches_per_second', 0) - base_train.get('batches_per_second', 0)) /
                base_train.get('batches_per_second', 1e-10) * 100
            ),
            'memory_overhead_gb': (
                ext_train.get('memory', {}).get('avg_allocated_gb', 0) -
                base_train.get('memory', {}).get('avg_allocated_gb', 0)
            )
        }
    
    # Compare evaluation metrics
    if 'accuracy' in baseline_results and 'accuracy' in extended_results:
        base_acc = baseline_results['accuracy']
        ext_acc = extended_results['accuracy']
        
        comparison['comparisons']['evaluation'] = {
            'translation_error_reduction_pct': (
                (base_acc.get('median_translation', 0) - ext_acc.get('median_translation', 0)) /
                base_acc.get('median_translation', 1e-10) * 100
            ),
            'rotation_error_reduction_pct': (
                (base_acc.get('median_rotation', 0) - ext_acc.get('median_rotation', 0)) /
                base_acc.get('median_rotation', 1e-10) * 100
            ),
            'success_rate_5cm_5deg_improvement': (
                ext_acc.get('success_rate_5cm_5deg', 0) -
                base_acc.get('success_rate_5cm_5deg', 0)
            ),
            'success_rate_2cm_2deg_improvement': (
                ext_acc.get('success_rate_2cm_2deg', 0) -
                base_acc.get('success_rate_2cm_2deg', 0)
            )
        }
    
    # Compare inference speed
    if 'speed' in baseline_results and 'speed' in extended_results:
        base_speed = baseline_results['speed']
        ext_speed = extended_results['speed']
        
        comparison['comparisons']['inference'] = {
            'fps_change_pct': (
                (ext_speed.get('fps', 0) - base_speed.get('fps', 0)) /
                base_speed.get('fps', 1e-10) * 100
            ),
            'latency_change_pct': (
                (ext_speed.get('mean_inference_time', 0) - base_speed.get('mean_inference_time', 0)) /
                base_speed.get('mean_inference_time', 1e-10) * 100
            )
        }
    
    return comparison


def run_parallel_benchmarks(
    configs: List[BenchmarkConfig],
    base_args: argparse.Namespace,
    output_dir: Path,
    max_workers: Optional[int] = None
) -> Dict[str, BenchmarkResults]:
    """
    Run benchmarks in parallel for multiple model types.
    
    Args:
        configs: List of benchmark configurations
        base_args: Base arguments from argument parser
        output_dir: Output directory for results
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping model_type to BenchmarkResults
    """
    if max_workers is None:
        max_workers = min(4, multiprocessing.cpu_count())
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare arguments for workers
    training_tasks = []
    evaluation_tasks = []
    
    for config in configs:
        args_dict = {
            'model_type': config.model_type,
            'config': base_args.config,
            'datadir': base_args.datadir,
            'model_path': base_args.model_path,
            'output': str(output_dir),
            'benchmark_epochs': config.num_epochs,
            'benchmark_batches': config.num_batches,
        }
        
        if config.checkpoint_path:
            args_dict['checkpoint_path'] = config.checkpoint_path
        
        if config.run_training:
            training_tasks.append((config.model_type, args_dict))
        if config.run_evaluation:
            evaluation_tasks.append((config.model_type, args_dict))
    
    results = {}
    
    # Run training benchmarks in parallel
    if training_tasks:
        print(f"\n{'='*70}")
        print(f"Running {len(training_tasks)} training benchmarks in parallel...")
        print(f"{'='*70}\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(run_training_benchmark_worker, args_dict): model_type
                for model_type, args_dict in training_tasks
            }
            
            for future in tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Training benchmarks"):
                model_type = future_to_model[future]
                try:
                    training_result = future.result()
                    if model_type not in results:
                        results[model_type] = BenchmarkResults(model_type=model_type)
                    results[model_type].training_results = training_result
                except Exception as e:
                    print(f"Error in training benchmark for {model_type}: {e}")
                    if model_type not in results:
                        results[model_type] = BenchmarkResults(model_type=model_type)
                    results[model_type].error = str(e)
    
    # Run evaluation benchmarks in parallel
    if evaluation_tasks:
        print(f"\n{'='*70}")
        print(f"Running {len(evaluation_tasks)} evaluation benchmarks in parallel...")
        print(f"{'='*70}\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model = {
                executor.submit(run_evaluation_benchmark_worker, args_dict): model_type
                for model_type, args_dict in evaluation_tasks
            }
            
            for future in tqdm(as_completed(future_to_model), total=len(future_to_model), desc="Evaluation benchmarks"):
                model_type = future_to_model[future]
                try:
                    eval_result = future.result()
                    if model_type not in results:
                        results[model_type] = BenchmarkResults(model_type=model_type)
                    results[model_type].evaluation_results = eval_result.get('accuracy', {})
                    results[model_type].inference_speed = eval_result.get('speed', {})
                except Exception as e:
                    print(f"Error in evaluation benchmark for {model_type}: {e}")
                    if model_type not in results:
                        results[model_type] = BenchmarkResults(model_type=model_type)
                    results[model_type].error = str(e)
    
    # Add timestamps
    for result in results.values():
        result.timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    return results


def generate_comparison_report(results: Dict[str, BenchmarkResults], output_file: Path):
    """
    Generate a comprehensive comparison report.
    
    Args:
        results: Dictionary of benchmark results
        output_file: Path to save the report
    """
    baseline = results.get('baseline')
    if not baseline:
        print("Warning: No baseline results found. Skipping comparison report.")
        return
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAP-ID BENCHMARK COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Baseline results summary
    report_lines.append("BASELINE RAP RESULTS")
    report_lines.append("-" * 80)
    if baseline.training_results:
        train = baseline.training_results.get('training', {})
        report_lines.append(f"Training:")
        report_lines.append(f"  Avg Batch Time:     {train.get('avg_batch_time', 0)*1000:.2f} ms")
        report_lines.append(f"  Batches/Second:     {train.get('batches_per_second', 0):.2f}")
        report_lines.append(f"  Estimated 30k iter: {train.get('estimated_30k_iter_time_hours', 0):.2f} hours")
    
    if baseline.evaluation_results:
        eval_acc = baseline.evaluation_results
        report_lines.append(f"\nEvaluation:")
        report_lines.append(f"  Median Translation Error: {eval_acc.get('median_translation', 0):.4f} m")
        report_lines.append(f"  Median Rotation Error:    {eval_acc.get('median_rotation', 0):.4f} deg")
        report_lines.append(f"  Success Rate (5cm, 5deg): {eval_acc.get('success_rate_5cm_5deg', 0):.4f}")
        report_lines.append(f"  Success Rate (2cm, 2deg): {eval_acc.get('success_rate_2cm_2deg', 0):.4f}")
    
    if baseline.inference_speed:
        speed = baseline.inference_speed
        report_lines.append(f"\nInference Speed:")
        report_lines.append(f"  FPS:            {speed.get('fps', 0):.2f}")
        report_lines.append(f"  Mean Latency:   {speed.get('mean_inference_time', 0)*1000:.2f} ms")
    
    report_lines.append("")
    
    # Extended models comparison
    extended_models = ['uaas', 'probabilistic', 'semantic']
    for model_type in extended_models:
        if model_type not in results:
            continue
        
        ext_result = results[model_type]
        report_lines.append("=" * 80)
        report_lines.append(f"{model_type.upper()} MODEL RESULTS")
        report_lines.append("-" * 80)
        
        if ext_result.error:
            report_lines.append(f"ERROR: {ext_result.error}")
            report_lines.append("")
            continue
        
        # Training comparison
        if ext_result.training_results and baseline.training_results:
            base_train = baseline.training_results.get('training', {})
            ext_train = ext_result.training_results.get('training', {})
            
            batch_time_improvement = (
                (base_train.get('avg_batch_time', 0) - ext_train.get('avg_batch_time', 0)) /
                base_train.get('avg_batch_time', 1e-10) * 100
            )
            
            report_lines.append("Training Performance:")
            report_lines.append(f"  Avg Batch Time:     {ext_train.get('avg_batch_time', 0)*1000:.2f} ms")
            report_lines.append(f"  Change vs Baseline: {batch_time_improvement:+.2f}%")
            report_lines.append(f"  Batches/Second:     {ext_train.get('batches_per_second', 0):.2f}")
            report_lines.append(f"  Estimated 30k iter: {ext_train.get('estimated_30k_iter_time_hours', 0):.2f} hours")
        
        # Evaluation comparison
        if ext_result.evaluation_results and baseline.evaluation_results:
            base_eval = baseline.evaluation_results
            ext_eval = ext_result.evaluation_results
            
            trans_reduction = (
                (base_eval.get('median_translation', 0) - ext_eval.get('median_translation', 0)) /
                base_eval.get('median_translation', 1e-10) * 100
            )
            rot_reduction = (
                (base_eval.get('median_rotation', 0) - ext_eval.get('median_rotation', 0)) /
                base_eval.get('median_rotation', 1e-10) * 100
            )
            
            report_lines.append("\nEvaluation Accuracy:")
            report_lines.append(f"  Median Translation Error: {ext_eval.get('median_translation', 0):.4f} m")
            report_lines.append(f"  Change vs Baseline:        {trans_reduction:+.2f}%")
            report_lines.append(f"  Median Rotation Error:     {ext_eval.get('median_rotation', 0):.4f} deg")
            report_lines.append(f"  Change vs Baseline:        {rot_reduction:+.2f}%")
            report_lines.append(f"  Success Rate (5cm, 5deg):  {ext_eval.get('success_rate_5cm_5deg', 0):.4f}")
            report_lines.append(f"  Success Rate (2cm, 2deg):  {ext_eval.get('success_rate_2cm_2deg', 0):.4f}")
            
            if 'mean_uncertainty' in ext_eval:
                report_lines.append(f"\n  Mean Uncertainty:          {ext_eval.get('mean_uncertainty', 0):.4f}")
        
        # Inference speed comparison
        if ext_result.inference_speed and baseline.inference_speed:
            base_speed = baseline.inference_speed
            ext_speed = ext_result.inference_speed
            
            fps_change = (
                (ext_speed.get('fps', 0) - base_speed.get('fps', 0)) /
                base_speed.get('fps', 1e-10) * 100
            )
            
            report_lines.append("\nInference Speed:")
            report_lines.append(f"  FPS:            {ext_speed.get('fps', 0):.2f}")
            report_lines.append(f"  Change vs Baseline: {fps_change:+.2f}%")
            report_lines.append(f"  Mean Latency:   {ext_speed.get('mean_inference_time', 0)*1000:.2f} ms")
        
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save text report
    output_file.write_text(report_text)
    print(f"\nComparison report saved to: {output_file}")
    
    # Print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive parallel benchmarking suite for RAP-ID")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["baseline", "uaas", "probabilistic", "semantic"],
                       choices=["baseline", "uaas", "probabilistic", "semantic"],
                       help="Model types to benchmark")
    parser.add_argument("--parallel", action="store_true", default=True,
                       help="Run benchmarks in parallel")
    parser.add_argument("--max_workers", type=int, default=None,
                       help="Maximum number of parallel workers")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for benchmark results")
    parser.add_argument("--benchmark_epochs", type=int, default=1,
                       help="Number of epochs for training benchmark")
    parser.add_argument("--benchmark_batches", type=int, default=None,
                       help="Number of batches per epoch (None = all)")
    parser.add_argument("--training_only", action="store_true",
                       help="Run only training benchmarks")
    parser.add_argument("--evaluation_only", action="store_true",
                       help="Run only evaluation benchmarks")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.model_path).parent / "benchmark_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create benchmark configurations
    configs = []
    for model_type in args.models:
        config = BenchmarkConfig(
            model_type=model_type,
            run_training=not args.evaluation_only,
            run_evaluation=not args.training_only,
            num_epochs=args.benchmark_epochs,
            num_batches=args.benchmark_batches
        )
        configs.append(config)
    
    print("=" * 70)
    print("RAP-ID Comprehensive Parallel Benchmarking Suite")
    print("=" * 70)
    print(f"\nModels to benchmark: {', '.join(args.models)}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel execution: {args.parallel}")
    print(f"Max workers: {args.max_workers or 'auto'}")
    print(f"Benchmark epochs: {args.benchmark_epochs}")
    print()
    
    # Run benchmarks
    results = run_parallel_benchmarks(
        configs,
        args,
        output_dir,
        max_workers=args.max_workers if args.parallel else 1
    )
    
    # Save individual results
    results_dict = {}
    for model_type, result in results.items():
        results_file = output_dir / f"results_{model_type}.json"
        results_dict[model_type] = asdict(result)
        with open(results_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
    
    # Generate comparison report
    report_file = output_dir / "comparison_report.txt"
    generate_comparison_report(results, report_file)
    
    # Save JSON summary
    summary_file = output_dir / "benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"\nBenchmark summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

