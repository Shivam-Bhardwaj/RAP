#!/usr/bin/env python3
"""
Benchmark script to compare RAP-ID against original RAP implementation.

This script can:
1. Clone and setup the original RAP repository for comparison
2. Run benchmarks on both implementations
3. Generate comparison reports

Usage:
    python benchmark_vs_original.py --original_repo_path /path/to/original/RAP --compare
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, Optional
import shutil

import numpy as np


def setup_original_repo(original_repo_path: Optional[str] = None, clone_path: Optional[str] = None) -> Path:
    """
    Setup original RAP repository for benchmarking.
    
    Args:
        original_repo_path: Path to existing original RAP repo
        clone_path: Path where to clone if repo doesn't exist
        
    Returns:
        Path to original RAP repository
    """
    if original_repo_path and Path(original_repo_path).exists():
        print(f"Using existing original RAP repo at: {original_repo_path}")
        return Path(original_repo_path)
    
    if clone_path is None:
        clone_path = Path.home() / "RAP_original"
    
    clone_path = Path(clone_path)
    
    if clone_path.exists():
        print(f"Original RAP repo already exists at: {clone_path}")
        return clone_path
    
    print("Cloning original RAP repository...")
    print("URL: https://github.com/ai4ce/RAP")
    
    try:
        subprocess.run(
            ["git", "clone", "--recursive", "https://github.com/ai4ce/RAP", str(clone_path)],
            check=True
        )
        print(f"Successfully cloned original RAP repo to: {clone_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        print("Please clone manually: git clone --recursive https://github.com/ai4ce/RAP")
        sys.exit(1)
    
    return clone_path


def run_original_rap_benchmark(original_repo_path: Path, config_path: str, model_path: str, output_dir: Path) -> Dict:
    """
    Run benchmark on original RAP implementation.
    
    Args:
        original_repo_path: Path to original RAP repository
        config_path: Path to config file
        model_path: Path to 3DGS model
        output_dir: Output directory for results
        
    Returns:
        Dictionary with benchmark results
    """
    original_repo_path = Path(original_repo_path)
    
    print(f"\n{'='*70}")
    print("Running benchmark on ORIGINAL RAP implementation")
    print(f"{'='*70}")
    
    # Check if original repo has benchmarking scripts
    # The original repo uses rap.py directly, so we'll need to adapt
    rap_script = original_repo_path / "rap.py"
    
    if not rap_script.exists():
        print(f"Warning: rap.py not found in {original_repo_path}")
        return {"error": "Original RAP script not found"}
    
    # For now, we'll note that manual benchmarking is needed
    # In a full implementation, we would:
    # 1. Setup Python environment in original repo
    # 2. Run training/evaluation
    # 3. Parse results
    
    results = {
        "note": "Manual benchmarking required",
        "original_repo_path": str(original_repo_path),
        "config_path": config_path,
        "model_path": model_path
    }
    
    return results


def extract_original_results(results_dir: Path) -> Dict:
    """
    Extract results from original RAP implementation.
    
    The original RAP repo outputs results in specific formats.
    This function parses those results.
    
    Args:
        results_dir: Directory containing original RAP results
        
    Returns:
        Dictionary with parsed results
    """
    results = {}
    
    # Look for evaluation output files
    # Original RAP typically outputs pose errors in text/log files
    
    # Check for common output patterns
    for file in results_dir.glob("*.txt"):
        if "eval" in file.name.lower() or "result" in file.name.lower():
            try:
                with open(file) as f:
                    content = f.read()
                    # Try to parse pose errors from text
                    # This is a placeholder - actual parsing depends on output format
                    results[file.name] = content
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    return results


def compare_with_original(rap_id_results: Dict, original_results: Dict) -> Dict:
    """
    Compare RAP-ID results with original RAP results.
    
    Args:
        rap_id_results: Results from RAP-ID benchmarks
        original_results: Results from original RAP benchmarks
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "rap_id_results": rap_id_results,
        "original_results": original_results,
        "improvements": {}
    }
    
    # Compare translation errors
    if "evaluation_results" in rap_id_results and "evaluation_results" in original_results:
        rap_id_trans = rap_id_results["evaluation_results"].get("median_translation", 0)
        orig_trans = original_results["evaluation_results"].get("median_translation", 0)
        
        if orig_trans > 0:
            improvement_pct = ((orig_trans - rap_id_trans) / orig_trans) * 100
            comparison["improvements"]["translation_error_reduction"] = improvement_pct
    
    # Compare rotation errors
    if "evaluation_results" in rap_id_results and "evaluation_results" in original_results:
        rap_id_rot = rap_id_results["evaluation_results"].get("median_rotation", 0)
        orig_rot = original_results["evaluation_results"].get("median_rotation", 0)
        
        if orig_rot > 0:
            improvement_pct = ((orig_rot - rap_id_rot) / orig_rot) * 100
            comparison["improvements"]["rotation_error_reduction"] = improvement_pct
    
    # Compare success rates
    if "evaluation_results" in rap_id_results and "evaluation_results" in original_results:
        rap_id_sr = rap_id_results["evaluation_results"].get("success_rate_5cm_5deg", 0)
        orig_sr = original_results["evaluation_results"].get("success_rate_5cm_5deg", 0)
        
        improvement = rap_id_sr - orig_sr
        comparison["improvements"]["success_rate_improvement"] = improvement * 100  # Convert to percentage points
    
    return comparison


def generate_original_comparison_report(comparison: Dict, output_file: Path):
    """
    Generate comparison report against original RAP.
    
    Args:
        comparison: Comparison dictionary
        output_file: Path to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("RAP-ID vs ORIGINAL RAP COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # RAP-ID Results
    if "rap_id_results" in comparison:
        rap_id = comparison["rap_id_results"]
        report_lines.append("RAP-ID RESULTS")
        report_lines.append("-" * 80)
        
        if "evaluation_results" in rap_id:
            eval_res = rap_id["evaluation_results"]
            report_lines.append(f"Median Translation Error: {eval_res.get('median_translation', 0):.4f} m")
            report_lines.append(f"Median Rotation Error:    {eval_res.get('median_rotation', 0):.4f} deg")
            report_lines.append(f"Success Rate (5cm, 5deg): {eval_res.get('success_rate_5cm_5deg', 0):.4f}")
    
    # Original RAP Results
    if "original_results" in comparison:
        orig = comparison["original_results"]
        report_lines.append("\nORIGINAL RAP RESULTS")
        report_lines.append("-" * 80)
        
        if "evaluation_results" in orig:
            eval_res = orig["evaluation_results"]
            report_lines.append(f"Median Translation Error: {eval_res.get('median_translation', 0):.4f} m")
            report_lines.append(f"Median Rotation Error:    {eval_res.get('median_rotation', 0):.4f} deg")
            report_lines.append(f"Success Rate (5cm, 5deg): {eval_res.get('success_rate_5cm_5deg', 0):.4f}")
    
    # Improvements
    if "improvements" in comparison:
        improvements = comparison["improvements"]
        report_lines.append("\nIMPROVEMENTS OVER ORIGINAL")
        report_lines.append("-" * 80)
        
        if "translation_error_reduction" in improvements:
            report_lines.append(f"Translation Error Reduction: {improvements['translation_error_reduction']:+.2f}%")
        
        if "rotation_error_reduction" in improvements:
            report_lines.append(f"Rotation Error Reduction:     {improvements['rotation_error_reduction']:+.2f}%")
        
        if "success_rate_improvement" in improvements:
            report_lines.append(f"Success Rate Improvement:      {improvements['success_rate_improvement']:+.2f} percentage points")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    output_file.write_text(report_text)
    
    print("\n" + report_text)
    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RAP-ID against original RAP implementation"
    )
    
    parser.add_argument("--original_repo_path", type=str, default=None,
                       help="Path to original RAP repository (if exists)")
    parser.add_argument("--clone_original", action="store_true",
                       help="Clone original RAP repo if not exists")
    parser.add_argument("--rap_id_results", type=str, default=None,
                       help="Path to RAP-ID benchmark results JSON")
    parser.add_argument("--original_results", type=str, default=None,
                       help="Path to original RAP benchmark results JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for comparison report")
    parser.add_argument("--compare", action="store_true",
                       help="Run comparison analysis")
    
    args = parser.parse_args()
    
    if args.clone_original or args.original_repo_path:
        original_repo = setup_original_repo(args.original_repo_path)
        print(f"\nOriginal RAP repository available at: {original_repo}")
        print("\nNote: To benchmark original RAP, run:")
        print(f"  cd {original_repo}")
        print(f"  python rap.py -c <config> -m <model_path>")
        print("\nThen use --original_results to load those results for comparison.")
    
    if args.compare:
        if not args.rap_id_results or not args.original_results:
            print("\nError: Both --rap_id_results and --original_results required for comparison")
            print("\nTo generate RAP-ID results, run:")
            print("  python benchmark_comparison.py -c <config> -m <model_path>")
            sys.exit(1)
        
        # Load results
        with open(args.rap_id_results) as f:
            rap_id_results = json.load(f)
        
        with open(args.original_results) as f:
            original_results = json.load(f)
        
        # Compare
        comparison = compare_with_original(rap_id_results, original_results)
        
        # Generate report
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = Path(".")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / "original_comparison_report.txt"
        
        generate_original_comparison_report(comparison, report_file)
        
        # Save JSON
        json_file = output_dir / "original_comparison.json"
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"\nComparison JSON saved to: {json_file}")


if __name__ == "__main__":
    main()

