#!/usr/bin/env python3
"""
Comprehensive analysis script for comparing all models.
"""
import json
import numpy as np
from pathlib import Path
import sys

def analyze_results(results_file="benchmark_full_pipeline_results.json"):
    """Analyze benchmark results comprehensively."""
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("   Run benchmark first: python benchmark_full_pipeline.py ...")
        return
    
    baseline = data['results']['baseline']
    
    print("="*80)
    print("COMPREHENSIVE MODEL ANALYSIS")
    print("="*80)
    
    print(f"\nDataset: {data['dataset']}")
    print(f"Test Samples: {data['num_test_samples']}")
    print(f"Device: {data['device']}")
    print(f"Timestamp: {data['timestamp']}")
    
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE")
    print("="*80)
    print(f"Translation Error (median): {baseline['accuracy']['translation_errors']['median']:.4f}m")
    print(f"Translation Error (mean): {baseline['accuracy']['translation_errors']['mean']:.4f}m")
    print(f"Rotation Error (median): {baseline['accuracy']['rotation_errors']['median']:.4f}°")
    print(f"Rotation Error (mean): {baseline['accuracy']['rotation_errors']['mean']:.4f}°")
    print(f"Inference Speed: {baseline['inference']['fps']:.2f} FPS")
    print(f"Model Size: {baseline['initialization']['model_size_mb']:.2f} MB")
    
    print("\n" + "="*80)
    print("MODEL COMPARISON & IMPROVEMENTS")
    print("="*80)
    
    improvements_summary = []
    
    for model_name in ['uaas', 'probabilistic', 'semantic']:
        if model_name not in data['results']:
            continue
        
        model = data['results'][model_name]
        imp = data['improvements'][model_name]
        
        print(f"\n{model_name.upper()}:")
        print("-" * 80)
        
        # Translation
        trans_imp = imp['accuracy']['translation']['improvement_pct']
        baseline_trans = imp['accuracy']['translation']['baseline']
        model_trans = imp['accuracy']['translation']['model']
        if trans_imp > 0:
            print(f"  ✅ Translation: {baseline_trans:.4f}m → {model_trans:.4f}m (+{trans_imp:.1f}% improvement)")
            improvements_summary.append(f"{model_name}: +{trans_imp:.1f}% translation")
        else:
            print(f"  ❌ Translation: {baseline_trans:.4f}m → {model_trans:.4f}m ({trans_imp:.1f}% worse)")
        
        # Rotation
        baseline_rot = imp['accuracy']['rotation']['baseline']
        model_rot = imp['accuracy']['rotation']['model']
        baseline_rot_mean = baseline['accuracy']['rotation_errors']['mean']
        model_rot_mean = model['accuracy']['rotation_errors']['mean']
        
        print(f"  Rotation (median): {baseline_rot:.4f}° → {model_rot:.4f}°")
        print(f"  Rotation (mean): {baseline_rot_mean:.4f}° → {model_rot_mean:.4f}°")
        
        if baseline_rot_mean > 0 and model_rot_mean < baseline_rot_mean:
            improvement = ((baseline_rot_mean - model_rot_mean) / baseline_rot_mean) * 100
            print(f"    ✅ Rotation mean improved by {improvement:.1f}%")
            improvements_summary.append(f"{model_name}: +{improvement:.1f}% rotation (mean)")
        
        # Speed
        speed_imp = imp.get('inference_speed', {}).get('improvement_pct', 0)
        if speed_imp > 0:
            print(f"  ✅ Speed: {imp['inference_speed']['baseline_fps']:.2f} → {imp['inference_speed']['model_fps']:.2f} FPS (+{speed_imp:.1f}%)")
            improvements_summary.append(f"{model_name}: +{speed_imp:.1f}% speed")
        else:
            print(f"  ❌ Speed: {imp['inference_speed']['baseline_fps']:.2f} → {imp['inference_speed']['model_fps']:.2f} FPS ({speed_imp:.1f}%)")
        
        # Model Size
        size_change = imp['model_size']['size_change_pct']
        if abs(size_change) < 1.0:
            print(f"  ➖ Model Size: {imp['model_size']['baseline_mb']:.2f}MB → {imp['model_size']['model_mb']:.2f}MB ({size_change:+.1f}%, minimal change)")
        else:
            print(f"  ⚠️  Model Size: {imp['model_size']['baseline_mb']:.2f}MB → {imp['model_size']['model_mb']:.2f}MB ({size_change:+.1f}%)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if improvements_summary:
        print("\n✅ IMPROVEMENTS FOUND:")
        for imp in improvements_summary:
            print(f"  - {imp}")
    else:
        print("\n⚠️  No clear improvements found in current results")
        print("   This may be due to:")
        print("   - Untrained or poorly trained model checkpoints")
        print("   - Small test dataset")
        print("   - Model architecture issues")
        print("   - Need for hyperparameter tuning")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("1. Ensure all models are trained with sufficient epochs")
    print("2. Train with proper hyperparameters (learning rate, batch size)")
    print("3. Use validation set for early stopping")
    print("4. Compare best checkpoints from each training run")
    print("5. Run on multiple datasets for robustness")

if __name__ == "__main__":
    results_file = sys.argv[1] if len(sys.argv) > 1 else "benchmark_full_pipeline_results.json"
    analyze_results(results_file)

