#!/usr/bin/env python3
"""
Practical Demo: RAP-ID Comparison Tool

This demo compares baseline RAP vs RAP-ID extensions on the same images,
showing improvements in accuracy, uncertainty calibration, and robustness.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.colmap_dataset import ColmapDataset
from utils.cameras import CamParams
from utils.general_utils import fix_seed
from models.apr.rapnet import RAPNet
from uaas.uaas_rap_net import UAASRAPNet
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from semantic.semantic_rap_net import SemanticRAPNet
from utils.eval_utils import get_pose_error


def demo_comparison(args):
    """Compare baseline vs RAP-ID extensions."""
    print("=" * 70)
    print("RAP-ID Demo: Baseline vs Extensions Comparison")
    print("=" * 70)
    
    device = torch.device(args.device)
    fix_seed(args.seed)
    
    # Load dataset
    print("\nLoading dataset...")
    with open(f"{args.model_path}/cameras.json") as f:
        camera = json.load(f)[0]
    
    rap_cam_params = CamParams(camera, args.rap_resolution, device)
    rap_hw = (rap_cam_params.h, rap_cam_params.w)
    
    val_set = ColmapDataset(train=False, data_path=args.datadir, hw=rap_hw)
    
    # Initialize models
    models = {}
    
    print("\nLoading models...")
    if args.baseline_checkpoint:
        print("  Loading baseline RAP...")
        models['baseline'] = RAPNet(args).to(device)
        checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            models['baseline'].load_state_dict(checkpoint['model_state_dict'])
        else:
            models['baseline'].load_state_dict(checkpoint)
        models['baseline'].eval()
    
    if args.uaas_checkpoint:
        print("  Loading UAAS...")
        models['uaas'] = UAASRAPNet(args).to(device)
        checkpoint = torch.load(args.uaas_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            models['uaas'].load_state_dict(checkpoint['model_state_dict'])
        else:
            models['uaas'].load_state_dict(checkpoint)
        models['uaas'].eval()
    
    if args.probabilistic_checkpoint:
        print("  Loading Probabilistic...")
        models['probabilistic'] = ProbabilisticRAPNet(args, num_gaussians=5).to(device)
        checkpoint = torch.load(args.probabilistic_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            models['probabilistic'].load_state_dict(checkpoint['model_state_dict'])
        else:
            models['probabilistic'].load_state_dict(checkpoint)
        models['probabilistic'].eval()
    
    # Evaluate on test set
    print("\nEvaluating models...")
    results = {name: {'trans_errors': [], 'rot_errors': [], 'uncertainties': []} 
               for name in models.keys()}
    
    with torch.no_grad():
        for i in range(min(args.num_samples, len(val_set))):
            img, pose_gt, _, _ = val_set[i]
            img = img.unsqueeze(0).to(device)
            pose_gt = pose_gt.reshape(3, 4).to(device)
            
            for model_name, model in models.items():
                if model_name == 'baseline':
                    _, pose_pred = model(img, return_feature=False)
                elif model_name == 'uaas':
                    pose_pred, log_var = model(img, return_feature=False)
                    uncertainty = torch.exp(log_var).mean().item()
                    results[model_name]['uncertainties'].append(uncertainty)
                elif model_name == 'probabilistic':
                    mixture_dist = model(img, return_feature=False)
                    pose_pred = mixture_dist.sample((1,))[0, 0, :].reshape(12)
                else:
                    continue
                
                # Compute errors
                pose_pred_mat = pose_pred.reshape(3, 4)
                trans_error, rot_error = get_pose_error(
                    pose_gt.unsqueeze(0), pose_pred_mat.unsqueeze(0)
                )
                
                results[model_name]['trans_errors'].append(trans_error.item())
                results[model_name]['rot_errors'].append(rot_error.item())
    
    # Print results
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Translation Error:")
        print(f"    Median: {np.median(result['trans_errors']):.4f} m")
        print(f"    Mean:   {np.mean(result['trans_errors']):.4f} m")
        print(f"  Rotation Error:")
        print(f"    Median: {np.median(result['rot_errors']):.4f} deg")
        print(f"    Mean:   {np.mean(result['rot_errors']):.4f} deg")
        
        if 'uncertainties' in result and len(result['uncertainties']) > 0:
            print(f"  Uncertainty:")
            print(f"    Mean: {np.mean(result['uncertainties']):.4f}")
            print(f"    Std:  {np.std(result['uncertainties']):.4f}")
    
    # Visualize
    print("\nGenerating comparison visualizations...")
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Translation errors
    ax = axes[0, 0]
    for model_name, result in results.items():
        ax.hist(result['trans_errors'], bins=20, alpha=0.6, label=model_name)
    ax.set_xlabel('Translation Error (m)')
    ax.set_ylabel('Frequency')
    ax.set_title('Translation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotation errors
    ax = axes[0, 1]
    for model_name, result in results.items():
        ax.hist(result['rot_errors'], bins=20, alpha=0.6, label=model_name)
    ax.set_xlabel('Rotation Error (deg)')
    ax.set_ylabel('Frequency')
    ax.set_title('Rotation Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax = axes[1, 0]
    data = [result['trans_errors'] for result in results.values()]
    ax.boxplot(data, labels=list(results.keys()))
    ax.set_ylabel('Translation Error (m)')
    ax.set_title('Translation Error Comparison')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    data = [result['rot_errors'] for result in results.values()]
    ax.boxplot(data, labels=list(results.keys()))
    ax.set_ylabel('Rotation Error (deg)')
    ax.set_title('Rotation Error Comparison')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "comparison_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    
    # Save JSON results
    json_results = {}
    for model_name, result in results.items():
        json_results[model_name] = {
            'trans_error_median': float(np.median(result['trans_errors'])),
            'trans_error_mean': float(np.mean(result['trans_errors'])),
            'rot_error_median': float(np.median(result['rot_errors'])),
            'rot_error_mean': float(np.mean(result['rot_errors']))
        }
        if 'uncertainties' in result and len(result['uncertainties']) > 0:
            json_results[model_name]['uncertainty_mean'] = float(np.mean(result['uncertainties']))
    
    json_path = output_dir / "comparison_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved JSON results to {json_path}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RAP-ID Comparison Demo")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                       help="Path to baseline RAP checkpoint")
    parser.add_argument("--uaas_checkpoint", type=str, default=None,
                       help="Path to UAAS checkpoint")
    parser.add_argument("--probabilistic_checkpoint", type=str, default=None,
                       help="Path to Probabilistic checkpoint")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                       help="Output directory for results")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    demo_comparison(args)


if __name__ == "__main__":
    main()

