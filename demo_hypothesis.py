#!/usr/bin/env python3
"""
Practical Demo: RAP-ID Multi-Hypothesis Pose Estimation

This demo shows how the probabilistic model handles ambiguous scenes by
predicting multiple pose hypotheses and ranking them using rendering.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, OptimizationParams, get_combined_args
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.colmap_dataset import ColmapDataset
from utils.cameras import CamParams
from utils.general_utils import fix_seed
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from probabilistic.hypothesis_validator import HypothesisValidator
from utils.eval_utils import get_pose_error


def demo_multi_hypothesis(args):
    """Demonstrate multi-hypothesis pose estimation."""
    print("=" * 70)
    print("RAP-ID Demo: Multi-Hypothesis Pose Estimation")
    print("=" * 70)
    
    device = torch.device(args.device)
    fix_seed(args.seed)
    
    # Load model
    print("\nLoading Probabilistic model...")
    model = ProbabilisticRAPNet(args, num_gaussians=5).to(device)
    
    if args.pretrained_model_path and os.path.exists(args.pretrained_model_path):
        print(f"Loading checkpoint from {args.pretrained_model_path}")
        checkpoint = torch.load(args.pretrained_model_path, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    else:
        print("Warning: No checkpoint provided. Using untrained model.")
    
    model.eval()
    
    # Load dataset
    print("\nLoading dataset...")
    with open(f"{args.model_path}/cameras.json") as f:
        import json
        camera = json.load(f)[0]
    
    rap_cam_params = CamParams(camera, args.rap_resolution, device)
    rap_hw = (rap_cam_params.h, rap_cam_params.w)
    
    val_set = ColmapDataset(train=False, data_path=args.datadir, hw=rap_hw)
    
    # Initialize validator (requires renderer)
    print("\nInitializing hypothesis validator...")
    validator = HypothesisValidator(renderer=None)  # Will need renderer in practice
    
    # Process a few images
    print("\nProcessing images and generating hypotheses...")
    
    results = []
    
    with torch.no_grad():
        for i in range(min(3, len(val_set))):
            img, pose_gt, _, _ = val_set[i]
            img = img.unsqueeze(0).to(device)
            pose_gt = pose_gt.reshape(3, 4).to(device)
            
            # Get mixture distribution
            mixture_dist = model(img, return_feature=False)
            
            # Sample multiple hypotheses
            n_hypotheses = 10
            hypotheses_samples = mixture_dist.sample((n_hypotheses,))  # (n_hyp, batch, 6)
            hypotheses = hypotheses_samples[:, 0, :]  # (n_hyp, 6)
            
            # Get mixture weights
            mixture_weights = torch.softmax(mixture_dist.component_distribution.mixture_distribution.logits, dim=-1)[0]
            
            print(f"\n  Image {i+1}:")
            print(f"    Mixture weights: {mixture_weights.cpu().numpy()}")
            print(f"    Generated {n_hypotheses} hypotheses")
            
            # Compute errors for each hypothesis
            errors = []
            for hyp in hypotheses:
                # Convert to pose matrix (simplified - actual conversion depends on representation)
                hyp_pose = hyp.reshape(3, 2)  # Placeholder
                # For demo, we'll use simplified error computation
                error = torch.norm(hyp - pose_gt.reshape(12)[:6]).item()
                errors.append(error)
            
            best_idx = np.argmin(errors)
            best_error = errors[best_idx]
            
            print(f"    Best hypothesis error: {best_error:.4f}")
            print(f"    Mean hypothesis error: {np.mean(errors):.4f}")
            print(f"    Std hypothesis error: {np.std(errors):.4f}")
            
            results.append({
                'errors': errors,
                'best_error': best_error,
                'mixture_weights': mixture_weights.cpu().numpy()
            })
    
    # Visualize
    print("\nGenerating visualizations...")
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.hist(result['errors'], bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(result['best_error'], color='r', linestyle='--', label=f'Best: {result["best_error"]:.4f}')
        ax.set_xlabel('Pose Error (L2 norm)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Image {i+1}: Hypothesis Errors')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "hypothesis_errors.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to {save_path}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nHypothesis Statistics:")
    print(f"  Mean best error: {np.mean([r['best_error'] for r in results]):.4f}")
    print(f"  Mean hypothesis diversity: {np.mean([np.std(r['errors']) for r in results]):.4f}")


def main():
    parser = argparse.ArgumentParser(description="RAP-ID Multi-Hypothesis Demo")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                       help="Path to pretrained Probabilistic model")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                       help="Output directory for visualizations")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    demo_multi_hypothesis(args)


if __name__ == "__main__":
    main()

