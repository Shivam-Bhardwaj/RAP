#!/usr/bin/env python3
"""
Practical Demo: RAP-ID Uncertainty Visualization

This demo shows how UAAS model provides uncertainty estimates for pose predictions,
helping identify unreliable predictions and improving robustness.
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
from uaas.uaas_rap_net import UAASRAPNet
from common.uncertainty import UncertaintyVisualizer, epistemic_uncertainty, aleatoric_uncertainty_regression


def demo_uncertainty_visualization(args):
    """Demonstrate uncertainty visualization for pose predictions."""
    print("=" * 70)
    print("RAP-ID Demo: Uncertainty-Aware Pose Estimation")
    print("=" * 70)
    
    device = torch.device(args.device)
    fix_seed(args.seed)
    
    # Load model
    print("\nLoading UAAS model...")
    model = UAASRAPNet(args).to(device)
    
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
    
    # Process a few images
    print("\nProcessing images and computing uncertainty...")
    visualizer = UncertaintyVisualizer()
    
    uncertainties = []
    images = []
    
    with torch.no_grad():
        for i in range(min(5, len(val_set))):
            img, pose_gt, _, _ = val_set[i]
            img = img.unsqueeze(0).to(device)
            
            # Get prediction with uncertainty
            pose_pred, log_var = model(img, return_feature=False)
            
            # Compute uncertainties
            aleatoric = aleatoric_uncertainty_regression(log_var)
            epistemic = epistemic_uncertainty(torch.stack([pose_pred, pose_pred]))  # Simplified
            
            total_uncertainty = (aleatoric + epistemic).mean().item()
            
            uncertainties.append(total_uncertainty)
            images.append(img.cpu().squeeze(0))
            
            print(f"  Image {i+1}: Total Uncertainty = {total_uncertainty:.4f}")
    
    # Visualize
    print("\nGenerating uncertainty visualizations...")
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') else Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    for i, (img, unc) in enumerate(zip(images, uncertainties)):
        # Create uncertainty map (spatial uncertainty)
        unc_map = torch.ones(rap_hw[0], rap_hw[1]) * unc
        
        save_path = output_dir / f"uncertainty_vis_{i}.png"
        visualizer.plot_uncertainty_map(img, unc_map, str(save_path))
        print(f"  Saved visualization to {save_path}")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\nUncertainty Statistics:")
    print(f"  Mean: {np.mean(uncertainties):.4f}")
    print(f"  Std:  {np.std(uncertainties):.4f}")
    print(f"  Min:  {np.min(uncertainties):.4f}")
    print(f"  Max:  {np.max(uncertainties):.4f}")


def main():
    parser = argparse.ArgumentParser(description="RAP-ID Uncertainty Visualization Demo")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                       help="Path to pretrained UAAS model")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                       help="Output directory for visualizations")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    demo_uncertainty_visualization(args)


if __name__ == "__main__":
    main()

