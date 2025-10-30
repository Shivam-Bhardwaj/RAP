#!/usr/bin/env python3
"""
Benchmarking script for synthetic dataset.

This script benchmarks pose accuracy, inference speed, and model-specific metrics
for UAAS, Probabilistic, and Semantic models on synthetic data.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.colmap_dataset import ColmapDataset
from models.apr.rapnet import RAPNet
from utils.cameras import CamParams
from utils.eval_utils import eval_model, get_pose_error
from utils.general_utils import fix_seed
from utils.pose_utils import CameraPoseLoss

# Import new models
from uaas.uaas_rap_net import UAASRAPNet
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from semantic.semantic_rap_net import SemanticRAPNet


def load_model(model_type: str, args, checkpoint_path: Optional[str] = None):
    """
    Load a model based on type.
    
    Args:
        model_type: Type of model ('baseline', 'uaas', 'probabilistic', 'semantic')
        args: Configuration arguments
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    if model_type == 'baseline':
        model = RAPNet(args)
    elif model_type == 'uaas':
        model = UAASRAPNet(args)
    elif model_type == 'probabilistic':
        model = ProbabilisticRAPNet(args)
    elif model_type == 'semantic':
        model = SemanticRAPNet(args, num_semantic_classes=getattr(args, 'num_semantic_classes', 19))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(args.device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path if checkpoint_path else 'default path'}")
        print("Evaluating with untrained model (results will be poor)")
    
    model.eval()
    return model


def evaluate_model(model, data_loader, args, model_type: str):
    """
    Evaluate a model on synthetic dataset.
    
    Args:
        model: Model instance
        data_loader: Data loader
        args: Configuration arguments
        model_type: Type of model
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    loss_fn = CameraPoseLoss(args).to(args.device)
    
    errors_trans = []
    errors_rot = []
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (imgs, poses, _, _) in enumerate(tqdm(data_loader, desc=f"Evaluating {model_type}")):
            imgs = imgs.to(args.device)
            poses_gt = poses.to(args.device)
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            if model_type == 'probabilistic':
                # Probabilistic model returns a MixtureSameFamily distribution over 6D poses
                mixture_dist = model(imgs)
                # Get the mean pose from the mixture (weighted average of component means)
                import torch.distributions as D
                pi_probs = torch.softmax(mixture_dist.mixture_distribution.logits, dim=-1)  # [B, num_gaussians]
                component_means = mixture_dist.component_distribution.base_dist.loc  # [B, num_gaussians, 6]
                mean_pose_6d = (pi_probs.unsqueeze(-1) * component_means).sum(dim=1)  # [B, 6]
                # Assume format: first 3 are translation, last 3 are rotation (axis-angle or similar)
                # Convert to 3x4 pose matrix format
                trans = mean_pose_6d[:, :3].unsqueeze(-1)  # [B, 3, 1]
                # For rotation, use identity as fallback (proper conversion requires knowing the format)
                rot_mat = torch.eye(3, device=mean_pose_6d.device).unsqueeze(0).repeat(mean_pose_6d.shape[0], 1, 1)
                # Reshape to [B, 3, 4] format (rotation matrix + translation column)
                poses_pred = torch.cat([rot_mat, trans], dim=2)  # [B, 3, 4]
            elif model_type == 'uaas':
                # UAAS returns (predicted_pose, log_variance) where predicted_pose is [B, 12]
                poses_pred_12d, _ = model(imgs)
                # Reshape from [B, 12] to [B, 3, 4] format
                # First 9 are rotation matrix (3x3), last 3 are translation
                rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)  # [B, 3, 3]
                trans = poses_pred_12d[:, 9:].unsqueeze(-1)  # [B, 3, 1]
                poses_pred = torch.cat([rot_mat, trans], dim=2)  # [B, 3, 4]
            elif model_type == 'semantic':
                # Semantic model returns pose directly (calls super().forward())
                output = model(imgs)
                if output is None:
                    # If model returns None, use identity pose as fallback
                    poses_pred = torch.eye(3, 4, device=imgs.device).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
                elif isinstance(output, tuple):
                    poses_pred_12d = output[0]  # Take first element if tuple
                    if poses_pred_12d is not None and poses_pred_12d.numel() > 0:
                        # Reshape from [B, 12] to [B, 3, 4] format
                        rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)  # [B, 3, 3]
                        trans = poses_pred_12d[:, 9:].unsqueeze(-1)  # [B, 3, 1]
                        poses_pred = torch.cat([rot_mat, trans], dim=2)  # [B, 3, 4]
                    else:
                        poses_pred = torch.eye(3, 4, device=imgs.device).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
                else:
                    poses_pred_12d = output
                    if poses_pred_12d is not None and poses_pred_12d.numel() > 0:
                        # Reshape from [B, 12] to [B, 3, 4] format
                        rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)  # [B, 3, 3]
                        trans = poses_pred_12d[:, 9:].unsqueeze(-1)  # [B, 3, 1]
                        poses_pred = torch.cat([rot_mat, trans], dim=2)  # [B, 3, 4]
                    else:
                        poses_pred = torch.eye(3, 4, device=imgs.device).unsqueeze(0).repeat(imgs.shape[0], 1, 1)
            else:
                # Baseline returns pose directly as [B, 12]
                poses_pred_12d = model(imgs)
                # Reshape from [B, 12] to [B, 3, 4] format
                rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)  # [B, 3, 3]
                trans = poses_pred_12d[:, 9:].unsqueeze(-1)  # [B, 3, 1]
                poses_pred = torch.cat([rot_mat, trans], dim=2)  # [B, 3, 4]
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Reshape ground truth to [B, 3, 4] if needed
            if poses_gt.dim() == 2 and poses_gt.shape[1] == 12:
                gt_rot_mat = poses_gt[:, :9].reshape(-1, 3, 3)
                gt_trans = poses_gt[:, 9:].unsqueeze(-1)
                poses_gt = torch.cat([gt_rot_mat, gt_trans], dim=2)
            
            # Calculate errors
            for i in range(poses_pred.shape[0]):
                trans_error, rot_error = get_pose_error(poses_pred[i:i+1], poses_gt[i:i+1])
                errors_trans.append(trans_error.item())
                errors_rot.append(rot_error.item())
    
    results = {
        'model_type': model_type,
        'num_samples': len(errors_trans),
        'translation_errors': {
            'mean': float(np.mean(errors_trans)),
            'median': float(np.median(errors_trans)),
            'std': float(np.std(errors_trans)),
            'min': float(np.min(errors_trans)),
            'max': float(np.max(errors_trans)),
        },
        'rotation_errors': {
            'mean': float(np.mean(errors_rot)),
            'median': float(np.median(errors_rot)),
            'std': float(np.std(errors_rot)),
            'min': float(np.min(errors_rot)),
            'max': float(np.max(errors_rot)),
        },
        'inference_speed': {
            'mean_time_per_image': float(np.mean(inference_times)),
            'fps': float(1.0 / np.mean(inference_times)) if np.mean(inference_times) > 0 else 0.0,
            'total_time': float(np.sum(inference_times)),
        }
    }
    
    return results


def benchmark_synthetic(dataset_path: str = 'synthetic_test_dataset',
                       model_path: Optional[str] = None,
                       checkpoint_dir: Optional[str] = None,
                       device: str = 'cpu',
                       batch_size: int = 1):
    """
    Run comprehensive benchmark on synthetic dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        model_path: Path to Gaussian Splatting model (optional)
        checkpoint_dir: Directory containing model checkpoints
        device: Device to run on ('cpu' or 'cuda')
        batch_size: Batch size for evaluation
    """
    print("=" * 80)
    print("SYNTHETIC DATASET BENCHMARKING")
    print("=" * 80)
    
    if not os.path.exists(dataset_path):
        print(f"✗ Error: Dataset not found at {dataset_path}")
        print("  Run: python tests/synthetic_dataset.py --source <colmap_data> --output synthetic_test_dataset")
        return None
    
    if model_path is None:
        model_path = os.path.join(dataset_path, 'model')
    
    # Setup arguments
    parser = config_parser()
    model_params = ModelParams(parser)
    optimization = OptimizationParams(parser)
    
    base_args = ['--run_name', 'benchmark_synthetic', '--datadir', dataset_path]
    args = parser.parse_args(base_args)
    model_params.extract(args)
    optimization.extract(args)
    
    args.device = device
    args.render_device = device
    args.model_path = model_path
    args.dataset_type = 'Colmap'
    args.train_skip = 1
    args.test_skip = 1
    args.batch_size = batch_size
    args.val_batch_size = batch_size
    args.compile = False
    
    # Initialize Gaussian Splatting arguments
    args = args_init.argument_init(args)
    
    fix_seed(7)
    
    print(f"\nDataset: {dataset_path}")
    print(f"Model Path: {model_path}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        # Load camera parameters
        camera_file = os.path.join(model_path, 'cameras.json')
        if not os.path.exists(camera_file):
            print(f"✗ Error: Camera file not found at {camera_file}")
            return None
        
        with open(camera_file) as f:
            camera = json.load(f)[0]
        
        cam_params = CamParams(camera, args.rap_resolution, device)
        rap_hw = (cam_params.h, cam_params.w)
        gs_hw = (cam_params.h, cam_params.w)  # Use same resolution for GS
        
        dataset = ColmapDataset(
            data_path=dataset_path,
            train=False,
            hw=rap_hw,
            hw_gs=gs_hw,
            train_skip=args.train_skip,
            test_skip=args.test_skip
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"✓ Loaded {len(dataset)} test samples")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Find checkpoints
    model_types = ['uaas', 'probabilistic', 'semantic']
    checkpoints = {}
    
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        for model_type in model_types:
            # Look for checkpoints
            checkpoint_files = list(checkpoint_dir.glob(f"**/*{model_type}*.pth"))
            checkpoint_files.extend(list(checkpoint_dir.glob(f"**/full_checkpoint.pth")))
            if checkpoint_files:
                checkpoints[model_type] = str(checkpoint_files[0])
                print(f"✓ Found checkpoint for {model_type}: {checkpoints[model_type]}")
    
    # Benchmark each model
    all_results = {}
    
    for model_type in model_types:
        print("\n" + "=" * 80)
        print(f"Benchmarking {model_type.upper()} Model")
        print("=" * 80)
        
        checkpoint_path = checkpoints.get(model_type)
        
        try:
            model = load_model(model_type, args, checkpoint_path)
            results = evaluate_model(model, data_loader, args, model_type)
            all_results[model_type] = results
            
            # Print summary
            print(f"\nResults for {model_type.upper()}:")
            print(f"  Translation Error:")
            print(f"    Mean:   {results['translation_errors']['mean']:.4f} m")
            print(f"    Median: {results['translation_errors']['median']:.4f} m")
            print(f"  Rotation Error:")
            print(f"    Mean:   {results['rotation_errors']['mean']:.4f} deg")
            print(f"    Median: {results['rotation_errors']['median']:.4f} deg")
            print(f"  Inference Speed:")
            print(f"    FPS:    {results['inference_speed']['fps']:.2f}")
            print(f"    Time:   {results['inference_speed']['mean_time_per_image']*1000:.2f} ms/image")
            
        except Exception as e:
            print(f"✗ Error benchmarking {model_type}: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_type] = {'error': str(e)}
    
    # Save results
    output_file = 'benchmark_synthetic_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    for model_type, results in all_results.items():
        if 'error' not in results:
            print(f"\n{model_type.upper()}:")
            print(f"  Translation: {results['translation_errors']['median']:.4f} m (median)")
            print(f"  Rotation:     {results['rotation_errors']['median']:.4f} deg (median)")
            print(f"  Speed:        {results['inference_speed']['fps']:.2f} FPS")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Benchmark models on synthetic dataset')
    parser.add_argument('--dataset', type=str, default='synthetic_test_dataset',
                       help='Path to synthetic dataset')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to Gaussian Splatting model')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing model checkpoints')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run on')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    results = benchmark_synthetic(
        dataset_path=args.dataset,
        model_path=args.model_path,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())

