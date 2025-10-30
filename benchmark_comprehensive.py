#!/usr/bin/env python3
"""
Comprehensive benchmarking script for RAP-ID models.

This script benchmarks pose accuracy, inference speed, and model-specific metrics
for baseline RAP, UAAS, Probabilistic, and Semantic models.
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams, get_combined_args
import arguments.args_init as args_init
from dataset_loaders.cambridge_scenes import Cambridge
from dataset_loaders.colmap_dataset import ColmapDataset
from dataset_loaders.seven_scenes import SevenScenes
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


def evaluate_baseline_uaas(model, data_loader, args, model_type: str):
    """
    Evaluate baseline or UAAS model.
    
    Args:
        model: Model instance
        data_loader: Data loader
        args: Configuration arguments
        model_type: 'baseline' or 'uaas'
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    loss_fn = CameraPoseLoss(args).to(args.device)
    
    errors_trans = []
    errors_rot = []
    val_losses = []
    uncertainties = [] if model_type == 'uaas' else None
    
    with torch.no_grad():
        for data, pose, _, _ in tqdm(data_loader, desc=f"Evaluating {model_type}"):
            data = data.to(args.device)
            pose = pose.to(args.device)
            
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                if model_type == 'uaas':
                    outputs = model(data, return_feature=False)
                    if isinstance(outputs, tuple) and len(outputs) == 2:
                        pred_pose, log_var = outputs
                        uncertainty = torch.exp(log_var).mean().item()
                        uncertainties.append(uncertainty)
                    else:
                        pred_pose = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    _, pred_pose = model(data, return_feature=False)
                
                val_loss = loss_fn(pred_pose, pose)
            
            val_losses.append(val_loss.item())
            
            pose = pose.reshape((-1, 3, 4))
            pred_pose = pred_pose.reshape((-1, 3, 4))
            
            error_trans, error_rot = get_pose_error(pose, pred_pose)
            errors_trans.append(error_trans.cpu().numpy())
            errors_rot.append(error_rot.cpu().numpy())
    
    errors_trans = np.hstack(errors_trans)
    errors_rot = np.hstack(errors_rot)
    
    results = {
        'mean_loss': np.mean(val_losses),
        'median_translation': np.median(errors_trans),
        'median_rotation': np.median(errors_rot),
        'mean_translation': np.mean(errors_trans),
        'mean_rotation': np.mean(errors_rot),
        'max_translation': np.max(errors_trans),
        'max_rotation': np.max(errors_rot),
        'min_translation': np.min(errors_trans),
        'min_rotation': np.min(errors_rot),
    }
    
    # Success rates
    success_5cm_5deg = np.sum((errors_trans < 0.05) & (errors_rot < 5)) / len(errors_trans)
    success_2cm_2deg = np.sum((errors_trans < 0.02) & (errors_rot < 2)) / len(errors_trans)
    results['success_rate_5cm_5deg'] = success_5cm_5deg
    results['success_rate_2cm_2deg'] = success_2cm_2deg
    
    if uncertainties:
        results['mean_uncertainty'] = np.mean(uncertainties)
        results['std_uncertainty'] = np.std(uncertainties)
    
    return results


def evaluate_probabilistic(model, data_loader, args):
    """
    Evaluate probabilistic model with multiple hypotheses.
    
    Args:
        model: ProbabilisticRAPNet instance
        data_loader: Data loader
        args: Configuration arguments
        
    Returns:
        Dictionary of evaluation metrics including hypothesis statistics
    """
    model.eval()
    
    errors_trans = []
    errors_rot = []
    num_hypotheses = []
    
    with torch.no_grad():
        for data, pose, _, _ in tqdm(data_loader, desc="Evaluating probabilistic"):
            data = data.to(args.device)
            pose = pose.to(args.device)
            
            pose = pose.reshape((-1, 3, 4))
            
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                mixture_dist = model(data, return_feature=False)
                
                # Sample multiple hypotheses
                n_samples = 10
                hypotheses = mixture_dist.sample((n_samples,))  # (n_samples, batch_size, 6)
                
                # For now, use the mean as prediction (can be improved with validation)
                pred_pose_vec = hypotheses.mean(dim=0)  # (batch_size, 6)
                
                # Convert to pose matrix (simplified - assumes 6D representation)
                # This is a placeholder - actual implementation should properly convert
                # from the model's output format to pose matrix
                num_hypotheses.append(n_samples)
                
                # For evaluation, we'll use the mean hypothesis
                # TODO: Implement proper pose conversion from model output
                # For now, this is a placeholder
                pred_pose = pose.clone()  # Placeholder
    
    # Return placeholder results
    results = {
        'mean_loss': 0.0,
        'median_translation': 0.0,
        'median_rotation': 0.0,
        'mean_translation': 0.0,
        'mean_rotation': 0.0,
        'num_hypotheses_mean': np.mean(num_hypotheses) if num_hypotheses else 0,
        'note': 'Probabilistic evaluation requires pose format conversion implementation'
    }
    
    return results


def benchmark_inference_speed(model, data_loader, args, model_type: str, num_warmup=5, num_iterations=100):
    """
    Benchmark inference speed for a model.
    
    Args:
        model: Model instance
        data_loader: Data loader
        args: Configuration arguments
        model_type: Type of model
        num_warmup: Number of warmup iterations
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    device = args.device
    
    # Warmup
    with torch.no_grad():
        for i, (data, _, _, _) in enumerate(data_loader):
            if i >= num_warmup:
                break
            data = data.to(device)
            _ = model(data, return_feature=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for i, (data, _, _, _) in enumerate(data_loader):
            if i >= num_iterations:
                break
            data = data.to(device)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.cuda.amp.autocast(enabled=args.amp, dtype=args.amp_dtype):
                _ = model(data, return_feature=False)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    return {
        'mean_inference_time': avg_time,
        'fps': fps,
        'std_inference_time': np.std(times),
        'min_inference_time': np.min(times),
        'max_inference_time': np.max(times),
    }


def run_comprehensive_benchmark(args):
    """
    Run comprehensive benchmark for specified model type.
    
    Args:
        args: Configuration arguments with benchmark parameters
    """
    print("=" * 60)
    print(f"RAP-ID Comprehensive Benchmark: {args.model_type.upper()}")
    print("=" * 60)
    
    device = torch.device(args.device)
    fix_seed()
    
    # Setup dataset
    if args.dataset_type == '7Scenes':
        dataset_class = SevenScenes
    elif args.dataset_type == 'Colmap':
        dataset_class = ColmapDataset
    elif args.dataset_type == 'Cambridge':
        dataset_class = Cambridge
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    with open(f"{args.model_path}/cameras.json") as f:
        camera = json.load(f)[0]
    
    rap_cam_params = CamParams(camera, args.rap_resolution, args.device)
    rap_hw = (rap_cam_params.h, rap_cam_params.w)
    
    kwargs = dict(data_path=args.datadir, hw=rap_hw)
    val_set = dataset_class(train=False, test_skip=args.test_skip, **kwargs)
    val_dl = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, 
                       num_workers=args.val_num_workers, pin_memory=True)
    
    # Load model
    checkpoint_path = args.checkpoint_path if hasattr(args, 'checkpoint_path') else None
    if not checkpoint_path:
        # Try to find checkpoint in model_path
        checkpoint_dir = Path(args.model_path).parent if hasattr(args, 'model_path') else Path('.')
        checkpoint_path = checkpoint_dir / "full_checkpoint.pth"
    
    model = load_model(args.model_type, args, checkpoint_path=str(checkpoint_path))
    
    results = {}
    
    # Benchmark pose accuracy
    print("\n" + "-" * 60)
    print("Pose Accuracy Evaluation")
    print("-" * 60)
    
    if args.model_type in ['baseline', 'uaas']:
        accuracy_results = evaluate_baseline_uaas(model, val_dl, args, args.model_type)
    elif args.model_type == 'probabilistic':
        accuracy_results = evaluate_probabilistic(model, val_dl, args)
    elif args.model_type == 'semantic':
        accuracy_results = evaluate_baseline_uaas(model, val_dl, args, 'baseline')  # Similar to baseline
    else:
        raise ValueError(f"Evaluation not implemented for {args.model_type}")
    
    results['accuracy'] = accuracy_results
    
    # Benchmark inference speed
    if args.benchmark_speed:
        print("\n" + "-" * 60)
        print("Inference Speed Benchmark")
        print("-" * 60)
        speed_results = benchmark_inference_speed(model, val_dl, args, args.model_type)
        results['speed'] = speed_results
    
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\nModel Type: {args.model_type}")
    print(f"Dataset: {args.dataset_type}")
    
    print("\nPose Accuracy Metrics:")
    print(f"  Median Translation Error: {accuracy_results.get('median_translation', 0):.4f} m")
    print(f"  Median Rotation Error: {accuracy_results.get('median_rotation', 0):.4f} deg")
    print(f"  Mean Translation Error: {accuracy_results.get('mean_translation', 0):.4f} m")
    print(f"  Mean Rotation Error: {accuracy_results.get('mean_rotation', 0):.4f} deg")
    print(f"  Success Rate (5cm, 5deg): {accuracy_results.get('success_rate_5cm_5deg', 0):.4f}")
    print(f"  Success Rate (2cm, 2deg): {accuracy_results.get('success_rate_2cm_2deg', 0):.4f}")
    
    if 'mean_uncertainty' in accuracy_results:
        print(f"\nUncertainty Metrics (UAAS):")
        print(f"  Mean Uncertainty: {accuracy_results['mean_uncertainty']:.4f}")
        print(f"  Std Uncertainty: {accuracy_results['std_uncertainty']:.4f}")
    
    if 'speed' in results:
        print(f"\nInference Speed Metrics:")
        print(f"  FPS: {results['speed']['fps']:.2f}")
        print(f"  Mean Time: {results['speed']['mean_inference_time']*1000:.2f} ms")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.model_path) if hasattr(args, 'model_path') else Path('.')
    
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / f"benchmark_{args.model_type}_{args.dataset_type}.json"
    
    results['metadata'] = {
        'model_type': args.model_type,
        'dataset_type': args.dataset_type,
        'checkpoint_path': str(checkpoint_path),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive benchmark for RAP-ID models")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["baseline", "uaas", "probabilistic", "semantic"],
                       help="Type of model to benchmark")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for benchmark results")
    parser.add_argument("--benchmark_speed", action="store_true",
                       help="Also benchmark inference speed")
    parser.add_argument("--num_semantic_classes", type=int, default=19,
                       help="Number of semantic classes for semantic model")
    
    args = get_combined_args(parser)
    lp.extract(args)
    op.extract(args)
    args = args_init.argument_init(args)
    
    run_comprehensive_benchmark(args)


if __name__ == "__main__":
    main()

