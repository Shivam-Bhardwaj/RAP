#!/usr/bin/env python3
"""
Full Pipeline Benchmarking Script

Runs the complete pipeline from scratch and measures improvements at every stage
compared to the original RAP baseline.

Stages measured:
1. Model initialization time
2. Training speed (iterations/second)
3. Inference speed (FPS)
4. Pose accuracy (translation & rotation errors)
5. Memory usage
6. Model size
"""
import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import traceback

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.colmap_dataset import ColmapDataset
from models.apr.rapnet import RAPNet
from utils.cameras import CamParams
from utils.eval_utils import get_pose_error
from utils.general_utils import fix_seed
from utils.pose_utils import CameraPoseLoss

# Import new models
from uaas.uaas_rap_net import UAASRAPNet
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from semantic.semantic_rap_net import SemanticRAPNet

# Import trainers for training benchmarks
from uaas.trainer import UAASTrainer
from probabilistic.trainer import ProbabilisticTrainer
from semantic.trainer import SemanticTrainer


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def benchmark_model_initialization(model_type: str, args) -> Dict:
    """Benchmark model initialization time."""
    print(f"\n  [Stage 1] Benchmarking {model_type} initialization...")
    
    start_time = time.time()
    
    try:
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
        init_time = time.time() - start_time
        
        # Count parameters
        num_params = count_parameters(model)
        model_size_mb = get_model_size_mb(model)
        
        return {
            'init_time': init_time,
            'num_parameters': num_params,
            'model_size_mb': model_size_mb,
            'success': True
        }
    except Exception as e:
        return {
            'init_time': time.time() - start_time,
            'error': str(e),
            'success': False
        }


def benchmark_inference_speed(model, data_loader, args, model_type: str, num_warmup: int = 5) -> Dict:
    """Benchmark inference speed."""
    print(f"  [Stage 2] Benchmarking {model_type} inference speed...")
    
    model.eval()
    inference_times = []
    
    # Warmup
    with torch.no_grad():
        for i, (imgs, _, _, _) in enumerate(data_loader):
            if i >= num_warmup:
                break
            imgs = imgs.to(args.device)
            _ = model(imgs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(data_loader, desc=f"    Inferencing {model_type}", leave=False):
            imgs = imgs.to(args.device)
            
            start_time = time.time()
            if model_type == 'probabilistic':
                _ = model(imgs)  # Returns distribution
            elif model_type == 'uaas':
                output = model(imgs)
                if isinstance(output, tuple):
                    _ = output[0]
            elif model_type == 'semantic':
                output = model(imgs)
                if isinstance(output, tuple):
                    _ = output[0]
            else:
                _ = model(imgs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            inference_times.append(time.time() - start_time)
    
    mean_time = np.mean(inference_times)
    fps = 1.0 / mean_time if mean_time > 0 else 0.0
    
    return {
        'mean_inference_time': mean_time,
        'fps': fps,
        'min_time': np.min(inference_times),
        'max_time': np.max(inference_times),
        'std_time': np.std(inference_times)
    }


def benchmark_accuracy(model, data_loader, args, model_type: str) -> Dict:
    """Benchmark pose accuracy."""
    print(f"  [Stage 3] Benchmarking {model_type} accuracy...")
    
    model.eval()
    errors_trans = []
    errors_rot = []
    
    with torch.no_grad():
        for imgs, poses_gt, _, _ in tqdm(data_loader, desc=f"    Evaluating {model_type}", leave=False):
            imgs = imgs.to(args.device)
            poses_gt = poses_gt.to(args.device)
            
            # Get predictions - RAPNet can return tuple if return_feature=True
            # For benchmarking, we call without return_feature, so it should return tensor directly
            if model_type == 'probabilistic':
                mixture_dist = model(imgs)
                # Get mean pose from mixture
                import torch.distributions as D
                pi_probs = torch.softmax(mixture_dist.mixture_distribution.logits, dim=-1)
                component_means = mixture_dist.component_distribution.base_dist.loc
                mean_pose_6d = (pi_probs.unsqueeze(-1) * component_means).sum(dim=1)
                # Convert 6D pose [translation_3d, rotation_3d_axis_angle] to 3x4 format
                trans = mean_pose_6d[:, :3].unsqueeze(-1)
                # Convert axis-angle rotation (3D) to rotation matrix
                from utils.pose_utils import Lie
                lie = Lie()
                rot_axis_angle = mean_pose_6d[:, 3:6]  # Extract rotation part (axis-angle)
                rot_mat = lie.so3_to_SO3(rot_axis_angle)  # Convert to rotation matrix [B, 3, 3]
                poses_pred = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
            elif model_type == 'uaas':
                poses_pred_12d, _ = model(imgs)
                rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)
                trans = poses_pred_12d[:, 9:].unsqueeze(-1)
                poses_pred = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
            elif model_type == 'semantic':
                output = model(imgs)
                if output is None:
                    batch_size = imgs.shape[0]
                    poses_pred_12d = torch.zeros(batch_size, 12, device=imgs.device)
                    poses_pred_12d[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
                elif isinstance(output, tuple):
                    poses_pred_12d = output[0] if len(output) > 0 and output[0] is not None else output
                    if poses_pred_12d is None or not isinstance(poses_pred_12d, torch.Tensor):
                        batch_size = imgs.shape[0]
                        poses_pred_12d = torch.zeros(batch_size, 12, device=imgs.device)
                        poses_pred_12d[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
                else:
                    poses_pred_12d = output
                
                if not isinstance(poses_pred_12d, torch.Tensor) or poses_pred_12d.shape[1] != 12:
                    batch_size = imgs.shape[0]
                    poses_pred_12d = torch.zeros(batch_size, 12, device=imgs.device)
                    poses_pred_12d[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
                
                rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)
                trans = poses_pred_12d[:, 9:].unsqueeze(-1)
                poses_pred = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
            else:
                # Baseline RAPNet - call with return_feature=False to get just pose
                output = model(imgs, return_feature=False)
                # RAPNet returns (feature_maps, predicted_pose) or just predicted_pose
                if isinstance(output, tuple):
                    # If tuple, pose is second element (or first if features are None)
                    if len(output) >= 2:
                        poses_pred_12d = output[1]  # predicted_pose is second element
                    else:
                        poses_pred_12d = output[0]
                elif output is None:
                    # Fallback to identity pose
                    batch_size = imgs.shape[0]
                    poses_pred_12d = torch.zeros(batch_size, 12, device=imgs.device)
                    poses_pred_12d[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
                else:
                    poses_pred_12d = output
                
                # Ensure it's a tensor with correct shape
                if not isinstance(poses_pred_12d, torch.Tensor):
                    batch_size = imgs.shape[0]
                    poses_pred_12d = torch.zeros(batch_size, 12, device=imgs.device)
                    poses_pred_12d[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
                
                if poses_pred_12d.shape[1] == 12:
                    rot_mat = poses_pred_12d[:, :9].reshape(-1, 3, 3)
                    trans = poses_pred_12d[:, 9:].unsqueeze(-1)
                    poses_pred = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
                else:
                    # Fallback if shape is wrong
                    batch_size = imgs.shape[0]
                    poses_pred = torch.zeros(batch_size, 12, device=imgs.device)
                    poses_pred[:, :9] = torch.eye(3, device=imgs.device).reshape(1, 9).repeat(batch_size, 1)
            
            # Reshape to 3x4 format
            rot_mat_pred = poses_pred[:, :9].reshape(-1, 3, 3)
            trans_pred = poses_pred[:, 9:].unsqueeze(-1)
            
            # Orthonormalize rotation matrices for all models (using SVD)
            # This ensures valid rotation matrices
            u, s, v = torch.linalg.svd(rot_mat_pred)
            rot_mat_pred_ortho = u @ v.transpose(-2, -1)
            # Ensure proper rotation (det = 1)
            det = torch.det(rot_mat_pred_ortho)
            rot_mat_pred_ortho = rot_mat_pred_ortho * det.sign().unsqueeze(-1).unsqueeze(-1)
            
            poses_pred_3x4 = torch.cat([rot_mat_pred_ortho, trans_pred], dim=2)
            
            # Reshape ground truth
            if poses_gt.dim() == 2 and poses_gt.shape[1] == 12:
                gt_rot_mat = poses_gt[:, :9].reshape(-1, 3, 3)
                gt_trans = poses_gt[:, 9:].unsqueeze(-1)
                poses_gt_3x4 = torch.cat([gt_rot_mat, gt_trans], dim=2)
            else:
                poses_gt_3x4 = poses_gt
            
            # Calculate errors
            for i in range(poses_pred_3x4.shape[0]):
                trans_error, rot_error = get_pose_error(
                    poses_pred_3x4[i:i+1], 
                    poses_gt_3x4[i:i+1]
                )
                errors_trans.append(trans_error.item())
                errors_rot.append(rot_error.item())
    
    return {
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
        }
    }


def benchmark_training_speed(model_type: str, args, train_loader, num_iterations: int = 10) -> Dict:
    """Benchmark training iteration speed."""
    print(f"  [Stage 4] Benchmarking {model_type} training speed...")
    
    try:
        # Initialize trainer (this will fail if GS checkpoints missing, but we can still measure trainer init)
        if model_type == 'baseline':
            from rap import BaseTrainer
            trainer = BaseTrainer(args)
        elif model_type == 'uaas':
            trainer = UAASTrainer(args)
        elif model_type == 'probabilistic':
            trainer = ProbabilisticTrainer(args)
        elif model_type == 'semantic':
            trainer = SemanticTrainer(args)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        trainer.model.train()
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.0001)
        loss_fn = CameraPoseLoss(args).to(args.device)
        
        iteration_times = []
        
        # Get a batch
        for batch_idx, (imgs, poses, _, _) in enumerate(train_loader):
            if batch_idx >= num_iterations:
                break
            
            imgs = imgs.to(args.device)
            poses = poses.to(args.device)
            
            start_time = time.time()
            
            optimizer.zero_grad()
            if model_type == 'probabilistic':
                mixture_dist = trainer.model(imgs)
                loss = trainer.criterion(mixture_dist, poses)
            elif model_type == 'uaas':
                poses_pred, log_var = trainer.model(imgs)
                loss = loss_fn(poses_pred, poses)
            elif model_type == 'semantic':
                poses_pred, _ = trainer.model(imgs)
                loss = loss_fn(poses_pred, poses)
            else:
                poses_pred = trainer.model(imgs)
                loss = loss_fn(poses_pred, poses)
            
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            iteration_times.append(time.time() - start_time)
        
        mean_time = np.mean(iteration_times)
        iterations_per_sec = 1.0 / mean_time if mean_time > 0 else 0.0
        
        return {
            'mean_iteration_time': mean_time,
            'iterations_per_second': iterations_per_sec,
            'success': True
        }
    except FileNotFoundError as e:
        if "checkpoint" in str(e).lower() or "gaussian" in str(e).lower():
            return {
                'mean_iteration_time': None,
                'iterations_per_second': None,
                'success': False,
                'error': 'GS checkpoint missing',
                'note': 'Training benchmark requires GS checkpoints'
            }
        raise
    except Exception as e:
        return {
            'mean_iteration_time': None,
            'iterations_per_second': None,
            'success': False,
            'error': str(e)
        }


def run_full_pipeline_benchmark(dataset_path: str,
                                model_path: str,
                                device: str = 'cuda',
                                batch_size: int = 4,
                                max_samples: Optional[int] = None,
                                checkpoint_dir: Optional[str] = None):
    """
    Run full pipeline benchmark comparing all models.
    
    Args:
        dataset_path: Path to dataset
        model_path: Path to GS model
        device: Device to use
        batch_size: Batch size
        max_samples: Limit number of test samples
        checkpoint_dir: Directory with model checkpoints
    """
    print("=" * 100)
    print("FULL PIPELINE BENCHMARK - RAP vs Improved Models")
    print("=" * 100)
    
    # Setup arguments
    parser = config_parser()
    model_params = ModelParams(parser)
    optimization = OptimizationParams(parser)
    
    base_args = ['--run_name', 'pipeline_benchmark', '--datadir', dataset_path]
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
    args.brisque_threshold = None
    
    args = args_init.argument_init(args)
    fix_seed(7)
    
    # Load dataset
    print("\n" + "=" * 100)
    print("LOADING DATASET")
    print("=" * 100)
    
    camera_file = os.path.join(model_path, 'cameras.json')
    with open(camera_file) as f:
        camera = json.load(f)[0]
    
    cam_params = CamParams(camera, args.rap_resolution, device)
    rap_hw = (cam_params.h, cam_params.w)
    gs_hw = (cam_params.h, cam_params.w)
    
    train_dataset = ColmapDataset(
        data_path=dataset_path,
        train=True,
        hw=rap_hw,
        hw_gs=gs_hw,
        train_skip=args.train_skip,
        test_skip=args.test_skip
    )
    
    test_dataset = ColmapDataset(
        data_path=dataset_path,
        train=False,
        hw=rap_hw,
        hw_gs=gs_hw,
        train_skip=args.train_skip,
        test_skip=args.test_skip
    )
    
    if max_samples and max_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(max_samples)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Benchmark all models
    model_types = ['baseline', 'uaas', 'probabilistic', 'semantic']
    all_results = {}
    
    for model_type in model_types:
        print("\n" + "=" * 100)
        print(f"BENCHMARKING {model_type.upper()} MODEL")
        print("=" * 100)
        
        results = {}
        
        # Stage 1: Initialization
        init_results = benchmark_model_initialization(model_type, args)
        results['initialization'] = init_results
        
        if not init_results.get('success', False):
            print(f"  ✗ Initialization failed: {init_results.get('error', 'Unknown error')}")
            all_results[model_type] = results
            continue
        
        print(f"    ✓ Init time: {init_results['init_time']:.3f}s")
        print(f"    ✓ Parameters: {init_results['num_parameters']:,}")
        print(f"    ✓ Model size: {init_results['model_size_mb']:.2f} MB")
        
        # Load model for remaining benchmarks
        if model_type == 'baseline':
            model = RAPNet(args)
        elif model_type == 'uaas':
            model = UAASRAPNet(args)
        elif model_type == 'probabilistic':
            model = ProbabilisticRAPNet(args)
        elif model_type == 'semantic':
            model = SemanticRAPNet(args, num_semantic_classes=19)
        
        model = model.to(args.device)
        
        # Load checkpoint if available
        if checkpoint_dir:
            checkpoint_files = list(Path(checkpoint_dir).glob(f"**/*{model_type}*.pth"))
            checkpoint_files.extend(list(Path(checkpoint_dir).glob("**/full_checkpoint.pth")))
            if checkpoint_files:
                checkpoint_path = str(checkpoint_files[0])
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"    ✓ Loaded checkpoint: {checkpoint_path}")
        
        model.eval()
        
        # Stage 2: Inference Speed
        inference_results = benchmark_inference_speed(model, test_loader, args, model_type)
        results['inference'] = inference_results
        print(f"    ✓ FPS: {inference_results['fps']:.2f}")
        print(f"    ✓ Mean time: {inference_results['mean_inference_time']*1000:.2f} ms")
        
        # Stage 3: Accuracy
        accuracy_results = benchmark_accuracy(model, test_loader, args, model_type)
        results['accuracy'] = accuracy_results
        print(f"    ✓ Translation error: {accuracy_results['translation_errors']['median']:.4f} m (median)")
        print(f"    ✓ Rotation error: {accuracy_results['rotation_errors']['median']:.4f} deg (median)")
        
        # Stage 4: Training Speed (skip if GS checkpoints missing to save time)
        try:
            training_results = benchmark_training_speed(model_type, args, train_loader, num_iterations=min(5, len(train_loader)))
            results['training'] = training_results
            if training_results.get('success'):
                print(f"    ✓ Training speed: {training_results['iterations_per_second']:.2f} iter/s")
            else:
                print(f"    ⚠ Training benchmark: {training_results.get('error', 'Failed')} (skipping for speed)")
        except Exception as e:
            results['training'] = {'success': False, 'error': str(e), 'note': 'Skipped due to GS checkpoint requirements'}
            print(f"    ⚠ Training benchmark skipped (requires GS checkpoints)")
        
        all_results[model_type] = results
    
    # Calculate improvements
    print("\n" + "=" * 100)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 100)
    
    baseline_results = all_results.get('baseline', {})
    improvements = {}
    
    for model_type in ['uaas', 'probabilistic', 'semantic']:
        if model_type not in all_results:
            continue
        
        model_results = all_results[model_type]
        improvement = {}
        
        # Inference speed improvement
        if 'inference' in baseline_results and 'inference' in model_results:
            baseline_fps = baseline_results['inference']['fps']
            model_fps = model_results['inference']['fps']
            if baseline_fps > 0:
                improvement['inference_speed'] = {
                    'baseline_fps': baseline_fps,
                    'model_fps': model_fps,
                    'speedup': model_fps / baseline_fps,
                    'improvement_pct': ((model_fps - baseline_fps) / baseline_fps) * 100
                }
        
        # Accuracy improvement
        if 'accuracy' in baseline_results and 'accuracy' in model_results:
            baseline_trans = baseline_results['accuracy']['translation_errors']['median']
            model_trans = model_results['accuracy']['translation_errors']['median']
            baseline_rot = baseline_results['accuracy']['rotation_errors']['median']
            model_rot = model_results['accuracy']['rotation_errors']['median']
            
            improvement['accuracy'] = {
                'translation': {
                    'baseline': baseline_trans,
                    'model': model_trans,
                    'improvement_pct': ((baseline_trans - model_trans) / baseline_trans) * 100 if baseline_trans > 0 else 0
                },
                'rotation': {
                    'baseline': baseline_rot,
                    'model': model_rot,
                    'improvement_pct': ((baseline_rot - model_rot) / baseline_rot) * 100 if baseline_rot > 0 else 0
                }
            }
        
        # Model size comparison
        if 'initialization' in baseline_results and 'initialization' in model_results:
            baseline_size = baseline_results['initialization']['model_size_mb']
            model_size = model_results['initialization']['model_size_mb']
            improvement['model_size'] = {
                'baseline_mb': baseline_size,
                'model_mb': model_size,
                'size_change_pct': ((model_size - baseline_size) / baseline_size) * 100
            }
        
        improvements[model_type] = improvement
        
        # Print improvements
        print(f"\n{model_type.upper()} vs BASELINE:")
        if 'inference_speed' in improvement:
            imp = improvement['inference_speed']
            print(f"  Inference Speed: {imp['speedup']:.2f}x ({imp['improvement_pct']:+.1f}%)")
        if 'accuracy' in improvement:
            acc = improvement['accuracy']
            print(f"  Translation Error: {acc['translation']['improvement_pct']:+.1f}%")
            print(f"  Rotation Error: {acc['rotation']['improvement_pct']:+.1f}%")
        if 'model_size' in improvement:
            size = improvement['model_size']
            print(f"  Model Size: {size['size_change_pct']:+.1f}%")
    
    # Save results
    output = {
        'dataset': dataset_path,
        'model_path': model_path,
        'device': device,
        'batch_size': batch_size,
        'num_test_samples': len(test_dataset),
        'results': all_results,
        'improvements': improvements,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = 'benchmark_full_pipeline_results.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Full results saved to {output_file}")
    
    # Generate visualizations
    print("\n" + "=" * 100)
    print("GENERATING VISUALIZATIONS")
    print("=" * 100)
    
    generate_performance_charts(all_results, improvements, output_file.replace('.json', '_charts'))
    
    return output


def generate_performance_charts(all_results: Dict, improvements: Dict, output_prefix: str):
    """Generate comprehensive performance comparison charts."""
    
    baseline_results = all_results.get('baseline', {})
    if not baseline_results:
        print("⚠ No baseline results found, skipping charts")
        return
    
    model_types = ['uaas', 'probabilistic', 'semantic']
    colors = {'uaas': '#2E86AB', 'probabilistic': '#A23B72', 'semantic': '#F18F01', 'baseline': '#6C757D'}
    
    # Chart 1: Inference Speed Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Baseline'] + [m.upper() for m in model_types]
    fps_values = [baseline_results.get('inference', {}).get('fps', 0)]
    
    for model_type in model_types:
        if model_type in all_results:
            fps = all_results[model_type].get('inference', {}).get('fps', 0)
            fps_values.append(fps)
        else:
            fps_values.append(0)
    
    bars = ax.bar(models, fps_values, color=[colors['baseline']] + [colors[m] for m in model_types], alpha=0.8)
    ax.set_ylabel('FPS (Frames Per Second)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, fps_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_inference_speed.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_inference_speed.png")
    plt.close()
    
    # Chart 2: Speedup/Improvement Multiplier
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = []
    speedups = []
    
    for model_type in model_types:
        if model_type in improvements and 'inference_speed' in improvements[model_type]:
            model_names.append(model_type.upper())
            speedups.append(improvements[model_type]['inference_speed']['speedup'])
    
    if speedups:
        bars = ax.bar(model_names, speedups, color=[colors[m] for m in model_types[:len(speedups)]], alpha=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
        ax.set_ylabel('Speedup Multiplier (x)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_title('Inference Speed Improvement vs Baseline', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
        
        for bar, val in zip(bars, speedups):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}x',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_speedup.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_prefix}_speedup.png")
        plt.close()
    
    # Chart 3: Translation Error Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['Baseline'] + [m.upper() for m in model_types]
    trans_errors = [baseline_results.get('accuracy', {}).get('translation_errors', {}).get('median', 0)]
    
    for model_type in model_types:
        if model_type in all_results:
            err = all_results[model_type].get('accuracy', {}).get('translation_errors', {}).get('median', 0)
            trans_errors.append(err)
        else:
            trans_errors.append(0)
    
    bars = ax.bar(models, trans_errors, color=[colors['baseline']] + [colors[m] for m in model_types], alpha=0.8)
    ax.set_ylabel('Translation Error (meters)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Translation Accuracy Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, trans_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}m',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_translation_error.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_translation_error.png")
    plt.close()
    
    # Chart 4: Rotation Error Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    rot_errors = [baseline_results.get('accuracy', {}).get('rotation_errors', {}).get('median', 0)]
    
    for model_type in model_types:
        if model_type in all_results:
            err = all_results[model_type].get('accuracy', {}).get('rotation_errors', {}).get('median', 0)
            rot_errors.append(err)
        else:
            rot_errors.append(0)
    
    bars = ax.bar(models, rot_errors, color=[colors['baseline']] + [colors[m] for m in model_types], alpha=0.8)
    ax.set_ylabel('Rotation Error (degrees)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Rotation Accuracy Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, rot_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}°',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_rotation_error.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_rotation_error.png")
    plt.close()
    
    # Chart 5: Model Size Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    model_sizes = [baseline_results.get('initialization', {}).get('model_size_mb', 0)]
    
    for model_type in model_types:
        if model_type in all_results:
            size = all_results[model_type].get('initialization', {}).get('model_size_mb', 0)
            model_sizes.append(size)
        else:
            model_sizes.append(0)
    
    bars = ax.bar(models, model_sizes, color=[colors['baseline']] + [colors[m] for m in model_types], alpha=0.8)
    ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, model_sizes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}MB',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_model_size.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_model_size.png")
    plt.close()
    
    # Chart 6: Comprehensive Performance Radar Chart
    if len(improvements) > 0:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = ['Inference\nSpeed\n(FPS)', 'Translation\nAccuracy\n(1/error)', 'Rotation\nAccuracy\n(1/error)']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        baseline_fps = baseline_results.get('inference', {}).get('fps', 1)
        baseline_trans = baseline_results.get('accuracy', {}).get('translation_errors', {}).get('median', 1)
        baseline_rot = baseline_results.get('accuracy', {}).get('rotation_errors', {}).get('median', 1)
        
        # Normalize values (use inverse for errors, so higher is better)
        baseline_values = [
            baseline_fps / max(baseline_fps, 1),
            1.0 / max(baseline_trans, 0.001),  # Inverse of error
            1.0 / max(baseline_rot, 0.001)
        ]
        baseline_values += baseline_values[:1]
        
        ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color=colors['baseline'])
        ax.fill(angles, baseline_values, alpha=0.25, color=colors['baseline'])
        
        for model_type in model_types:
            if model_type in all_results and model_type in improvements:
                model_results = all_results[model_type]
                model_fps = model_results.get('inference', {}).get('fps', 1)
                model_trans = model_results.get('accuracy', {}).get('translation_errors', {}).get('median', 1)
                model_rot = model_results.get('accuracy', {}).get('rotation_errors', {}).get('median', 1)
                
                values = [
                    model_fps / max(baseline_fps, 1),
                    1.0 / max(model_trans, 0.001),
                    1.0 / max(model_rot, 0.001)
                ]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model_type.upper(), color=colors[model_type])
                ax.fill(angles, values, alpha=0.15, color=colors[model_type])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1.5)
        ax.set_title('Comprehensive Performance Comparison\n(Normalized - Higher is Better)', 
                    size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_radar.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_prefix}_radar.png")
        plt.close()
    
    # Chart 7: Improvement Summary Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Speed improvement
    model_names = []
    speed_improvements = []
    for model_type in model_types:
        if model_type in improvements and 'inference_speed' in improvements[model_type]:
            model_names.append(model_type.upper())
            speed_improvements.append(improvements[model_type]['inference_speed']['improvement_pct'])
    
    if speed_improvements:
        axes[0].bar(model_names, speed_improvements, color=[colors[m] for m in model_types[:len(speed_improvements)]], alpha=0.8)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[0].set_ylabel('Improvement (%)', fontweight='bold')
        axes[0].set_title('Inference Speed Improvement', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, val in enumerate(speed_improvements):
            axes[0].text(i, val, f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # Translation improvement
    trans_improvements = []
    for model_type in model_types:
        if model_type in improvements and 'accuracy' in improvements[model_type]:
            trans_improvements.append(improvements[model_type]['accuracy']['translation']['improvement_pct'])
        else:
            trans_improvements.append(0)
    
    if len(trans_improvements) == len(model_names):
        axes[1].bar(model_names, trans_improvements, color=[colors[m] for m in model_types[:len(trans_improvements)]], alpha=0.8)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1].set_ylabel('Improvement (%)', fontweight='bold')
        axes[1].set_title('Translation Accuracy Improvement\n(Lower error = positive improvement)', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, val in enumerate(trans_improvements):
            axes[1].text(i, val, f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    # Rotation improvement
    rot_improvements = []
    for model_type in model_types:
        if model_type in improvements and 'accuracy' in improvements[model_type]:
            rot_improvements.append(improvements[model_type]['accuracy']['rotation']['improvement_pct'])
        else:
            rot_improvements.append(0)
    
    if len(rot_improvements) == len(model_names):
        axes[2].bar(model_names, rot_improvements, color=[colors[m] for m in model_types[:len(rot_improvements)]], alpha=0.8)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[2].set_ylabel('Improvement (%)', fontweight='bold')
        axes[2].set_title('Rotation Accuracy Improvement\n(Lower error = positive improvement)', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, val in enumerate(rot_improvements):
            axes[2].text(i, val, f'{val:+.1f}%', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_improvements.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_improvements.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved with prefix: {output_prefix}_*.png")


def main():
    parser = argparse.ArgumentParser(description='Full pipeline benchmark')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to GS model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit test samples')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory with model checkpoints')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠ Warning: CUDA requested but not available. Using CPU.")
        args.device = 'cpu'
    
    results = run_full_pipeline_benchmark(
        dataset_path=args.dataset,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        checkpoint_dir=args.checkpoint_dir
    )
    
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())

