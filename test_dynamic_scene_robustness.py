#!/usr/bin/env python3
"""
Test robustness to dynamic scene changes.

This script tests how well models handle modified images:
- Inpainting (object removal/addition)
- Occlusion changes
- Lighting variations
- Structural changes

Measures pose prediction consistency when scene changes occur.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arguments import ModelParams, OptimizationParams
from arguments.options import config_parser
import arguments.args_init as args_init
from dataset_loaders.colmap_dataset import ColmapDataset
from models.apr.rapnet import RAPNet
from utils.eval_utils import get_pose_error
from utils.general_utils import fix_seed
from utils.cameras import CamParams

# Import enhanced models
from uaas.uaas_rap_net import UAASRAPNet
from probabilistic.probabilistic_rap_net import ProbabilisticRAPNet
from semantic.semantic_rap_net import SemanticRAPNet

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ImageModifier:
    """Apply various modifications to images to simulate dynamic scene changes."""
    
    def __init__(self, device='cpu'):
        self.device = device
    
    def inpaint_region(self, image: torch.Tensor, mask: torch.Tensor, method='telea') -> torch.Tensor:
        """
        Inpaint a region of the image.
        
        Args:
            image: [H, W, 3] or [3, H, W] tensor
            mask: [H, W] binary mask (1 = region to inpaint)
            method: 'telea' or 'ns' (Navier-Stokes)
        """
        # Convert to numpy for OpenCV
        if image.dim() == 3 and image.shape[0] == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.cpu().numpy().astype(np.uint8)
        
        # Apply inpainting
        if method == 'telea':
            inpainted = cv2.inpaint(img_np, mask_np, 3, cv2.INPAINT_TELEA)
        else:  # ns
            inpainted = cv2.inpaint(img_np, mask_np, 3, cv2.INPAINT_NS)
        
        # Convert back to tensor
        inpainted = (inpainted / 255.0).astype(np.float32) if img_np.max() <= 255 else inpainted.astype(np.float32)
        
        if image.dim() == 3 and image.shape[0] == 3:
            return torch.from_numpy(inpainted).permute(2, 0, 1).to(self.device)
        else:
            return torch.from_numpy(inpainted).to(self.device)
    
    def add_occlusion(self, image: torch.Tensor, num_patches: int = 3, patch_size: int = 50) -> torch.Tensor:
        """Add random occlusion patches to simulate objects blocking the scene."""
        modified = image.clone()
        H, W = image.shape[-2:]
        
        for _ in range(num_patches):
            # Random position
            y = np.random.randint(0, H - patch_size)
            x = np.random.randint(0, W - patch_size)
            
            # Add black patch (or random noise)
            if image.dim() == 3 and image.shape[0] == 3:
                modified[:, y:y+patch_size, x:x+patch_size] = 0.0
            else:
                modified[y:y+patch_size, x:x+patch_size] = 0.0
        
        return modified
    
    def modify_lighting(self, image: torch.Tensor, brightness: float = 0.3, contrast: float = 1.2) -> torch.Tensor:
        """Modify image brightness and contrast."""
        # Clamp to valid range
        image = torch.clamp(image, 0, 1)
        
        # Apply brightness and contrast
        modified = (image * contrast) + brightness
        modified = torch.clamp(modified, 0, 1)
        
        return modified
    
    def add_object_mask(self, image: torch.Tensor, center: Tuple[int, int], 
                       size: Tuple[int, int], fill_value: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a mask for object addition/removal."""
        H, W = image.shape[-2:]
        mask = torch.zeros(H, W, dtype=torch.float32, device=image.device)
        
        y, x = center
        h, w = size
        
        y1, y2 = max(0, y - h//2), min(H, y + h//2)
        x1, x2 = max(0, x - w//2), min(W, x + w//2)
        
        mask[y1:y2, x1:x2] = 1.0
        
        # Create modified image with filled region
        modified = image.clone()
        if image.dim() == 3 and image.shape[0] == 3:
            modified[:, y1:y2, x1:x2] = fill_value
        else:
            modified[y1:y2, x1:x2] = fill_value
        
        return modified, mask
    
    def blur_region(self, image: torch.Tensor, mask: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
        """Apply blur to a specific region."""
        # Convert to numpy for OpenCV
        if image.dim() == 3 and image.shape[0] == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
        mask_np = mask.cpu().numpy().astype(np.uint8)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
        
        # Apply mask
        result = np.where(mask_np[..., None] > 0, blurred, img_np)
        
        # Convert back
        result = (result / 255.0).astype(np.float32) if img_np.max() <= 255 else result.astype(np.float32)
        
        if image.dim() == 3 and image.shape[0] == 3:
            return torch.from_numpy(result).permute(2, 0, 1).to(self.device)
        else:
            return torch.from_numpy(result).to(self.device)


def test_model_robustness(model, data_loader, modifier: ImageModifier, args, 
                         model_type: str, modifications: List[str]) -> Dict:
    """Test model robustness to various image modifications."""
    model.eval()
    
    results = {
        'original': {'translation_errors': [], 'rotation_errors': []},
    }
    
    # Add results for each modification type
    for mod in modifications:
        results[mod] = {'translation_errors': [], 'rotation_errors': []}
    
    print(f"\nTesting {model_type} robustness...")
    
    with torch.no_grad():
        for imgs, poses_gt, _, _ in tqdm(data_loader, desc=f"  Testing {model_type}"):
            imgs = imgs.to(args.device)
            poses_gt = poses_gt.to(args.device)
            B = imgs.shape[0]
            
            # Get original predictions
            if model_type == 'probabilistic':
                mixture_dist = model(imgs)
                pi_probs = torch.softmax(mixture_dist.mixture_distribution.logits, dim=-1)
                component_means = mixture_dist.component_distribution.base_dist.loc
                mean_pose_6d = (pi_probs.unsqueeze(-1) * component_means).sum(dim=1)
                from utils.pose_utils import Lie
                lie = Lie()
                rot_axis_angle = mean_pose_6d[:, 3:6]
                rot_mat = lie.so3_to_SO3(rot_axis_angle)
                trans = mean_pose_6d[:, :3].unsqueeze(-1)
                poses_pred_original = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
            elif model_type == 'uaas':
                poses_pred_original, _ = model(imgs)
            elif model_type == 'semantic':
                output = model(imgs)
                if isinstance(output, tuple):
                    poses_pred_original = output[0] if output[0] is not None else output[1]
                else:
                    poses_pred_original = output
            else:  # baseline
                output = model(imgs, return_feature=False)
                if isinstance(output, tuple):
                    poses_pred_original = output[1] if len(output) >= 2 else output[0]
                else:
                    poses_pred_original = output
            
            # Convert to 3x4 format
            rot_mat_orig = poses_pred_original[:, :9].reshape(-1, 3, 3)
            trans_orig = poses_pred_original[:, 9:].unsqueeze(-1)
            u, s, v = torch.linalg.svd(rot_mat_orig)
            rot_mat_orig = u @ v.transpose(-2, -1)
            poses_pred_orig_3x4 = torch.cat([rot_mat_orig, trans_orig], dim=2)
            
            # Reshape ground truth
            if poses_gt.dim() == 2 and poses_gt.shape[1] == 12:
                gt_rot_mat = poses_gt[:, :9].reshape(-1, 3, 3)
                gt_trans = poses_gt[:, 9:].unsqueeze(-1)
                poses_gt_3x4 = torch.cat([gt_rot_mat, gt_trans], dim=2)
            else:
                poses_gt_3x4 = poses_gt
            
            # Calculate original errors
            for i in range(B):
                trans_err, rot_err = get_pose_error(
                    poses_pred_orig_3x4[i:i+1],
                    poses_gt_3x4[i:i+1]
                )
                results['original']['translation_errors'].append(trans_err.item())
                results['original']['rotation_errors'].append(rot_err.item())
            
            # Test each modification
            for mod_type in modifications:
                modified_imgs = imgs.clone()
                
                if mod_type == 'inpaint_center':
                    # Inpaint center region
                    H, W = imgs.shape[-2:]
                    mask = torch.zeros(H, W, device=imgs.device)
                    center_y, center_x = H // 2, W // 2
                    patch_size = min(H, W) // 4
                    mask[center_y-patch_size:center_y+patch_size, 
                         center_x-patch_size:center_x+patch_size] = 1.0
                    
                    for b in range(B):
                        img_b = modified_imgs[b]
                        modified_imgs[b] = modifier.inpaint_region(img_b, mask)
                
                elif mod_type == 'occlusion':
                    # Add random occlusions
                    for b in range(B):
                        modified_imgs[b] = modifier.add_occlusion(modified_imgs[b], num_patches=3, patch_size=50)
                
                elif mod_type == 'lighting':
                    # Modify lighting
                    for b in range(B):
                        modified_imgs[b] = modifier.modify_lighting(modified_imgs[b], brightness=0.2, contrast=1.3)
                
                elif mod_type == 'object_removal':
                    # Remove object (inpaint region)
                    H, W = imgs.shape[-2:]
                    for b in range(B):
                        # Random object position
                        center = (np.random.randint(H//4, 3*H//4), 
                                 np.random.randint(W//4, 3*W//4))
                        size = (H//6, W//6)
                        modified_img, mask = modifier.add_object_mask(modified_imgs[b], center, size)
                        modified_imgs[b] = modifier.inpaint_region(modified_img, mask)
                
                elif mod_type == 'blur_region':
                    # Blur specific region
                    H, W = imgs.shape[-2:]
                    mask = torch.zeros(H, W, device=imgs.device)
                    center_y, center_x = H // 2, W // 2
                    patch_size = min(H, W) // 5
                    mask[center_y-patch_size:center_y+patch_size, 
                         center_x-patch_size:center_x+patch_size] = 1.0
                    
                    for b in range(B):
                        modified_imgs[b] = modifier.blur_region(modified_imgs[b], mask)
                
                # Get predictions on modified images
                if model_type == 'probabilistic':
                    mixture_dist = model(modified_imgs)
                    pi_probs = torch.softmax(mixture_dist.mixture_distribution.logits, dim=-1)
                    component_means = mixture_dist.component_distribution.base_dist.loc
                    mean_pose_6d = (pi_probs.unsqueeze(-1) * component_means).sum(dim=1)
                    rot_axis_angle = mean_pose_6d[:, 3:6]
                    rot_mat = lie.so3_to_SO3(rot_axis_angle)
                    trans = mean_pose_6d[:, :3].unsqueeze(-1)
                    poses_pred_mod = torch.cat([rot_mat.reshape(-1, 9), trans.squeeze(-1)], dim=1)
                elif model_type == 'uaas':
                    poses_pred_mod, _ = model(modified_imgs)
                elif model_type == 'semantic':
                    output = model(modified_imgs)
                    if isinstance(output, tuple):
                        poses_pred_mod = output[0] if output[0] is not None else output[1]
                    else:
                        poses_pred_mod = output
                else:  # baseline
                    output = model(modified_imgs, return_feature=False)
                    if isinstance(output, tuple):
                        poses_pred_mod = output[1] if len(output) >= 2 else output[0]
                    else:
                        poses_pred_mod = output
                
                # Convert to 3x4 and orthonormalize
                rot_mat_mod = poses_pred_mod[:, :9].reshape(-1, 3, 3)
                trans_mod = poses_pred_mod[:, 9:].unsqueeze(-1)
                u, s, v = torch.linalg.svd(rot_mat_mod)
                rot_mat_mod = u @ v.transpose(-2, -1)
                poses_pred_mod_3x4 = torch.cat([rot_mat_mod, trans_mod], dim=2)
                
                # Calculate errors
                for i in range(B):
                    trans_err, rot_err = get_pose_error(
                        poses_pred_mod_3x4[i:i+1],
                        poses_gt_3x4[i:i+1]
                    )
                    results[mod_type]['translation_errors'].append(trans_err.item())
                    results[mod_type]['rotation_errors'].append(rot_err.item())
    
    # Calculate statistics
    stats = {}
    for key in ['original'] + modifications:
        stats[key] = {
            'translation': {
                'mean': float(np.mean(results[key]['translation_errors'])),
                'median': float(np.median(results[key]['translation_errors'])),
                'std': float(np.std(results[key]['translation_errors'])),
            },
            'rotation': {
                'mean': float(np.mean(results[key]['rotation_errors'])),
                'median': float(np.median(results[key]['rotation_errors'])),
                'std': float(np.std(results[key]['rotation_errors'])),
            }
        }
        
        # Calculate degradation vs original
        if key != 'original':
            trans_degradation = ((stats[key]['translation']['median'] - stats['original']['translation']['median']) 
                                / max(stats['original']['translation']['median'], 0.001)) * 100
            rot_degradation = ((stats[key]['rotation']['median'] - stats['original']['rotation']['median'])
                              / max(stats['original']['rotation']['median'], 0.001)) * 100 if stats['original']['rotation']['median'] > 0 else 0
            
            stats[key]['degradation'] = {
                'translation_pct': float(trans_degradation),
                'rotation_pct': float(rot_degradation)
            }
    
    return stats


def generate_robustness_charts(all_results: Dict, output_file: str):
    """Generate visualization charts for robustness testing."""
    modifications = ['inpaint_center', 'occlusion', 'lighting', 'object_removal', 'blur_region']
    models = list(all_results.keys())
    
    # Chart 1: Translation error degradation
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(modifications))
    width = 0.15
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, model_name in enumerate(models):
        degradations = [all_results[model_name][mod].get('degradation', {}).get('translation_pct', 0) 
                       for mod in modifications]
        ax.bar(x + i*width, degradations, width, label=model_name.upper(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Modification Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Translation Error Increase (%)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness to Dynamic Scene Changes - Translation Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in modifications])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_file}_translation_degradation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}_translation_degradation.png")
    plt.close()
    
    # Chart 2: Rotation error degradation
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, model_name in enumerate(models):
        degradations = [all_results[model_name][mod].get('degradation', {}).get('rotation_pct', 0) 
                       for mod in modifications]
        ax.bar(x + i*width, degradations, width, label=model_name.upper(), color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Modification Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rotation Error Increase (%)', fontsize=12, fontweight='bold')
    ax.set_title('Robustness to Dynamic Scene Changes - Rotation Error', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in modifications])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_file}_rotation_degradation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}_rotation_degradation.png")
    plt.close()
    
    # Chart 3: Comparison table
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Model'] + [m.replace('_', ' ').title() for m in modifications]
    
    for model_name in models:
        row = [model_name.upper()]
        for mod in modifications:
            deg = all_results[model_name][mod].get('degradation', {}).get('translation_pct', 0)
            row.append(f"{deg:+.1f}%")
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Translation Error Degradation Under Dynamic Scene Changes', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{output_file}_comparison_table.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file}_comparison_table.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test robustness to dynamic scene changes')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to GS model')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory with model checkpoints')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_samples', type=int, help='Limit test samples')
    parser.add_argument('--modifications', nargs='+', 
                       default=['inpaint_center', 'occlusion', 'lighting', 'object_removal', 'blur_region'],
                       help='Modifications to test')
    parser.add_argument('--models', nargs='+', 
                       default=['baseline', 'uaas', 'probabilistic', 'semantic'],
                       help='Models to test')
    parser.add_argument('--output', type=str, default='dynamic_scene_robustness_results.json')
    
    args = parser.parse_args()
    
    # Setup arguments
    config_parser_instance = config_parser()
    model_params = ModelParams(config_parser_instance)
    optimization = OptimizationParams(config_parser_instance)
    
    base_args = ['--run_name', 'dynamic_test', '--datadir', args.dataset]
    parsed_args = config_parser_instance.parse_args(base_args)
    model_params.extract(parsed_args)
    optimization.extract(parsed_args)
    
    parsed_args.device = args.device
    parsed_args.render_device = args.device
    parsed_args.model_path = args.model_path
    parsed_args.dataset_type = 'Colmap'
    parsed_args.train_skip = 1
    parsed_args.test_skip = 1
    parsed_args.batch_size = args.batch_size
    parsed_args.val_batch_size = args.batch_size
    parsed_args.compile = False
    parsed_args.brisque_threshold = None
    
    parsed_args = args_init.argument_init(parsed_args)
    fix_seed(7)
    
    # Load dataset
    print("Loading dataset...")
    camera_file = os.path.join(args.model_path, 'cameras.json')
    with open(camera_file) as f:
        camera = json.load(f)[0]
    
    cam_params = CamParams(camera, parsed_args.rap_resolution, args.device)
    rap_hw = (cam_params.h, cam_params.w)
    gs_hw = (cam_params.h, cam_params.w)
    
    test_dataset = ColmapDataset(
        data_path=args.dataset,
        train=False,
        hw=rap_hw,
        hw_gs=gs_hw,
        train_skip=parsed_args.train_skip,
        test_skip=parsed_args.test_skip
    )
    
    if args.max_samples and args.max_samples < len(test_dataset):
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, list(range(args.max_samples)))
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                              shuffle=False, num_workers=0)
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # Initialize modifier
    modifier = ImageModifier(device=args.device)
    
    # Test all models
    all_results = {}
    
    for model_type in args.models:
        print(f"\n{'='*80}")
        print(f"Testing {model_type.upper()} model")
        print(f"{'='*80}")
        
        # Load model
        if model_type == 'baseline':
            model = RAPNet(parsed_args)
        elif model_type == 'uaas':
            model = UAASRAPNet(parsed_args)
        elif model_type == 'probabilistic':
            model = ProbabilisticRAPNet(parsed_args)
        elif model_type == 'semantic':
            model = SemanticRAPNet(parsed_args, num_semantic_classes=19)
        else:
            continue
        
        model = model.to(args.device)
        
        # Load checkpoint if available
        if args.checkpoint_dir:
            checkpoint_files = list(Path(args.checkpoint_dir).glob(f"**/*{model_type}*.pth"))
            checkpoint_files.extend(list(Path(args.checkpoint_dir).glob("**/full_checkpoint.pth")))
            if checkpoint_files:
                checkpoint_path = str(checkpoint_files[0])
                checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"  ✓ Loaded checkpoint: {checkpoint_path}")
        
        model.eval()
        
        # Test robustness
        results = test_model_robustness(
            model, test_loader, modifier, parsed_args, model_type, args.modifications
        )
        
        all_results[model_type] = results
        
        # Print summary
        print(f"\n{model_type.upper()} Results:")
        print(f"  Original - Translation: {results['original']['translation']['median']:.4f}m, "
              f"Rotation: {results['original']['rotation']['median']:.4f}°")
        
        for mod in args.modifications:
            deg = results[mod].get('degradation', {})
            print(f"  {mod}: Translation +{deg.get('translation_pct', 0):.1f}%, "
                  f"Rotation +{deg.get('rotation_pct', 0):.1f}%")
    
    # Save results
    output_data = {
        'dataset': args.dataset,
        'test_samples': len(test_dataset),
        'modifications': args.modifications,
        'models': args.models,
        'results': all_results,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    
    # Generate charts
    print("\nGenerating visualizations...")
    generate_robustness_charts(all_results, args.output.replace('.json', ''))
    
    print("\n✅ Dynamic scene robustness testing complete!")


if __name__ == "__main__":
    main()

