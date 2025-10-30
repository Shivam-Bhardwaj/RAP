# RAP Benchmarking Guide

## Overview

This document provides a comprehensive guide for benchmarking the RAP (Visual Localization) system across multiple datasets and scenarios. The benchmarking evaluates both the initial RAPNet predictions and post-refined results (RAP_ref).

## Benchmark Metrics

### Primary Metrics

1. **Translation Error (MedTrans, AvgTrans)**
   - Units: meters (m)
   - Computation: L2 norm of translation difference between predicted and ground truth poses
   - Reported as: Median, Mean, Max, Min

2. **Rotation Error (MedRot, AvgRot)**
   - Units: degrees (°)
   - Computation: Angular difference in quaternion space (2 * arccos(|q_pred · q_gt|))
   - Reported as: Median, Mean, Max, Min

3. **Success Rates**
   - **% Suc 5**: Percentage of images with translation < 5cm AND rotation < 5°
   - **% Suc 2**: Percentage of images with translation < 2cm AND rotation < 2°

### Secondary Metrics (RAP_ref only)

4. **Rendering Quality**
   - **PSNR**: Peak Signal-to-Noise Ratio (dB) for rendered vs. ground truth images
   - **Average FPS**: Frame rate during refinement pipeline

5. **Visualization Outputs**
   - Trajectory visualization (`traj.npz`)
   - Pose comparison plots (`vis.jpg`)
   - Rotation error bars (`bar/` directory)
   - Before/after comparison images (`compare/` directory)
   - Matching visualization (`match/` directory)

## Benchmark Datasets

### 1. Cambridge Landmarks (Outdoor)
- **Scenes**: `shop`, `hospital`, `college`, `church`, `court`
- **Dataset Type**: `Cambridge` (except `court` uses `Colmap`)
- **Config Location**: `configs/Cambridge/`
- **Data Path**: `data/Cambridge/{scene}/` or `data/Cambridge/{scene}/colmap/undistorted` for court
- **Expected Performance**: ~20cm/0.5° (outdoor scenarios)

### 2. 7Scenes (Indoor RGB-D)
- **Scenes**: `chess`, `fire`, `heads`, `office`, `pumpkin`, `kitchen`, `stairs`
- **Dataset Type**: `7Scenes` (original) or `Colmap` (SfM version)
- **Config Locations**: 
  - `configs/7Scenes/` (original DSLAM dataset)
  - `configs/7Scenes_sfm/` (COLMAP SfM version)
- **Data Path**: `data/7Scenes/{scene}/` or `data/7Scenes_sfm/{scene}/`
- **Expected Performance**: <1cm/0.3° (indoor scenarios)

### 3. MARS (Driving Scenarios)
- **Scenes**: `11`, `15`, `37`, `41`
- **Dataset Type**: `Colmap`
- **Config Location**: `configs/MARS/`
- **Data Path**: `data/MARS/{scene}/`
- **Scale Multipliers**:
  - Scene 11: 7.2162
  - Scene 15: 6.6005
  - Scene 37: 7.68605
  - Scene 41: 7.67
- **Special Config**: Scenes 37 and 41 use `--rap_resolution 4`
- **Expected Performance**: ~10cm/0.2° (driving scenarios)

### 4. Aachen (Outdoor Urban)
- **Scene**: `aachen_sub`
- **Dataset Type**: `Colmap`
- **Config Location**: `configs/aachen.txt`
- **Data Path**: `data/aachen_sub/`
- **Expected Performance**: ~20cm/0.5° (outdoor scenarios)

### 5. St. George's Basilica
- **Config Location**: `configs/new_church.txt`
- **Data Path**: `data/new_church/`
- **Note**: Trained with testing set

## Benchmarking Workflow

### Step 1: Prepare 3DGS Models

First, train 3DGS models for each scene:

```bash
python gs.py -s /path/to/colmap/data -m /path/to/output
```

**Key Arguments**:
- `--source_path`: COLMAP dataset directory
- `--model_path`: Output directory for 3DGS model
- `--eval`: Enable evaluation during training
- `--use_depth_loss`: Enable depth regularization (optional, slower)
- `--deblur`: Enable deblurring training (optional)

**Output**: 3DGS checkpoint in `{model_path}/` with `cameras.json` and `cfg_args`

### Step 2: Train RAPNet

Train RAPNet for each scene:

```bash
python rap.py -c configs/{dataset}/{scene}.txt -m /path/to/3dgs/model
```

**Config File Structure**:
```ini
run_name=scene_name
logbase=ckpts/DatasetName
datadir=data/DatasetName/scene
dataset_type=Colmap|7Scenes|Cambridge
train_skip=5
test_skip=1
rap_resolution=2
epochs=2000
batch_size=8
val_batch_size=8
rvs_refresh_rate=20
rvs_trans=0.2
rvs_rotation=10
d_max=0.2
brisque_threshold=0
```

**Output**: Trained model checkpoint in `{logbase}/{run_name}/full_checkpoint.pth` or best model

### Step 3: Evaluate RAPNet

Evaluate the trained model:

```bash
python eval.py -c configs/{dataset}/{scene}.txt -m /path/to/3dgs/model \
    -p /path/to/checkpoint.pth
```

**Output Metrics** (printed to console and logged to wandb):
- `MedTrans`, `MedRot`: Median translation/rotation errors
- `AvgTrans`, `AvgRot`: Mean errors
- `MaxTrans`, `MaxRot`: Maximum errors
- `MinTrans`, `MinRot`: Minimum errors
- `% Suc 5`, `% Suc 2`: Success rates
- `ValLoss`: Validation loss

**Visualization Outputs**:
- `{logbase}/{run_name}/vis.jpg`: 3D trajectory visualization
- `{logbase}/{run_name}/traj.npz`: Trajectory data with error coloring
- `{logbase}/{run_name}/bar/`: Rotation error bar visualization frames

### Step 4: Post-Refinement (RAP_ref)

Refine predictions using feature matching:

```bash
python refine.py -c configs/{dataset}/{scene}.txt -m /path/to/3dgs/model \
    -p /path/to/checkpoint.pth
```

**Additional Arguments**:
- `--confidence_threshold`: Feature matching confidence threshold
- `--pnp_mode`: PnP solver mode
- `--pnp_max_points`: Maximum points for PnP
- `--cpu_affinity_ids`: CPU affinity for multi-threading (space-separated list)
- `--poses_txt`: Use pre-computed poses from file (skip RAPNet inference)

**Output Metrics** (in addition to Step 3):
- `FPS`: Average frames per second
- `AvgPSNR`, `MedPSNR`: PSNR statistics for rendered images

**Visualization Outputs**:
- `vis/{run_name}/render/`: Rendered images (before/after refinement, ground truth)
- `vis/{run_name}/depth/`: Depth maps
- `vis/{run_name}/match/`: Feature matching visualization
- `vis/{run_name}/compare/`: Before/after comparison overlays

## Batch Benchmarking

### Multi-GPU Evaluation

Use `eval_all.py` to run evaluation across multiple GPUs:

```bash
python eval_all.py
```

**Configuration**:
- Edit `GPUS` list to specify GPU IDs
- Edit `TASKS` list to add/remove benchmark tasks
- Tasks format: `(name, data_dir, dataset_type, ckpt_path, gs_dir, extra_args)`
- Logs saved to `logs_out/{name}.log`

**Example Task**:
```python
TASKS.append((
    "Cam_shop_ref",              # Task name
    "data/Cambridge/shop",       # Data directory
    "Cambridge",                 # Dataset type
    "logs/Cambridge/shop.pth",   # Checkpoint path
    "output/Cambridge/shop",     # 3DGS model directory
    ""                           # Extra arguments
))
```

### Multi-GPU Training

Use `rap_all.py` for batch training:

```bash
python rap_all.py
```

**Configuration**:
- Edit `GPUS` list
- Edit `TASKS` list with format: `(name, config_path, gs_dir, extra_args)`
- Logs saved to `logs_out/{name}.log`

### Multi-GPU Refinement

Use `refine_all.py` for batch refinement:

```bash
python refine_all.py
```

**Configuration**:
- Edit `GPU_CPU_MAP` to map GPU IDs to CPU affinity lists
- Edit `TASKS` list (same format as `eval_all.py`)
- Uses CPU affinity for better multi-threading performance

## Evaluation Scripts Reference

### `eval.py`
- **Purpose**: Evaluate RAPNet on a single scene
- **Metrics**: Translation/rotation errors, success rates
- **Output**: Console output, wandb logs, visualization files

### `eval_all.py`
- **Purpose**: Batch evaluation across multiple scenes with multi-GPU support
- **Features**: Multiprocessing with GPU assignment, queue-based task distribution
- **Output**: Per-task log files in `logs_out/`

### `refine.py`
- **Purpose**: Post-refinement with feature matching and PnP
- **Features**: Iterative refinement, visualization generation
- **Output**: Refined poses, visualization outputs, performance metrics

### `refine_all.py`
- **Purpose**: Batch refinement with multi-GPU and CPU affinity support
- **Features**: CPU affinity mapping for optimization
- **Output**: Per-task log files in `logs_out/`

## Metrics Computation Details

### Pose Error Calculation (`utils/eval_utils.py`)

```python
def get_pose_error(gt_pose, pred_pose):
    # Convert rotation matrices to quaternions
    gt_q = rotation_matrix_to_quaternion(gt_pose[..., :3, :3])
    gt_t = gt_pose[..., :3, 3]
    pred_q = rotation_matrix_to_quaternion(pred_pose[..., :3, :3])
    pred_t = pred_pose[..., :3, 3]
    
    # Rotation error: 2 * arccos(|q_gt · q_pred|)
    theta = gt_q.mul_(pred_q).sum(dim=-1).abs_().clamp_(-1., 1.).acos_().mul_(2 * 180 / math.pi)
    
    # Translation error: L2 norm
    error_x = torch.linalg.vector_norm(gt_t - pred_t, ord=2, dim=-1)
    
    return error_x, theta
```

### Success Rate Calculation

```python
# 5cm/5° threshold
success_condition_5 = (errors_trans < 0.05) & (errors_rot < 5)
success_rate_5 = np.sum(success_condition_5) / errors_trans.shape[0]

# 2cm/2° threshold
success_condition_2 = (errors_trans < 0.02) & (errors_rot < 2)
success_rate_2 = np.sum(success_condition_2) / errors_trans.shape[0]
```

## Results Format

### Console Output Example

```
MedTrans: 0.0123 m, MedRot: 0.456 degrees
AvgTrans: 0.0156 m, AvgRot: 0.523 degrees
MaxTrans: 0.0892 m, MaxRot: 2.345 degrees
MinTrans: 0.0012 m, MinRot: 0.012 degrees
Success rate (5cm/5degree): 87.50%
Success rate (2cm/2degree): 65.23%
```

### WandB Logging

Metrics are automatically logged to WandB with project name "RAP_A6000" (or configurable):
- `Epoch`, `TrainLoss`, `ValLoss`
- `MedTrans`, `MedRot`, `AvgTrans`, `AvgRot`
- `MaxTrans`, `MaxRot`, `MinTrans`, `MinRot`
- `# Suc 5`, `# Suc 2`, `% Suc 5`, `% Suc 2`
- `FPS` (refinement only)
- `AvgPSNR`, `MedPSNR`, `MaxPSNR`, `MinPSNR` (refinement only)

### File Outputs

**Evaluation**:
- `{logbase}/{run_name}/vis.jpg`: 3D trajectory plot
- `{logbase}/{run_name}/traj.npz`: Trajectory data (points, colors)
- `{logbase}/{run_name}/bar/frame_*.jpg`: Rotation error bar frames
- `{logbase}/{run_name}/dir.txt`: Direction vectors

**Refinement**:
- `vis/{run_name}/render/`: Rendered images
- `vis/{run_name}/depth/`: Depth maps
- `vis/{run_name}/match/`: Matching visualizations
- `vis/{run_name}/compare/`: Before/after comparisons

## Benchmarking Best Practices

### 0. Pre-Benchmark Checklist

Before running benchmarks, use the automated tool to verify setup:

```bash
python3 run_benchmark.py
```

This will check:
- ✅ Python version (3.11+)
- ✅ PyTorch installation
- ✅ Required dependencies
- ✅ Dataset availability
- ✅ Checkpoint existence
- ✅ 3DGS model availability

### 1. Reproducibility
- Set `--seed` argument consistently across runs
- Note: Floating-point computation order may vary across devices/batch sizes

### 2. Resource Management
- Use `eval_all.py` / `refine_all.py` for multi-GPU setups
- Monitor GPU memory (24GB VRAM recommended)
- Use CPU affinity for refinement (`--cpu_affinity_ids`)

### 3. Configuration
- Dataset-specific configs in `configs/` directory
- Adjust `rap_resolution` based on scene size (lower for larger scenes)
- Tune `rvs_trans` and `rvs_rotation` based on scene scale

### 4. Evaluation
- Always evaluate on test set (`test_skip=1`)
- Use consistent `val_batch_size` across benchmarks
- Enable `--vis_featuremap` for debugging if needed

### 5. Post-Processing
- Refinement improves results but is slower
- Use `--poses_txt` to skip RAPNet inference if only refining
- Monitor success rates to assess improvement

## Common Benchmark Configurations

### Indoor (7Scenes)
```ini
rap_resolution=2
rvs_trans=0.2
rvs_rotation=10
d_max=0.2
train_skip=5
```

### Outdoor (Cambridge)
```ini
rap_resolution=2
rvs_trans=1.5
rvs_rotation=4
d_max=1.0
train_skip=2
```

### Large-Scale (MARS)
```ini
rap_resolution=4  # For scenes 37, 41
rvs_trans=5
rvs_rotation=1.2
d_max=1.0
```

## Troubleshooting

### Memory Issues
- Reduce `val_batch_size`
- Reduce `rap_resolution`
- Disable `--compile_model` if using older PyTorch

### Poor Results
- Check 3DGS model quality (render some test images)
- Verify COLMAP reconstruction quality
- Ensure scale consistency (especially for MARS)
- Check config file parameters match dataset characteristics

### Refinement Failures
- Increase `--confidence_threshold` if too many false matches
- Adjust `--pnp_max_points` based on scene
- Check if 3DGS renders are good quality (PSNR)

## Performance Targets

Based on paper results:

- **Indoor (7Scenes)**: <1cm / <0.3°
- **Outdoor (Cambridge)**: <20cm / <0.5°
- **Driving (MARS)**: <10cm / <0.2°

Success rates should be:
- **% Suc 5**: >80% for indoor, >60% for outdoor
- **% Suc 2**: >60% for indoor, >40% for outdoor

## Project Structure

### Core Components

#### 1. **3D Gaussian Splatting (3DGS)**
- **Location**: `models/gs/`, `gs.py`, `render.py`
- **Purpose**: Train 3DGS models to synthesize novel views for training data augmentation
- **Main Script**: `gs.py` - Trains 3DGS models from COLMAP data
- **Key Features**:
  - Supports depth regularization (`use_depth_loss`)
  - Supports deblurring (`deblur`)
  - Configurable via `cfg_args` in model directory

#### 2. **RAPNet (Absolute Pose Regression Network)**
- **Location**: `models/apr/rapnet.py`
- **Architecture**:
  - Backbone: EfficientNet-B0/B3 (via `backbone.py`)
  - Feature adaptation layers (`AdaptLayers2`)
  - Transformer encoder (`transformer_encoder.py`)
  - Pose regressor (MLP)
  - Two-branch structure with discriminator (`discriminator.py`)

#### 3. **Dataset Loaders**
- **Location**: `dataset_loaders/`
- **Supported Datasets**:
  - `ColmapDataset` - Generic COLMAP format
  - `SevenScenes` - Microsoft 7-Scenes dataset
  - `Cambridge` - Cambridge Landmarks dataset
- **Key Files**:
  - `colmap_dataset.py` - Main COLMAP loader
  - `seven_scenes.py` - 7-Scenes specific loader
  - `cambridge_scenes.py` - Cambridge specific loader

#### 4. **RVS (Random View Synthesis)**
- **Location**: `utils/nvs_utils.py`
- **Purpose**: Synthesize novel views from 3DGS during training
- **Features**:
  - BRISQUE score filtering for quality control
  - Appearance augmentation
  - Configurable sampling strategies (uniform, sphere, XZ-plane only)

#### 5. **Post-Refinement (RAP_ref)**
- **Location**: `refine.py`, `refine_*.py`
- **Purpose**: Refine initial pose estimates using feature matching
- **Components**:
  - Matcher (`matcher.py`) - Feature matching
  - PnP solver (via `dust3r_visloc`)

### Main Entry Points

1. **Train 3DGS**: `python gs.py -s /path/to/colmap/data -m /path/to/output`
2. **Train RAPNet**: `python rap.py -c configs/config_file.txt -m /path/to/3dgs`
3. **Evaluate**: `python eval.py -c configs/config_file.txt -m /path/to/3dgs`
4. **Post-Refinement**: `python refine.py -c configs/config_file.txt -m /path/to/3dgs`

### Configuration System

- **Config Files**: Located in `configs/` directory
- **Format**: Text files with key-value pairs
- **Parser**: `arguments/options.py` - Uses `configargparse`
- **Model Params**: `arguments/ModelParams` - 3DGS specific parameters
- **Optimization Params**: `arguments/OptimizationParams` - Training parameters

### Key Configuration Options

#### Training (RAPNet)
- `--run_name`: Experiment name
- `--datadir`: Dataset directory
- `--model_path`: Path to trained 3DGS model
- `--pretrained_model_path`: Path to pretrained RAPNet checkpoint
- `--device`: Training device (default: 'cuda')
- `--render_device`: Device for 3DGS rendering (can be different)
- `--epochs`: Max training epochs (default: 2000)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--batch_size`: Training batch size (default: 8)
- `--amp`: Enable automatic mixed precision (default: True)
- `--compile_model`: Compile model with torch.compile (default: True)

#### RVS (Random View Synthesis)
- `--max_attempts`: Max attempts for RVS (default: 100)
- `--brisque_threshold`: BRISQUE score threshold (default: 50)
- `--rvs_refresh_rate`: Epochs between RVS refreshes (default: 2)
- `--rvs_trans`: Translation jitter range (default: 5)
- `--rvs_rotation`: Rotation jitter range (default: 1.2)
- `--d_max`: Maximum RVS bound (default: 1)

#### Loss Configuration
- `--loss_weights`: Weights for combined loss [pose, feature, adversarial, ...] (default: [1, 1, 1, 0.7])
- `--feature_loss`: Feature loss type ('triplet', 'vicreg', 'ntxent', 'mse')
- `--loss_learnable`: Enable learnable pose loss (default: True)
- `--loss_norm`: Pose loss norm order (default: 2)
- `--s_x`, `--s_q`: Pose loss scaling parameters

#### Model Architecture
- `--reduction`: Feature extraction layers (default: ["reduction_4", "reduction_3"])
- `--hidden_dim`: Hidden dimension (default: 256)
- `--num_heads`: Attention heads (default: 4)
- `--num_encoder_layers`: Transformer encoder layers (default: 6)
- `--dropout`: Dropout rate (default: 0.1)

### Dataset Formats

#### COLMAP Format
Expected structure:
```
dataset/
├── images/
│   └── *.jpg
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── list_test.txt  # Optional: list of test images
```

#### Setup Scripts
- `utils/run_colmap.sh` (Linux) - Convert images to COLMAP format
- `utils/run_colmap.ps1` (Windows) - Windows version
- `utils/setup_cambridge.py` - Setup Cambridge Landmarks

### Dependencies

#### Main Requirements
- PyTorch 2.0+ (2.6+ recommended)
- CUDA Toolkit (nvcc compiler)
- Python 3.11+
- Core packages: `configargparse`, `efficientnet_pytorch`, `kornia`, `lpips`, `wandb`, `kapture`, `roma`

#### Submodules
- `submodules/gsplat` - Gaussian Splatting implementation
- `submodules/fused-ssim` - SSIM metric
- `dust3r` - Visual localization components (copied to root)
- `mast3r` - Visual localization models (copied to root)

### Training Workflow

1. **Prepare Data**: Convert images to COLMAP format
2. **Train 3DGS**: `python gs.py -s <data_path> -m <output_path>`
3. **Train RAPNet**: `python rap.py -c configs/config.txt -m <3dgs_path>`
4. **Evaluate**: `python eval.py -c configs/config.txt -m <3dgs_path>`
5. **Optional Refinement**: `python refine.py -c configs/config.txt -m <3dgs_path>`

### Key Utilities

#### Evaluation (`utils/eval_utils.py`)
- `eval_model()`: Evaluates RAPNet on validation/test set
- `get_pose_error()`: Computes pose errors (translation/rotation)
- `vis_pose()`: Visualizes pose estimates

#### Pose Utilities (`utils/pose_utils.py`)
- `CameraPoseLoss`: Learnable pose loss with configurable norms
- `compute_rotation_matrix_from_ortho6d`: Rotation representation

#### Camera Utilities (`utils/cameras.py`)
- `CamParams`: Camera parameter handling
- `Camera`: Camera pose representation

#### NVS Utilities (`utils/nvs_utils.py`)
- `GaussianRendererWithAttempts`: Renders views from 3DGS
- `GaussianRendererWithBrisqueAttempts`: With BRISQUE filtering

### Model Checkpoints

- **3DGS Checkpoints**: Stored in model directory with `cfg_args` file
- **RAPNet Checkpoints**: Stored in `{logbase}/{run_name}/` directory
  - `full_checkpoint.pth`: Full training state
  - Best model saved via `EarlyStopper`

### Important Notes

1. **Memory Requirements**: 
   - ~16GB RAM for training (64GB+ for depth regularization)
   - 24GB VRAM recommended for single-device training

2. **Device Configuration**: Can train RAPNet and render 3DGS on different devices (`args.device != args.render_device`)

3. **Randomness**: Floating-point computation order may vary across devices/batch sizes, leading to slight result differences

4. **Compilation**: `torch.compile` only works on Linux. Windows requires `--compile_model False`

5. **Scale**: COLMAP outputs are not in metric scale. Use scale multipliers (see README for MARS dataset) or transformation matrices

### Supported Datasets

1. **Cambridge Landmarks**: Indoor/outdoor scenes
2. **7Scenes**: Indoor RGB-D scenes
3. **MARS**: Driving scenarios
4. **Aachen**: Outdoor urban scenes
5. **Custom Datasets**: Via COLMAP format

### Code Organization

- `arguments/`: Command-line argument parsing
- `models/apr/`: RAPNet and related models
- `models/gs/`: 3DGS implementation
- `dataset_loaders/`: Dataset loading logic
- `utils/`: Utility functions (evaluation, pose, cameras, etc.)
- `configs/`: Configuration files for different datasets/scenes

### Research Background

- Built on: Gaussian-Wild, Deblur-GS, DFNet
- Key innovation: Joint learning with adversarial training + 3DGS data synthesis
- Published: ICCV 2025

### Common Tasks

1. **Adding a new dataset**: Create dataset loader in `dataset_loaders/` following existing patterns
2. **Modifying architecture**: Edit `models/apr/rapnet.py` and related components
3. **Adjusting loss**: Modify loss weights in config or edit `utils/pose_utils.py`
4. **Adding visualization**: Use `--vis_rvs`, `--vis_featuremap` flags or add custom visualization

