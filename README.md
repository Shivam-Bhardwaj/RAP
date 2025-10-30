<p align="center">
  <h1 align="center">RAP-ID<br>Robust Absolute Pose Regression<br>with Improvements & Extensions</h1>
  <h3 align="center">Enhanced Fork of RAP</h3>
  <p align="center">
    <a href="https://github.com/Shivam-Bhardwaj">Shivam Bhardwaj</a>
  </p>
  <p align="center">
    Forked from <a href="https://github.com/ai4ce/RAP">ai4ce/RAP</a> | 
    <a href="https://ai4ce.github.io/RAP/static/RAP_Paper.pdf">Original Paper</a>
  </p>
</p>

## Overview

**RAP-ID** (RAP with Improvements & Extensions) is an enhanced fork of the [RAP](https://github.com/ai4ce/RAP) system for 6-DoF camera localization. This fork extends the original RAP architecture with three major improvements:

1. **Uncertainty-Aware Adversarial Synthesis (UAAS)** - Predicts pose uncertainty and uses it to guide training data synthesis
2. **Multi-Hypothesis Probabilistic APR** - Handles pose ambiguity through probabilistic multi-hypothesis predictions
3. **Semantic-Adversarial Scene Synthesis** - Generates challenging training samples through semantic-aware scene manipulation

These extensions address key limitations in visual localization: uncertainty quantification, handling ambiguous scenes, and robustness to semantic scene variations.

## Key Features

### Uncertainty-Aware Adversarial Synthesis (UAAS)

The UAAS module extends RAPNet to output both pose predictions and explicit uncertainty estimates (epistemic and aleatoric). Key components:

- **Uncertainty Prediction**: Model outputs log-variance estimates for pose uncertainty
- **Targeted Data Synthesis**: Uses uncertainty estimates to guide 3DGS rendering toward high-uncertainty regions
- **Uncertainty-Weighted Adversarial Loss**: Discriminator prioritizes domain adaptation in uncertain regions
- **Visualization Tools**: Utilities for visualizing uncertainty across training sets and synthetic samples

**Usage:**
```bash
python train.py --config configs/7scenes.txt --trainer_type uaas --run_name experiment_uaas
```

### Multi-Hypothesis Probabilistic APR

Replaces single-point pose predictions with probabilistic outputs capable of expressing multiple plausible pose hypotheses:

- **Mixture Density Network (MDN)**: Models pose distribution as a mixture of Gaussians
- **Hypothesis Ranking**: Validates hypotheses using 3DGS rendering and image comparison
- **Ambiguity Resolution**: Downstream selection and refinement modules resolve pose ambiguity

**Usage:**
```bash
python train.py --config configs/7scenes.txt --trainer_type probabilistic --run_name experiment_prob
```

### Semantic-Adversarial Scene Synthesis

Incorporates semantic segmentation into the training pipeline for targeted scene manipulation:

- **Semantic Integration**: Scene annotated by semantic classes (sky, building, road, etc.)
- **Targeted Appearance Variations**: 3DGS synthesizer produces variations targeting specific semantic regions
- **Adversarial Hard Negative Mining**: Creates synthetic scenes designed to maximize prediction errors
- **Curriculum Learning**: Gradually increases synthetic scene difficulty based on model performance

**Usage:**
```bash
python train.py --config configs/7scenes.txt --trainer_type semantic --run_name experiment_semantic --num_semantic_classes 19
```

## Architecture

### Module Structure

```
RAP-ID/
├── uaas/                    # Uncertainty-Aware Adversarial Synthesis
│   ├── uaas_rap_net.py      # Extended RAPNet with uncertainty head
│   ├── loss.py              # Uncertainty-weighted adversarial loss
│   ├── sampler.py           # Uncertainty-guided data sampling
│   └── trainer.py           # UAAS training loop
├── probabilistic/           # Multi-Hypothesis Probabilistic APR
│   ├── probabilistic_rap_net.py  # MDN-based pose prediction
│   ├── loss.py              # Mixture NLL loss
│   ├── hypothesis_validator.py   # Hypothesis ranking via rendering
│   ├── selection.py         # Best hypothesis selection
│   └── trainer.py           # Probabilistic training loop
├── semantic/               # Semantic-Adversarial Scene Synthesis
│   ├── semantic_rap_net.py  # Semantic-aware RAPNet
│   ├── semantic_synthesizer.py  # Semantic scene manipulation
│   ├── hard_negative_miner.py    # Adversarial hard negative mining
│   ├── curriculum.py        # Curriculum learning scheduler
│   └── trainer.py           # Semantic training loop
├── common/                  # Shared utilities
│   └── uncertainty.py      # Uncertainty calculation and visualization
└── train.py                 # Unified training script
```

## Quick Start

### Installation

1. Clone this repository:

```sh
git clone https://github.com/Shivam-Bhardwaj/RAP.git
cd RAP
```

2. Install dependencies (Python 3.11+, PyTorch 2.0+, CUDA):

```sh
pip install -r requirements.txt
```

See the [original RAP setup instructions](#setup-instructions-from-original-rap) for detailed environment requirements.

### Training

Use the unified training script with different trainer types:

```bash
# Train UAAS model
python train.py --config configs/7scenes.txt --trainer_type uaas --run_name my_uaas_exp

# Train Probabilistic model
python train.py --config configs/7scenes.txt --trainer_type probabilistic --run_name my_prob_exp

# Train Semantic model
python train.py --config configs/7scenes.txt --trainer_type semantic --run_name my_semantic_exp --num_semantic_classes 19
```

### Benchmarking

Evaluate trained models:

```bash
# Benchmark rendering speed and pose accuracy
python benchmark_speed.py --model_path /path/to/model --benchmark_pose --model_type uaas
```

Supported model types: `uaas`, `probabilistic`, `semantic`, `baseline`

## Evaluation

Evaluation results will be added after running experiments. Metrics to include:

- **Generalization**: Pose accuracy on held-out test sets and novel datasets
- **Calibrated Uncertainty**: Correlation between predicted uncertainty and prediction error (UAAS)
- **Ambiguity Handling**: Multi-modal pose predictions in ambiguous scenes (Probabilistic)
- **Error Reduction**: Performance improvement over baseline RAP across benchmarks

## Citation

If you use RAP-ID in your research, please cite both this work and the original RAP paper:

```bibtex
@misc{bhardwaj2025rapid,
  title={RAP-ID: Robust Absolute Pose Regression with Improvements \& Extensions},
  author={Shivam Bhardwaj},
  year={2025},
  howpublished={\url{https://github.com/Shivam-Bhardwaj/RAP}}
}

@inproceedings{Li2025unleashing,
  title={Unleashing the Power of Data Synthesis},
  author={Sihang Li and Siqi Tan and Bowen Chang and Jing Zhang and Chen Feng and Yiming Li},
  year={2025},
  booktitle={International Conference on Computer Vision (ICCV)}
}
```

## Acknowledgments

This work is built upon the RAP system by Sihang Li, Siqi Tan, Bowen Chang, Jing Zhang, Chen Feng, and Yiming Li. The original RAP paper was accepted to ICCV 2025.

The original RAP repository is built on [Gaussian-Wild](https://github.com/EastbeanZhang/Gaussian-Wild), [Deblur-GS](https://github.com/Chaphlagical/Deblur-GS), and [DFNet](https://github.com/ActiveVisionLab/DFNet).

## Links

- **Original RAP Repository:** https://github.com/ai4ce/RAP
- **Original Paper:** https://ai4ce.github.io/RAP/static/RAP_Paper.pdf
- **Project Page:** https://ai4ce.github.io/RAP/

---

## Setup Instructions (from Original RAP)

The following section contains setup instructions from the original RAP repository for reference.

### Requirements

The current implementation stores the entire training set in memory, requiring approximately 16GB of RAM (training 3DGS on the office scene with depth regularization may require more than 64GB).

Running RAP-ID on a single device requires a CUDA-compatible GPU, with 24GB VRAM recommended.

We also support training RAPNet and rendering 3DGS in parallel on different devices. If you choose to do so, ensure that `args.device != args.render_device`. This is the default behavior when using the BRISQUE score to filter out low-quality rendered images (i.e., `args.brisque_threshold != 0`). The current implementation calculates the BRISQUE score on the CPU due to its better SVM model, which, despite optimizations achieving ~70 FPS, remains slower than a GPU version.

In theory, RAPNet can be trained on devices other than NVIDIA GPUs, but this has not been tested. Still, rendering 3DGS requires a CUDA-compatible GPU.

Post-refinement requires a CUDA-compatible GPU with at least 6GB of VRAM.

1. Clone the repository in recursive mode as it contains submodules:

   ```sh
   git clone https://github.com/Shivam-Bhardwaj/RAP.git --recursive
   ```

2. Make sure you have an environment with Python 3.11+ and CUDA Toolkit `nvcc` compiler accessible from the command line.

   > If you are on Windows, you need to install Visual Studio with MSVC C++ SDK first, and then install CUDA Toolkit.

3. Make sure PyTorch 2.0 or later is installed in your environment. We recommend PyTorch 2.6+. The CUDA version of PyTorch should match the version used by `nvcc` (check with `nvcc -v`), and should not exceed the version supported by your GPU driver (check with `nvidia-smi`).

   > We use `torch.compile` for acceleration and reducing memory and its `triton` backend only supports Linux. When `torch.compile` is enabled for a module, it will seem to stuck for a while during its first and last forward pass in the first epoch of training and validating depending on how high your CPU's single-core performance is. Windows and older PyTorch versions might work if you set `args.compile_model = False` and make sure `args.compile = False` when you run the code, but it might be buggy, slower, and consume more memory, so it is not recommended.

4. Install packages. This might take a while as it involves compiling two CUDA extensions.

   ```sh
   pip install -r requirements.txt
   ```

   > The original `diff-gaussian-rasterizer` is needed if you want to use inverted depth maps for supervision. Use the following command to build and install:
   >
   > ```sh
   > pip install "git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@dr_aa"
   > ```
   >
   > `pytorch3d` is needed if you want to use Bezier interpolation when training deblurring Gaussians. Use the following command to build and install:
   >
   > ```shell
   > pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   > ```

## Training, Rendering, and Evaluating 3DGS

```sh
python gs.py -s /path/to/colmap/data -m /path/to/output
```

Useful arguments include: `white_background`, `eval`, `train_fraction`, `antialiasing`, `use_masks`, `iterations`, `position_lr_final`, `position_lr_max_steps`, `percent_dense`, `densify_until_iter`, `densify_grad_threshold`, `use_depth_loss`, `depth_is_inverted`, `deblur`, `prune_more`.

If you want to use depth supervision for datasets that do not come with metric depths, please following the instructions provided [here](https://github.com/graphdeco-inria/gaussian-splatting#depth-regularization). Training will be slower, and we do not observe much benefits in our subsequent APR.

> Note that for 3DGS-related arguments, only `-s, --source_path` and `-m, --model_path` will be taken from the command line when running `render.py`, `rap.py`, and `refine.py`. Other arguments will be loaded from `cfg_args` in the 3DGS model directory. If you want to change some arguments, you may just edit the `cfg_args` file, or assign values in the code.

## Running Original RAP

```sh
python rap.py -c configs/actual_config_file.txt -m /path/to/3dgs
```

See `arguments/options.py` for arguments usage.

Due to uncontrollable randomness, the computation order of floating-point numbers may vary across different devices, batch sizes, and whether the model is compiled, potentially leading to results that differ from those reported in the paper.

## RAP<sub>ref</sub> Post Refinement

```sh
python refine.py -c configs/actual_config_file.txt -m /path/to/3dgs
```

Post-refinement is more CPU-intensive than other tasks.
