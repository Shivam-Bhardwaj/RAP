# Practical Demos for RAP-ID

## Overview

RAP-ID includes three practical demo scripts that showcase the improvements over baseline RAP:

1. **Uncertainty Visualization Demo** (`demo_uncertainty.py`)
2. **Multi-Hypothesis Demo** (`demo_hypothesis.py`)
3. **Comparison Demo** (`demo_comparison.py`)

## Demo 1: Uncertainty Visualization

**Purpose:** Demonstrate how UAAS provides uncertainty estimates for pose predictions.

**Features:**
- Visualizes uncertainty maps overlaid on images
- Shows which regions have high/low uncertainty
- Identifies unreliable predictions

**Usage:**
```bash
python demo_uncertainty.py \
    --config configs/7scenes.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs \
    --pretrained_model_path /path/to/uaas_checkpoint.pth \
    --output_dir demo_output
```

**Output:**
- Uncertainty heatmaps for each image
- Statistics (mean, std, min, max uncertainty)

## Demo 2: Multi-Hypothesis Pose Estimation

**Purpose:** Demonstrate probabilistic model handling ambiguous scenes.

**Features:**
- Generates multiple pose hypotheses
- Shows hypothesis diversity
- Validates hypotheses using rendering

**Usage:**
```bash
python demo_hypothesis.py \
    --config configs/7scenes.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs \
    --pretrained_model_path /path/to/probabilistic_checkpoint.pth \
    --output_dir demo_output
```

**Output:**
- Histogram of hypothesis errors
- Mixture weight visualization
- Best hypothesis selection

## Demo 3: Comparison Demo

**Purpose:** Compare baseline RAP vs RAP-ID extensions.

**Features:**
- Side-by-side comparison of all models
- Error distribution visualization
- Statistical summary

**Usage:**
```bash
python demo_comparison.py \
    --config configs/7scenes.txt \
    --datadir /path/to/data \
    --model_path /path/to/3dgs \
    --baseline_checkpoint /path/to/baseline.pth \
    --uaas_checkpoint /path/to/uaas.pth \
    --probabilistic_checkpoint /path/to/probabilistic.pth \
    --num_samples 50 \
    --output_dir demo_output
```

**Output:**
- Error distribution plots
- Box plot comparisons
- JSON results file
- Improvement statistics

## Demo Output Format

All demos generate:
- **Visualizations:** PNG images with plots and heatmaps
- **Statistics:** Printed summaries and JSON files
- **Reports:** Comparison metrics and analysis

## Integration with Testing

Demos can be used as integration tests:
```bash
# Run demos as part of test suite
pytest tests/test_demos.py -v
```

## Use Cases

### 1. **Research Presentations**
- Visual demonstrations of improvements
- Error analysis and failure cases
- Uncertainty quantification

### 2. **Ablation Studies**
- Compare individual components
- Measure contribution of each extension
- Identify best configurations

### 3. **Robustness Testing**
- Test on challenging scenes
- Analyze failure modes
- Validate uncertainty calibration

### 4. **User Documentation**
- Show practical usage
- Demonstrate capabilities
- Provide examples

## Future Enhancements

1. **Interactive Demos:** Web-based visualization
2. **Video Demos:** Animated comparisons
3. **Notebooks:** Jupyter notebook versions
4. **Real-time Demos:** Live camera feed processing

