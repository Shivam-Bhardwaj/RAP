# Dynamic Scene Robustness Testing Guide

This guide explains how to test model robustness to dynamic scene changes using inpainting and other image modifications.

## Overview

Tests how well models handle modified images that simulate real-world scene changes:
- **Inpainting** - Object removal/addition
- **Occlusion** - Objects blocking the scene
- **Lighting changes** - Brightness/contrast variations
- **Object removal** - Complete object removal with inpainting
- **Blur** - Regional blurring

## Why This Matters

Real scenes change over time:
- Objects are moved/added/removed
- Lighting conditions change
- Temporary occlusions occur
- Structures are modified

A robust model should maintain pose accuracy despite these changes.

---

## Quick Start

### Basic Usage

```bash
python test_dynamic_scene_robustness.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda \
    --batch_size 4 \
    --max_samples 50
```

### Test Specific Modifications

```bash
python test_dynamic_scene_robustness.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --modifications inpaint_center occlusion lighting \
    --device cuda
```

---

## Modification Types

### 1. Inpainting Center (`inpaint_center`)
- Removes center region of image
- Uses OpenCV inpainting to fill
- Simulates object removal from scene center

### 2. Occlusion (`occlusion`)
- Adds random black patches
- Simulates objects blocking parts of scene
- Configurable number and size of patches

### 3. Lighting (`lighting`)
- Modifies brightness and contrast
- Simulates time-of-day changes
- Tests invariance to lighting variations

### 4. Object Removal (`object_removal`)
- Removes random object-sized region
- Uses inpainting to fill
- Tests robustness to structural changes

### 5. Blur Region (`blur_region`)
- Applies Gaussian blur to center region
- Simulates motion blur or focus issues
- Tests robustness to image quality degradation

---

## Arguments

- `--dataset`: Path to dataset (required)
- `--model_path`: Path to GS model directory (required)
- `--checkpoint_dir`: Directory with model checkpoints (optional)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--batch_size`: Batch size (default: 4)
- `--max_samples`: Limit test samples (optional)
- `--modifications`: List of modifications to test (default: all)
- `--models`: Models to test (default: all)
- `--output`: Output JSON file (default: `dynamic_scene_robustness_results.json`)

---

## Output

### JSON Results

Results saved to `dynamic_scene_robustness_results.json`:

```json
{
  "dataset": "...",
  "test_samples": 50,
  "modifications": ["inpaint_center", "occlusion", ...],
  "models": ["baseline", "uaas", ...],
  "results": {
    "baseline": {
      "original": {
        "translation": {"median": 2.29, ...},
        "rotation": {"median": 0.0, ...}
      },
      "inpaint_center": {
        "translation": {"median": 2.45, ...},
        "degradation": {"translation_pct": 6.8, ...}
      },
      ...
    }
  }
}
```

### Visualization Charts

1. **Translation Error Degradation** - Shows how translation error increases under modifications
2. **Rotation Error Degradation** - Shows how rotation error increases under modifications  
3. **Comparison Table** - Side-by-side comparison across models

---

## Interpretation

### Degradation Percentage

- **0%**: No degradation (perfect robustness)
- **+5-10%**: Minor degradation (acceptable)
- **+10-20%**: Moderate degradation (concerning)
- **+20%+**: Significant degradation (poor robustness)

### Best Model

The model with **lowest degradation** across modifications is most robust.

---

## Example Analysis

```bash
# Run test
python test_dynamic_scene_robustness.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs \
    --device cuda

# Analyze results
python -c "
import json
with open('dynamic_scene_robustness_results.json') as f:
    d = json.load(f)

print('Model Robustness Summary:')
for model in d['results']:
    print(f'\n{model.upper()}:')
    for mod in d['modifications']:
        deg = d['results'][model][mod].get('degradation', {})
        print(f'  {mod}: +{deg.get(\"translation_pct\", 0):.1f}% translation, '
              f'+{deg.get(\"rotation_pct\", 0):.1f}% rotation')
"
```

---

## Advanced Usage

### Custom Modification

Add custom modifications in `ImageModifier` class:

```python
def custom_modification(self, image, ...):
    # Your modification logic
    return modified_image
```

### Testing on Multiple Datasets

```bash
for dataset in data/Cambridge/*/colmap; do
    python test_dynamic_scene_robustness.py \
        --dataset "$dataset" \
        --model_path "output/$(basename $(dirname $dataset))" \
        --checkpoint_dir "output/$(basename $(dirname $dataset))/logs" \
        --output "robustness_$(basename $(dirname $dataset)).json"
done
```

---

## Integration with Benchmarking

Combine with full pipeline benchmark:

```bash
# 1. Standard benchmark
python benchmark_full_pipeline.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs

# 2. Robustness test
python test_dynamic_scene_robustness.py \
    --dataset data/Cambridge/KingsCollege/colmap \
    --model_path output/Cambridge/KingsCollege \
    --checkpoint_dir output/Cambridge/KingsCollege/logs

# 3. Compare results
# - Standard benchmark shows baseline performance
# - Robustness test shows how performance degrades under changes
```

---

## Research Applications

This testing is valuable for:

1. **Robustness Evaluation** - Compare model robustness
2. **Failure Analysis** - Identify which modifications cause failures
3. **Real-world Validation** - Test applicability to dynamic scenes
4. **Model Selection** - Choose most robust model for production
5. **Improvement Direction** - Guide future model improvements

---

## Troubleshooting

### OpenCV Import Error
```bash
pip install opencv-python opencv-contrib-python
```

### CUDA Out of Memory
Reduce batch size:
```bash
--batch_size 2
```

### Slow Execution
Limit test samples:
```bash
--max_samples 20
```

---

## Next Steps

1. Run robustness tests on all trained models
2. Compare degradation across models
3. Identify which modifications are most challenging
4. Document findings in research paper
5. Use results to improve model robustness

