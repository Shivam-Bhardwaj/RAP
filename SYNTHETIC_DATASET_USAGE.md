# Synthetic Dataset Usage

## Overview

Synthetic datasets are **generated on-demand** from source data and are **not stored in Git**. This approach:
- Keeps repository size small
- Avoids Git LFS storage issues
- Allows flexible dataset sizes
- Ensures data is always fresh

## Generating Synthetic Datasets

### Command Line

```bash
python tests/synthetic_dataset.py \
    --source data/Cambridge/KingsCollege/colmap \
    --output tests/synthetic_test_dataset \
    --num_train 800 \
    --num_test 200 \
    --seed 42
```

### In Tests

The e2e tests automatically generate synthetic datasets if:
1. A source dataset is found in common locations
2. Or `RAP_TEST_DATASET_PATH` environment variable is set
3. Or an existing synthetic dataset exists at `tests/synthetic_test_dataset`

### Parameters

- `--source`: Path to source COLMAP dataset (must have `sparse/0/` and `images/`)
- `--output`: Output directory for synthetic dataset
- `--num_train`: Number of training images (default: 10)
- `--num_test`: Number of test images (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)

## What Gets Created

```
synthetic_test_dataset/
├── images/           # Copied image files
├── sparse/0/         # COLMAP structure
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
├── model/
│   └── cameras.json  # Camera parameters
└── list_test.txt     # Test split
```

## Usage in Tests

Synthetic datasets are automatically used by:
- `tests/test_e2e.py` - End-to-end tests
- All trainer initialization tests
- Full training loop tests

## Example: Generate Large Dataset

```bash
# Generate 1000 images (800 train, 200 test)
python tests/synthetic_dataset.py \
    --source data/Cambridge/KingsCollege/colmap \
    --output tests/synthetic_test_dataset \
    --num_train 800 \
    --num_test 200
```

## CI/CD Integration

In CI/CD pipelines, generate synthetic datasets on-the-fly:

```yaml
# .gitlab-ci.yml or .github/workflows/test.yml
test:
  script:
    - python tests/synthetic_dataset.py --source $SOURCE_DATASET --output tests/synthetic_test_dataset --num_train 100 --num_test 20
    - pytest tests/test_e2e.py -v
```

## Benefits

1. **No Git Storage**: Synthetic datasets are not committed, saving space
2. **Flexible**: Generate any size dataset needed
3. **Reproducible**: Uses random seed for consistency
4. **Fast**: Copies from existing dataset, no rendering needed
5. **Clean**: Each test run gets fresh dataset

## Notes

- Source dataset must be a valid COLMAP dataset
- Generated datasets are temporary (can be deleted after tests)
- For permanent datasets, use `--output` to a persistent location
- Large datasets (800+ images) may take a few minutes to generate

