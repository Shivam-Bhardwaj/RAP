# End-to-End Testing Results

## Test Status

Running e2e tests with synthetic dataset generated from source data.

## Synthetic Dataset

- **Location**: `tests/synthetic_test_dataset/`
- **Source**: Generated from actual dataset (not stored in Git)
- **Regeneration**: Use `tests/synthetic_dataset.py` or automatic in tests

## Test Commands

```bash
# Run all e2e tests
pytest tests/test_e2e.py -v

# Run specific test
pytest tests/test_e2e.py::TestTrainingPipeline::test_uaas_trainer_initialization -v

# Run with synthetic dataset
pytest tests/test_e2e.py -v --tb=short
```

## Expected Tests

1. **Model Forward Pass** - All models can process input
2. **UAAS Trainer Initialization** - Trainer sets up correctly
3. **Full Training Iteration** - Can run at least one training step

## Notes

- Synthetic datasets are generated on-demand
- Tests skip if no source dataset available
- GPU tests skip if CUDA not available
- Renderer tests skip if 3DGS not configured

