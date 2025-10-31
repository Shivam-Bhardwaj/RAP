# Full Pipeline Status

## Quick Commands

### Check if pipeline is running:
```bash
tmux list-sessions | grep rap_full_pipeline
```

### Watch progress:
```bash
tmux attach -t rap_full_pipeline
# Press Ctrl+B, then D to detach (keeps running)
```

### View logs in real-time:
```bash
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/full_pipeline_*.log
```

### Check latest log:
```bash
ls -lth output/Cambridge/KingsCollege/full_pipeline_logs/ | head -5
```

### Check training progress:
```bash
# GS training
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/gs_training.log

# RAP training (check which one is running)
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/rap_*_training.log
```

### Check GPU usage:
```bash
watch -n 1 nvidia-smi
```

### Stop pipeline (if needed):
```bash
tmux kill-session -t rap_full_pipeline
```

## Pipeline Stages

1. **GS Training** (~30min - 2 hours)
   - Log: `full_pipeline_logs/gs_training.log`

2. **RAP Training** (~4-10 hours total)
   - Baseline: `full_pipeline_logs/rap_baseline_training.log`
   - UAAS: `full_pipeline_logs/rap_uaas_training.log`
   - Probabilistic: `full_pipeline_logs/rap_probabilistic_training.log`
   - Semantic: `full_pipeline_logs/rap_semantic_training.log`

3. **Benchmarking** (~30min - 1 hour)
   - Log: `full_pipeline_logs/benchmark_full_pipeline.log`
   - Results: `benchmark_full_pipeline_results.json`
   - Charts: `benchmark_full_pipeline_results_charts_*.png`

4. **Dynamic Robustness** (~30min - 1 hour)
   - Log: `full_pipeline_logs/dynamic_robustness_test.log`
   - Results: `dynamic_scene_robustness_results.json`
   - Charts: `dynamic_scene_robustness_results_*.png`

## Expected Timeline

- **GS Training**: 30min - 2 hours
- **RAP Training**: 4-10 hours (4 models × 1-2.5 hours each)
- **Benchmarking**: 30min - 1 hour
- **Dynamic Robustness**: 30min - 1 hour
- **Total**: ~6-14 hours

## When Pipeline Completes

Check the final summary:
```bash
tail -50 output/Cambridge/KingsCollege/full_pipeline_logs/full_pipeline_*.log
```

View results:
```bash
python analyze_results.py benchmark_full_pipeline_results.json
ls -lh benchmark_full_pipeline_results_charts_*.png
ls -lh dynamic_scene_robustness_results_*.png
```

