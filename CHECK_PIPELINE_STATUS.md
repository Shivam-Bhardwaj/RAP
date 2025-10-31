# Full Pipeline - Quick Reference

## Check Status
```bash
# Check if running
ps aux | grep run_full_pipeline_end_to_end | grep -v grep

# Check PID
cat /tmp/rap_pipeline.pid 2>/dev/null || echo "No PID file"

# Check GPU usage
nvidia-smi
```

## View Logs
```bash
# Main log (nohup output)
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/nohup.log

# Detailed pipeline log
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/full_pipeline_*.log

# Stage-specific logs
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/gs_training.log
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/rap_*_training.log
tail -f output/Cambridge/KingsCollege/full_pipeline_logs/benchmark_full_pipeline.log
```

## Pipeline Stages

1. **GS Training** (~30min - 2 hours)
   - Checks if checkpoint exists, skips if found
   - Log: `full_pipeline_logs/gs_training.log`

2. **RAP Training** (~4-10 hours)
   - Trains: Baseline → UAAS → Probabilistic → Semantic
   - Logs: `full_pipeline_logs/rap_*_training.log`

3. **Benchmarking** (~30min - 1 hour)
   - Compares all 4 models
   - Results: `benchmark_full_pipeline_results.json`
   - Charts: `benchmark_full_pipeline_results_charts_*.png`

4. **Dynamic Robustness** (~30min - 1 hour)
   - Tests scene modification robustness
   - Results: `dynamic_scene_robustness_results.json`

## Stop Pipeline (if needed)
```bash
kill $(cat /tmp/rap_pipeline.pid) 2>/dev/null || echo "Not running"
```

## When Complete
```bash
# View final summary
tail -100 output/Cambridge/KingsCollege/full_pipeline_logs/full_pipeline_*.log

# Analyze results
python analyze_results.py benchmark_full_pipeline_results.json

# View charts
ls -lh benchmark_full_pipeline_results_charts_*.png
ls -lh dynamic_scene_robustness_results_*.png
```

