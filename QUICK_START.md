# Quick Start Guide - Pipeline Steps

## Setup

1. **Start tmux** (if not already running):
   ```bash
   tmux
   # Or if you want a named session:
   tmux new -s rap_pipeline
   ```

2. **Source the pipeline steps script**:
   ```bash
   source pipeline_steps.sh
   ```

3. **Check configuration**:
   ```bash
   show_config
   ```

## Available Commands

After sourcing `pipeline_steps.sh`, you can use these simple commands:

- **`step_one`** - Train Gaussian Splatting model (~30min-2hr)
- **`step_two`** - Train all 4 RAP models (~4-10hr)
- **`step_three`** - Run full benchmark (~5-10min)
- **`step_four`** - Test dynamic robustness (~10-20min)
- **`step_all`** - Run complete pipeline (with pauses between steps)
- **`show_config`** - Show current configuration

## Customizing Configuration

Before running steps, you can set custom values:

```bash
export DATASET="data/Cambridge/KingsCollege/colmap"
export MODEL_PATH="output/Cambridge/KingsCollege"
export CONFIG="configs/7scenes.txt"
export EPOCHS=100
export BATCH_SIZE=4
export DEVICE="cuda"
```

Then source the script again:
```bash
source pipeline_steps.sh
```

## Running the Pipeline

### Option 1: Run step by step (recommended)
```bash
step_one    # Wait for completion
step_two    # Wait for completion
step_three  # Wait for completion
step_four   # Wait for completion
```

### Option 2: Run all at once
```bash
step_all    # Will pause between steps for you to review
```

## Monitoring Progress

All steps output verbose logs:
- GS training: `${MODEL_PATH}/gs_training.log`
- RAP training: `${MODEL_PATH}/rap_training.log`
- Benchmark: `benchmark_full_pipeline.log`
- Robustness: `dynamic_scene_robustness.log`

## Tmux Tips

- **Detach from tmux**: Press `Ctrl+B` then `D`
- **Reattach**: `tmux attach -t rap_pipeline` (or just `tmux attach` if only one session)
- **Split pane**: `Ctrl+B` then `%` (vertical) or `"` (horizontal)
- **Switch panes**: `Ctrl+B` then arrow keys
- **Scroll**: `Ctrl+B` then `[`, use arrow keys, press `q` to exit

## Troubleshooting

If a step fails:
1. Check the log file for that step
2. Fix the issue
3. Re-run just that step (e.g., `step_one` again)
4. Continue with the next step

## Example Session

```bash
# Start tmux
tmux new -s rap_pipeline

# Navigate to project
cd ~/RAP

# Source pipeline steps
source pipeline_steps.sh

# Check config
show_config

# Run steps
step_one    # Wait ~30min-2hr
step_two    # Wait ~4-10hr
step_three  # Wait ~5-10min
step_four   # Wait ~10-20min

# Done! Check results
ls -lh benchmark_full_pipeline_results*.json
ls -lh dynamic_scene_robustness_results*.json
```
