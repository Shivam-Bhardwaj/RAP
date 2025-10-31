#!/bin/bash
# Complete pipeline runner: Train all models, benchmark, and test robustness

set -e  # Exit on error

# Configuration
DATASET="${1:-data/Cambridge/KingsCollege/colmap}"
MODEL_PATH="${2:-output/Cambridge/KingsCollege}"
CONFIG="${3:-configs/7scenes.txt}"
EPOCHS="${4:-100}"
BATCH_SIZE="${5:-4}"
DEVICE="${6:-cuda}"
SKIP_GS="${7:-false}"  # Set to 'true' to skip GS training

echo "=========================================="
echo "COMPLETE PIPELINE EXECUTION"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Skip GS Training: $SKIP_GS"
echo "=========================================="
echo ""

# Step 1: Train GS Model (or skip if checkpoint exists or SKIP_GS=true)
echo "=========================================="
echo "STEP 1: Gaussian Splatting Model"
echo "=========================================="

GS_CHECKPOINT="$MODEL_PATH/model/ckpts_point_cloud"
if [ "$SKIP_GS" = "true" ] || [ -d "$GS_CHECKPOINT" ]; then
    if [ -d "$GS_CHECKPOINT" ]; then
        echo "✓ GS checkpoint found at $GS_CHECKPOINT"
        echo "  Skipping GS training (using existing checkpoint)"
    else
        echo "⚠️  Skipping GS training (SKIP_GS=true)"
        echo "  Warning: No GS checkpoint found - RAP model training will fail!"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "Training GS model (this will take 30min - 2 hours)..."
    echo ""
    
    # Try with CUDA_LAUNCH_BLOCKING=1 for better error messages
    CUDA_LAUNCH_BLOCKING=1 python gs.py \
        -s "$DATASET" \
        -m "$MODEL_PATH" \
        --iterations 30000 \
        --eval || {
        echo ""
        echo "⚠️  GS training failed!"
        echo "   This may be due to:"
        echo "   - Dataset format issues"
        echo "   - GPU memory issues"
        echo "   - CUDA compatibility"
        echo ""
        echo "   Options:"
        echo "   1. Fix the GS training issue and re-run"
        echo "   2. Use existing GS checkpoint (if available)"
        echo "   3. Skip GS step: ./run_full_pipeline.sh ... ... ... false true"
        echo ""
        exit 1
    }
    
    # Verify GS checkpoint was created
    if [ ! -d "$GS_CHECKPOINT" ]; then
        echo "⚠️  Warning: GS checkpoint not found after training."
        exit 1
    fi
    
    echo ""
    echo "✓ GS model trained successfully"
fi

echo ""

# Step 2: Train All Models
echo "=========================================="
echo "STEP 2: Training All Models"
echo "=========================================="
echo "This will take 4-10 hours..."
echo ""

./train_all_models.sh "$DATASET" "$MODEL_PATH" "$CONFIG" "$EPOCHS" "$BATCH_SIZE"

# Verify checkpoints
CHECKPOINT_DIR="$MODEL_PATH/logs"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "⚠️  Warning: Checkpoints directory not found. Training may have failed."
    exit 1
fi

echo ""
echo "✓ All models trained"
echo ""

# Step 3: Run Full Pipeline Benchmark
echo "=========================================="
echo "STEP 3: Running Full Pipeline Benchmark"
echo "=========================================="
echo "Comparing all 4 methods..."
echo ""

python benchmark_full_pipeline.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "✓ Benchmark complete"
echo ""

# Step 4: Analyze Results
echo "=========================================="
echo "STEP 4: Analyzing Results"
echo "=========================================="
echo ""

python analyze_results.py benchmark_full_pipeline_results.json

echo ""
echo "✓ Analysis complete"
echo ""

# Step 5: Test Dynamic Robustness
echo "=========================================="
echo "STEP 5: Testing Dynamic Scene Robustness"
echo "=========================================="
echo "Testing robustness to scene changes..."
echo ""

python test_dynamic_scene_robustness.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "✓ Dynamic robustness test complete"
echo ""

# Summary
echo "=========================================="
echo "✅ COMPLETE PIPELINE FINISHED"
echo "=========================================="
echo ""
echo "Results generated:"
echo "  ✓ benchmark_full_pipeline_results.json - Full benchmark results"
echo "  ✓ benchmark_full_pipeline_results_charts_*.png - 7 visualization charts"
echo "  ✓ dynamic_scene_robustness_results.json - Robustness test results"
echo "  ✓ dynamic_scene_robustness_results_*.png - Robustness visualization charts"
echo ""
echo "Next steps:"
echo "  1. Review results: python analyze_results.py"
echo "  2. Check charts: ls -lh *results_charts*.png"
echo "  3. Update README with new results"
echo "  4. Document findings in research paper"
echo ""

