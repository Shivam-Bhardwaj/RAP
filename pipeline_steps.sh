#!/bin/bash
# Simple pipeline commands - source this file to use: source pipeline_steps.sh

set -e  # Exit on error

# Default configuration - modify as needed
DATASET="${DATASET:-data/Cambridge/KingsCollege/colmap}"
MODEL_PATH="${MODEL_PATH:-output/Cambridge/KingsCollege}"
CONFIG="${CONFIG:-configs/7scenes.txt}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cuda}"

echo "=========================================="
echo "Pipeline Configuration"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  step_one    - Train GS model"
echo "  step_two    - Train all 4 RAP models"
echo "  step_three  - Run full benchmark"
echo "  step_four   - Test dynamic robustness"
echo "  step_all    - Run complete pipeline"
echo "  show_config - Show current configuration"
echo ""

# Step 1: Train GS Model
step_one() {
    echo "=========================================="
    echo "STEP 1: Training Gaussian Splatting Model"
    echo "=========================================="
    echo "Dataset: $DATASET"
    echo "Output: $MODEL_PATH"
    echo "This will take 30min - 2 hours..."
    echo ""
    
    python gs.py \
        -s "$DATASET" \
        -m "$MODEL_PATH" \
        --iterations 30000 \
        --eval \
        2>&1 | tee "${MODEL_PATH}/gs_training.log"
    
    echo ""
    if [ -d "$MODEL_PATH/model/ckpts_point_cloud" ]; then
        echo "✅ GS training completed successfully!"
        echo "   Checkpoint: $MODEL_PATH/model/ckpts_point_cloud"
    else
        echo "⚠️  Warning: GS checkpoint not found. Training may have failed."
    fi
}

# Step 2: Train All RAP Models
step_two() {
    echo "=========================================="
    echo "STEP 2: Training All RAP Models"
    echo "=========================================="
    echo "This will train: Baseline, UAAS, Probabilistic, Semantic"
    echo "This will take 4-10 hours..."
    echo ""
    
    ./train_all_models.sh "$DATASET" "$MODEL_PATH" "$CONFIG" "$EPOCHS" "$BATCH_SIZE" \
        2>&1 | tee "${MODEL_PATH}/rap_training.log"
    
    echo ""
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "✅ All RAP models trained successfully!"
        echo "   Checkpoints: $CHECKPOINT_DIR"
    else
        echo "⚠️  Warning: Checkpoints directory not found."
    fi
}

# Step 3: Run Full Benchmark
step_three() {
    echo "=========================================="
    echo "STEP 3: Running Full Pipeline Benchmark"
    echo "=========================================="
    echo "Comparing all 4 methods..."
    echo ""
    
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    python benchmark_full_pipeline.py \
        --dataset "$DATASET" \
        --model_path "$MODEL_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee benchmark_full_pipeline.log
    
    echo ""
    echo "✅ Benchmark complete!"
    echo "   Results: benchmark_full_pipeline_results.json"
    echo "   Charts: benchmark_full_pipeline_results_charts_*.png"
    
    # Analyze results
    echo ""
    echo "Running analysis..."
    python analyze_results.py benchmark_full_pipeline_results.json
}

# Step 4: Test Dynamic Robustness
step_four() {
    echo "=========================================="
    echo "STEP 4: Testing Dynamic Scene Robustness"
    echo "=========================================="
    echo "Testing robustness to scene changes..."
    echo ""
    
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    python test_dynamic_scene_robustness.py \
        --dataset "$DATASET" \
        --model_path "$MODEL_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee dynamic_scene_robustness.log
    
    echo ""
    echo "✅ Dynamic robustness test complete!"
    echo "   Results: dynamic_scene_robustness_results.json"
    echo "   Charts: dynamic_scene_robustness_results_*.png"
}

# Run All Steps
step_all() {
    echo "=========================================="
    echo "RUNNING COMPLETE PIPELINE"
    echo "=========================================="
    echo ""
    
    step_one
    echo ""
    echo "Press Enter to continue to Step 2, or Ctrl+C to stop..."
    read
    
    step_two
    echo ""
    echo "Press Enter to continue to Step 3, or Ctrl+C to stop..."
    read
    
    step_three
    echo ""
    echo "Press Enter to continue to Step 4, or Ctrl+C to stop..."
    read
    
    step_four
    echo ""
    echo "=========================================="
    echo "✅ COMPLETE PIPELINE FINISHED"
    echo "=========================================="
}

# Show current configuration
show_config() {
    echo "Current Pipeline Configuration:"
    echo "  DATASET=$DATASET"
    echo "  MODEL_PATH=$MODEL_PATH"
    echo "  CONFIG=$CONFIG"
    echo "  EPOCHS=$EPOCHS"
    echo "  BATCH_SIZE=$BATCH_SIZE"
    echo "  DEVICE=$DEVICE"
    echo ""
    echo "To change configuration:"
    echo "  export DATASET=your/dataset/path"
    echo "  export MODEL_PATH=your/output/path"
    echo "  etc."
}

# Export functions so they're available
export -f step_one step_two step_three step_four step_all show_config

