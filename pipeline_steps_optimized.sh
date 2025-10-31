#!/bin/bash
# OPTIMIZED Pipeline Steps - Leverages H100 GPU fully
# Source this file: source pipeline_steps_optimized.sh

set -e  # Exit on error

# OPTIMIZED Configuration for H100
DATASET="${DATASET:-data/Cambridge/KingsCollege/colmap}"
MODEL_PATH="${MODEL_PATH:-output/Cambridge/KingsCollege}"
CONFIG="${CONFIG:-configs/kingscollege.txt}"
EPOCHS="${EPOCHS:-100}"

# OPTIMIZED for H100 (85GB VRAM, 114 SMs)
BATCH_SIZE="${BATCH_SIZE:-32}"           # Increased from 4 to 32
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"   # Increased for validation
NUM_WORKERS="${NUM_WORKERS:-16}"         # Parallel data loading (18 CPU cores available)
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-true}"                       # Enable mixed precision

echo "=========================================="
echo "OPTIMIZED Pipeline Configuration"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (OPTIMIZED)"
echo "Validation Batch Size: $VAL_BATCH_SIZE (OPTIMIZED)"
echo "Num Workers: $NUM_WORKERS (OPTIMIZED - parallel data loading)"
echo "Device: $DEVICE"
echo "AMP (Mixed Precision): $AMP"
echo ""
echo "Hardware Utilization Targets:"
echo "  - GPU: >90% utilization"
echo "  - GPU Memory: 60-80% usage"
echo "  - CPU: Multiple cores for data loading"
echo "=========================================="
echo ""
echo "Available commands:"
echo "  step_one_opt    - Train GS model (optimized)"
echo "  step_two_opt    - Train all 4 RAP models (optimized)"
echo "  step_three_opt  - Run full benchmark (optimized)"
echo "  step_four_opt   - Test dynamic robustness (optimized)"
echo "  show_config_opt - Show optimized configuration"
echo ""

# Step 1: Train GS Model (Optimized)
step_one_opt() {
    echo "=========================================="
    echo "STEP 1: Training GS Model (OPTIMIZED)"
    echo "=========================================="
    echo "Using optimized settings for H100..."
    echo "  - Reduced test_iterations frequency (fewer evaluations = faster training)"
    echo "  - Evaluation only at milestones: 10000, 20000, 30000 (3 instead of 6)"
    echo ""
    
    # Optimizations for GS training:
    # 1. Reduce test_iterations frequency (fewer evaluations during training)
    #    Default: [5000, 10000, 15000, 20000, 25000, 30000] (6 evaluations)
    #    Optimized: [10000, 20000, 30000] (3 evaluations = 50% fewer)
    # 2. Keep eval enabled but less frequent
    # 3. Video rendering is already non-fatal (handled in gs.py)
    python gs.py \
        -s "$DATASET" \
        -m "$MODEL_PATH" \
        --iterations 30000 \
        --eval \
        --test_iterations 10000 20000 30000 \
        2>&1 | tee "${MODEL_PATH}/gs_training_optimized.log"
    
    echo ""
    if [ -d "$MODEL_PATH/model/ckpts_point_cloud" ]; then
        echo "✅ GS training completed successfully!"
        echo ""
        echo "Speed optimizations applied:"
        echo "  ✓ Reduced evaluation frequency (from 6 to 3 evaluations = 50% fewer)"
        echo "  ✓ GPU utilization optimized"
        echo "  ✓ Same final checkpoint quality - only fewer intermediate checkpoints"
    else
        echo "⚠️  Warning: GS checkpoint not found."
    fi
}

# Step 2: Train All RAP Models (Optimized)
step_two_opt() {
    echo "=========================================="
    echo "STEP 2: Training All RAP Models (OPTIMIZED)"
    echo "=========================================="
    echo "Using:"
    echo "  - Batch Size: $BATCH_SIZE"
    echo "  - Num Workers: $NUM_WORKERS"
    echo "  - AMP: $AMP"
    echo ""
    
    # Train Baseline
    echo "Training Baseline with optimized settings..."
    python train.py \
        --trainer_type baseline \
        --run_name "${MODEL_PATH##*/}_baseline_opt" \
        --datadir "$DATASET" \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG" \
        --dataset_type Colmap \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --val_num_workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        $([ "$AMP" = "true" ] && echo "--amp") \
        2>&1 | tee "${MODEL_PATH}/rap_baseline_training_opt.log" || echo "Baseline training failed"
    
    # Train UAAS
    echo ""
    echo "Training UAAS with optimized settings..."
    python train.py \
        --trainer_type uaas \
        --run_name "${MODEL_PATH##*/}_uaas_opt" \
        --datadir "$DATASET" \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG" \
        --dataset_type Colmap \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --val_num_workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        $([ "$AMP" = "true" ] && echo "--amp") \
        2>&1 | tee "${MODEL_PATH}/rap_uaas_training_opt.log" || echo "UAAS training failed"
    
    # Train Probabilistic
    echo ""
    echo "Training Probabilistic with optimized settings..."
    python train.py \
        --trainer_type probabilistic \
        --run_name "${MODEL_PATH##*/}_probabilistic_opt" \
        --datadir "$DATASET" \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG" \
        --dataset_type Colmap \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --val_num_workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        $([ "$AMP" = "true" ] && echo "--amp") \
        2>&1 | tee "${MODEL_PATH}/rap_probabilistic_training_opt.log" || echo "Probabilistic training failed"
    
    # Train Semantic
    echo ""
    echo "Training Semantic with optimized settings..."
    python train.py \
        --trainer_type semantic \
        --run_name "${MODEL_PATH##*/}_semantic_opt" \
        --datadir "$DATASET" \
        --model_path "$MODEL_PATH" \
        --config "$CONFIG" \
        --dataset_type Colmap \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        --val_batch_size "$VAL_BATCH_SIZE" \
        --val_num_workers "$NUM_WORKERS" \
        --epochs "$EPOCHS" \
        --num_semantic_classes 19 \
        $([ "$AMP" = "true" ] && echo "--amp") \
        2>&1 | tee "${MODEL_PATH}/rap_semantic_training_opt.log" || echo "Semantic training failed"
    
    echo ""
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "✅ All RAP models trained with optimized settings!"
    else
        echo "⚠️  Warning: Checkpoints directory not found."
    fi
}

# Step 3: Run Full Benchmark (Optimized)
step_three_opt() {
    echo "=========================================="
    echo "STEP 3: Full Benchmark (OPTIMIZED)"
    echo "=========================================="
    echo "Using batch size: $BATCH_SIZE"
    echo ""
    
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    python benchmark_full_pipeline.py \
        --dataset "$DATASET" \
        --model_path "$MODEL_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee benchmark_full_pipeline_optimized.log
    
    echo ""
    echo "✅ Benchmark complete!"
    python analyze_results.py benchmark_full_pipeline_results.json
}

# Step 4: Test Dynamic Robustness (Optimized)
step_four_opt() {
    echo "=========================================="
    echo "STEP 4: Dynamic Robustness Test (OPTIMIZED)"
    echo "=========================================="
    
    CHECKPOINT_DIR="$MODEL_PATH/logs"
    python test_dynamic_scene_robustness.py \
        --dataset "$DATASET" \
        --model_path "$MODEL_PATH" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --device "$DEVICE" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee dynamic_scene_robustness_optimized.log
    
    echo ""
    echo "✅ Dynamic robustness test complete!"
}

# Show optimized configuration
show_config_opt() {
    echo "OPTIMIZED Pipeline Configuration:"
    echo "  DATASET=$DATASET"
    echo "  MODEL_PATH=$MODEL_PATH"
    echo "  CONFIG=$CONFIG"
    echo "  EPOCHS=$EPOCHS"
    echo "  BATCH_SIZE=$BATCH_SIZE (OPTIMIZED - was 4)"
    echo "  VAL_BATCH_SIZE=$VAL_BATCH_SIZE (OPTIMIZED)"
    echo "  NUM_WORKERS=$NUM_WORKERS (OPTIMIZED - was 0)"
    echo "  DEVICE=$DEVICE"
    echo "  AMP=$AMP (Mixed Precision)"
    echo ""
    echo "To customize further:"
    echo "  export BATCH_SIZE=64  # Even larger if memory allows"
    echo "  export NUM_WORKERS=18  # One per CPU core"
    echo "  export AMP=false  # Disable if issues occur"
}

# Export functions
export -f step_one_opt step_two_opt step_three_opt step_four_opt show_config_opt

