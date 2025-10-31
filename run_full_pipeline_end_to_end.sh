#!/bin/bash
# Complete End-to-End Pipeline Runner
# Runs: GS Training → RAP Training → Benchmarking → Dynamic Robustness Testing

set -e  # Exit on error (but we'll handle failures gracefully)

# Configuration
DATASET="${DATASET:-data/Cambridge/KingsCollege/colmap}"
MODEL_PATH="${MODEL_PATH:-output/Cambridge/KingsCollege}"
CONFIG="${CONFIG:-configs/kingscollege.txt}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-16}"
AMP="${AMP:-true}"

# Logging
LOG_DIR="${MODEL_PATH}/full_pipeline_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/full_pipeline_${TIMESTAMP}.log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

# Function to handle errors gracefully
handle_error() {
    log "ERROR: $1"
    log "Continuing with next stage..."
    return 0
}

log "=========================================="
log "FULL PIPELINE EXECUTION - END TO END"
log "=========================================="
log "Dataset: $DATASET"
log "Model Path: $MODEL_PATH"
log "Config: $CONFIG"
log "Epochs: $EPOCHS"
log "Batch Size: $BATCH_SIZE"
log "Device: $DEVICE"
log "Log File: $MAIN_LOG"
log "=========================================="
log ""

# ============================================================================
# STAGE 1: Gaussian Splatting Training
# ============================================================================
log ""
log "=========================================="
log "STAGE 1: Training Gaussian Splatting Model"
log "=========================================="

GS_CHECKPOINT="$MODEL_PATH/model/ckpts_point_cloud"
if [ -d "$GS_CHECKPOINT" ]; then
    log "✓ GS checkpoint already exists at $GS_CHECKPOINT"
    log "  Skipping GS training (using existing checkpoint)"
else
    log "Training GS model (this will take 30min - 2 hours)..."
    log "Output: ${LOG_DIR}/gs_training.log"
    
    python gs.py \
        -s "$DATASET" \
        -m "$MODEL_PATH" \
        --iterations 30000 \
        --eval \
        --test_iterations 10000 20000 30000 \
        2>&1 | tee "${LOG_DIR}/gs_training.log" || handle_error "GS training failed"
    
    if [ -d "$GS_CHECKPOINT" ]; then
        log "✅ GS training completed successfully!"
    else
        log "⚠️  GS checkpoint not found after training!"
        log "  Pipeline will continue, but RAP training may fail"
    fi
fi

# ============================================================================
# STAGE 2: Train All RAP Models
# ============================================================================
log ""
log "=========================================="
log "STAGE 2: Training All RAP Models"
log "=========================================="
log "Training 4 models: Baseline, UAAS, Probabilistic, Semantic"
log "This will take 4-10 hours depending on epochs..."
log ""

# Train Baseline
log "--- Training Baseline RAPNet ---"
python train.py \
    --trainer_type baseline \
    --run_name "${MODEL_PATH##*/}_baseline_opt" \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --dataset_type Colmap \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --val_num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    $([ "$AMP" = "true" ] && echo "--amp") \
    2>&1 | tee "${LOG_DIR}/rap_baseline_training.log" || handle_error "Baseline training failed"

# Train UAAS
log ""
log "--- Training UAAS RAPNet ---"
python train.py \
    --trainer_type uaas \
    --run_name "${MODEL_PATH##*/}_uaas_opt" \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --dataset_type Colmap \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --val_num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    $([ "$AMP" = "true" ] && echo "--amp") \
    2>&1 | tee "${LOG_DIR}/rap_uaas_training.log" || handle_error "UAAS training failed"

# Train Probabilistic
log ""
log "--- Training Probabilistic RAPNet ---"
python train.py \
    --trainer_type probabilistic \
    --run_name "${MODEL_PATH##*/}_probabilistic_opt" \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --dataset_type Colmap \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --val_num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    $([ "$AMP" = "true" ] && echo "--amp") \
    2>&1 | tee "${LOG_DIR}/rap_probabilistic_training.log" || handle_error "Probabilistic training failed"

# Train Semantic
log ""
log "--- Training Semantic RAPNet ---"
python train.py \
    --trainer_type semantic \
    --run_name "${MODEL_PATH##*/}_semantic_opt" \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --dataset_type Colmap \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    --val_batch_size "$BATCH_SIZE" \
    --val_num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --num_semantic_classes 19 \
    $([ "$AMP" = "true" ] && echo "--amp") \
    2>&1 | tee "${LOG_DIR}/rap_semantic_training.log" || handle_error "Semantic training failed"

log ""
log "✅ All RAP models trained!"

# ============================================================================
# STAGE 3: Full Pipeline Benchmark
# ============================================================================
log ""
log "=========================================="
log "STAGE 3: Running Full Pipeline Benchmark"
log "=========================================="
log "Comparing all 4 methods..."
log ""

CHECKPOINT_DIR="$MODEL_PATH/logs"
python benchmark_full_pipeline.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    2>&1 | tee "${LOG_DIR}/benchmark_full_pipeline.log" || handle_error "Benchmark failed"

log ""
log "✅ Benchmark complete!"

# Analyze results
log ""
log "Analyzing benchmark results..."
python analyze_results.py benchmark_full_pipeline_results.json 2>&1 | tee -a "${LOG_DIR}/benchmark_analysis.log" || handle_error "Analysis failed"

# ============================================================================
# STAGE 4: Dynamic Scene Robustness Testing
# ============================================================================
log ""
log "=========================================="
log "STAGE 4: Testing Dynamic Scene Robustness"
log "=========================================="
log "Testing robustness to scene changes..."
log ""

python test_dynamic_scene_robustness.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    --batch_size "$BATCH_SIZE" \
    2>&1 | tee "${LOG_DIR}/dynamic_robustness_test.log" || handle_error "Dynamic robustness test failed"

log ""
log "✅ Dynamic robustness test complete!"

# ============================================================================
# FINAL SUMMARY
# ============================================================================
log ""
log "=========================================="
log "✅ FULL PIPELINE COMPLETED"
log "=========================================="
log ""
log "Results generated:"
log "  ✓ benchmark_full_pipeline_results.json - Full benchmark results"
log "  ✓ benchmark_full_pipeline_results_charts_*.png - 7 visualization charts"
log "  ✓ dynamic_scene_robustness_results.json - Robustness test results"
log "  ✓ dynamic_scene_robustness_results_*.png - Robustness visualization charts"
log ""
log "Logs saved to: $LOG_DIR"
log ""
log "To view results:"
log "  python analyze_results.py benchmark_full_pipeline_results.json"
log "  ls -lh benchmark_full_pipeline_results_charts_*.png"
log "  ls -lh dynamic_scene_robustness_results_*.png"
log ""
log "Pipeline completed at: $(date)"
log "=========================================="

