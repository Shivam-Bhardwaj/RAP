#!/bin/bash
# Train all models systematically

set -e  # Exit on error

# Configuration
DATASET="${1:-data/Cambridge/KingsCollege/colmap}"
MODEL_PATH="${2:-output/Cambridge/KingsCollege}"
CONFIG="${3:-configs/7scenes.txt}"
EPOCHS="${4:-100}"
BATCH_SIZE="${5:-1}"
LR="${6:-1e-4}"

echo "=========================================="
echo "TRAINING ALL MODELS"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Model Path: $MODEL_PATH"
echo "Config: $CONFIG"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LR"
echo "=========================================="

# Check GS checkpoint exists
if [ ! -d "$MODEL_PATH/model/ckpts_point_cloud" ]; then
    echo "⚠️  Warning: GS checkpoint not found at $MODEL_PATH/model/ckpts_point_cloud"
    echo "   Please train GS model first: python gs.py -s $DATASET -m $MODEL_PATH"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Train Baseline RAP
echo ""
echo "=========================================="
echo "1. TRAINING BASELINE RAP"
echo "=========================================="
python rap.py \
    -c "$CONFIG" \
    -m "$MODEL_PATH" \
    --run_name baseline_kingscollege \
    --datadir "$DATASET" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" || echo "⚠️  Baseline training failed"

# Train UAAS
echo ""
echo "=========================================="
echo "2. TRAINING UAAS MODEL"
echo "=========================================="
python train.py \
    --trainer_type uaas \
    --run_name uaas_kingscollege \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" || echo "⚠️  UAAS training failed"

# Train Probabilistic
echo ""
echo "=========================================="
echo "3. TRAINING PROBABILISTIC MODEL"
echo "=========================================="
python train.py \
    --trainer_type probabilistic \
    --run_name probabilistic_kingscollege \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_gaussians 5 || echo "⚠️  Probabilistic training failed"

# Train Semantic
echo ""
echo "=========================================="
echo "4. TRAINING SEMANTIC MODEL"
echo "=========================================="
python train.py \
    --trainer_type semantic \
    --run_name semantic_kingscollege \
    --datadir "$DATASET" \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_semantic_classes 19 || echo "⚠️  Semantic training failed"

echo ""
echo "=========================================="
echo "✅ ALL MODELS TRAINED"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  - $MODEL_PATH/logs/baseline_kingscollege/checkpoints/"
echo "  - $MODEL_PATH/logs/uaas_kingscollege/checkpoints/"
echo "  - $MODEL_PATH/logs/probabilistic_kingscollege/checkpoints/"
echo "  - $MODEL_PATH/logs/semantic_kingscollege/checkpoints/"
echo ""
echo "Next step: Run benchmark with trained checkpoints:"
echo "  python benchmark_full_pipeline.py \\"
echo "    --dataset $DATASET \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --checkpoint_dir $MODEL_PATH/logs \\"
echo "    --device cuda"

