#!/bin/bash
# Cleanup script - removes runtime files while preserving datasets and code

set -e

echo "=========================================="
echo "CLEANING RUNTIME FILES"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - Training checkpoints (output/*/logs)"
echo "  - GS model checkpoints (output/*/model/ckpts_point_cloud)"
echo "  - Wandb logs"
echo "  - Python cache files"
echo "  - Benchmark results (optional)"
echo ""
echo "Will preserve:"
echo "  - Datasets (data/)"
echo "  - Source code"
echo "  - Configs"
echo "  - Scripts"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Remove training logs
echo "Removing training logs..."
find . -type d -name "logs" -not -path "./submodules/*" -exec rm -rf {} + 2>/dev/null || true

# Remove GS checkpoints
echo "Removing GS model checkpoints..."
find output -type d -name "ckpts_point_cloud" -exec rm -rf {} + 2>/dev/null || true
find output -type d -name "ckpts" -exec rm -rf {} + 2>/dev/null || true

# Remove wandb logs
echo "Removing wandb logs..."
rm -rf wandb/ 2>/dev/null || true

# Remove Python cache
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -not -path "./submodules/*" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -not -path "./submodules/*" -delete 2>/dev/null || true
find . -type d -name "*.egg-info" -not -path "./submodules/*" -exec rm -rf {} + 2>/dev/null || true

# Remove benchmark results (optional - commented out by default)
# echo "Removing benchmark results..."
# rm -f benchmark_full_pipeline_results.json
# rm -f benchmark_synthetic_results.json
# rm -f benchmark_full_pipeline_results_charts*.png
# rm -f benchmark_synthetic_results_charts*.png

# Clean output directory structure (keep directory but remove contents)
echo "Cleaning output directory..."
find output -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ CLEANUP COMPLETE"
echo "=========================================="
echo ""
echo "Removed:"
echo "  ✓ Training logs"
echo "  ✓ GS checkpoints"
echo "  ✓ Wandb logs"
echo "  ✓ Python cache files"
echo ""
echo "Preserved:"
echo "  ✓ Datasets in data/"
echo "  ✓ All source code"
echo "  ✓ Configs and scripts"
echo ""
echo "Next steps:"
echo "  1. Train GS models: python gs.py -s <dataset> -m <output>"
echo "  2. Train all models: ./train_all_models.sh <dataset> <output> <config> <epochs>"
echo "  3. Run benchmark: python benchmark_full_pipeline.py --dataset <dataset> --model_path <output>"
echo "  4. Analyze: python analyze_results.py"

