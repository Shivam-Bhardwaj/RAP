# Full Training Run Status

## Training Started

Training command:
```bash
python train.py --trainer_type UAAS \
  --run_name kingscollege_uaas \
  --datadir data/Cambridge/KingsCollege/colmap \
  --model_path data/Cambridge/KingsCollege/colmap/model \
  --dataset_type Colmap \
  --device cuda \
  --batch_size 4 \
  --iterations 100 \
  --rap_resolution 240,320 \
  --epochs 5
```

## Status

- Training is running in background
- Log file: `/tmp/training_full.log`
- Dataset: Cambridge KingsCollege (full dataset)
- Trainer: UAAS (Uncertainty-Aware Adversarial Synthesis)

## Monitor Training

```bash
# Watch logs
tail -f /tmp/training_full.log

# Check GPU usage
nvidia-smi

# Check process
ps aux | grep train.py
```

## Expected Output

- Training iterations
- Loss values
- Validation metrics
- Model checkpoints saved

## Completion

Training will complete after:
- 5 epochs
- 100 iterations per epoch (or dataset size)
- Final evaluation on test set

