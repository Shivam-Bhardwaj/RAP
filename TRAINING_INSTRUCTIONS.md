# Training Instructions

## Prerequisites

Before running RAP training, you need to train a Gaussian Splatting model first:

### Step 1: Train Gaussian Splatting Model

```bash
cd /home/ubuntu/RAP

python gs.py \
    --source_path data/Cambridge/KingsCollege/colmap \
    --model_path data/Cambridge/KingsCollege/colmap/model \
    --images data/Cambridge/KingsCollege/colmap/images \
    --resolution 1 \
    --iterations 30000 \
    --eval
```

This will create checkpoints in `data/Cambridge/KingsCollege/colmap/model/ckpts_point_cloud/iteration_XXXX/`

### Step 2: Train RAP Model

Once GS training is complete, run RAP training:

```bash
python train.py \
    --trainer_type uaas \
    --run_name kingscollege_uaas \
    --datadir data/Cambridge/KingsCollege/colmap \
    --model_path data/Cambridge/KingsCollege/colmap/model \
    --dataset_type Colmap \
    --device cuda \
    --batch_size 4 \
    --iterations 100 \
    --epochs 5
```

## Alternative: Check if checkpoint exists

If you already have a GS checkpoint, verify it exists:

```bash
ls -la data/Cambridge/KingsCollege/colmap/model/ckpts_point_cloud/
```

If it exists, you can proceed directly to Step 2.

## Full Training Pipeline

For complete end-to-end training:

1. **GS Training** (30k iterations, ~hours depending on GPU)
   ```bash
   python gs.py --source_path data/Cambridge/KingsCollege/colmap \
                --model_path data/Cambridge/KingsCollege/colmap/model \
                --images data/Cambridge/KingsCollege/colmap/images \
                --resolution 1 --iterations 30000 --eval
   ```

2. **RAP Training** (5 epochs, ~hours depending on GPU)
   ```bash
   python train.py --trainer_type uaas \
                   --run_name kingscollege_uaas \
                   --datadir data/Cambridge/KingsCollege/colmap \
                   --model_path data/Cambridge/KingsCollege/colmap/model \
                   --dataset_type Colmap \
                   --device cuda \
                   --batch_size 4 \
                   --iterations 100 \
                   --epochs 5
   ```

## Monitor Training

```bash
# Watch GS training logs
tail -f logs/gs_kingscollege/*.log

# Watch RAP training logs  
tail -f logs/kingscollege_uaas/*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Notes

- GS training creates the 3D Gaussian representation needed for rendering
- RAP training uses the GS model for rendering novel views
- Both can run on GPU (recommended) or CPU (much slower)
- Training times vary based on dataset size and hardware

