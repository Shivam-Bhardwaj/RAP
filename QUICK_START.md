# Quick Start Commands

## Step 1: Change to RAP Directory

```bash
cd /home/ubuntu/RAP
```

## Step 2: Train Gaussian Splatting (long-running)

```bash
python gs.py \
    --source_path data/Cambridge/KingsCollege/colmap \
    --model_path data/Cambridge/KingsCollege/colmap/model \
    --images data/Cambridge/KingsCollege/colmap/images \
    --resolution 1 \
    --iterations 30000 \
    --eval
```

## Step 3: Train RAP (after GS completes)

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

## One-Liner for GS Training

```bash
cd /home/ubuntu/RAP && python gs.py --source_path data/Cambridge/KingsCollege/colmap --model_path data/Cambridge/KingsCollege/colmap/model --images data/Cambridge/KingsCollege/colmap/images --resolution 1 --iterations 30000 --eval
```

## One-Liner for RAP Training

```bash
cd /home/ubuntu/RAP && python train.py --trainer_type uaas --run_name kingscollege_uaas --datadir data/Cambridge/KingsCollege/colmap --model_path data/Cambridge/KingsCollege/colmap/model --dataset_type Colmap --device cuda --batch_size 4 --iterations 100 --epochs 5
```

