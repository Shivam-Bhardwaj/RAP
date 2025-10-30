# Performance Profiling Guide

## Quick Benchmark

Run the quick benchmark to test basic optimizations:

```bash
cd /home/curious/projects/RAP
source venv/bin/activate
source activate_cuda.sh
python quick_benchmark.py
```

This will measure:
- CPU-GPU transfer speedup (blocking vs non-blocking)
- Stacking operations (individual vs batch CPU transfer)
- Batch processing efficiency

## Full Training Profiling

For detailed training profiling:

```bash
python profile_performance.py \
    -c configs/7Scenes/chess.txt \
    -m /path/to/3dgs/model \
    -d data/7Scenes/chess \
    --iterations 100 \
    --batch_size 8 \
    --export profile_results.json
```

This will profile:
- Model inference at different batch sizes
- Complete training step breakdown
- Data loading performance
- Memory usage

## Using the Profiler in Code

Add profiling to your training code:

```python
from utils.profiler import profile, print_profile_summary

# In your training loop
for epoch in range(epochs):
    for batch in dataloader:
        with profile("data_loading"):
            data, poses = batch
        
        with profile("forward_pass"):
            output = model(data)
        
        with profile("loss_computation"):
            loss = criterion(output, poses)
        
        with profile("backward_pass"):
            loss.backward()
            optimizer.step()

# Print summary
print_profile_summary()
```

## Comparing Before/After

1. **Before optimizations**:
   ```bash
   python profile_performance.py -c configs/... -m ... --export before.json
   ```

2. **After optimizations**:
   ```bash
   python profile_performance.py -c configs/... -m ... --export after.json
   ```

3. **Compare results**:
   ```python
   import json
   before = json.load(open('before.json'))
   after = json.load(open('after.json'))
   
   # Calculate speedup
   for key in before['summary']:
       if key in after['summary']:
           speedup = before['summary'][key]['mean'] / after['summary'][key]['mean']
           print(f"{key}: {speedup:.2f}x speedup")
   ```

## Expected Results

After optimizations, you should see:
- **Data transfers**: 1.1-1.2x faster (non-blocking)
- **RVS rendering**: 1.2-1.5x faster (batch transfers)
- **Overall training**: 1.15-1.3x faster

## Tips

- Run multiple iterations for stable results
- Use warmup iterations before profiling
- Profile on actual data if possible
- Monitor GPU utilization with `nvidia-smi`

