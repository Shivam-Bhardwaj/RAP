# Performance Optimizations for RAP

## Identified Bottlenecks

### 1. Data Loading (IMMEDIATE WIN)
**Issue**: `num_workers=1` in training DataLoader
**Location**: `rap.py:65`
**Impact**: Serial data loading is slow
**Fix**: Increase num_workers (already done in val_dl)

### 2. RVS Rendering (MAJOR BOTTLENECK)
**Issue**: Sequential rendering loop, one image at a time
**Location**: `utils/nvs_utils.py:395, 451, 531`
**Impact**: ~100x slower than batch rendering
**Fix**: Batch rendering if possible, or optimize the loop

### 3. Memory Transfers
**Issue**: Frequent `.cpu()` calls in loops
**Location**: Multiple places
**Impact**: Slow CPU-GPU transfers
**Fix**: Keep data on GPU, batch transfers

### 4. Tensor Operations
**Issue**: Multiple `.to(device)` calls per batch
**Location**: Training loops
**Impact**: Overhead from repeated transfers
**Fix**: Pre-allocate on device

## Optimization Plan

1. ✅ Optimize data loading (quick fix)
2. ⏳ Optimize RVS rendering (biggest impact)
3. ⏳ Reduce memory transfers
4. ⏳ Batch tensor operations

