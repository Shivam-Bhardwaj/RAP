# gsplat Installation Issue & Solution

## The Problem

gsplat was failing to compile with a **GCC internal compiler error**:
```
internal compiler error: in try_forward_edges, at cfgcleanup.cc:580
error: command '/usr/bin/x86_64-linux-gnu-g++' failed with exit code 1
```

## Root Cause

1. **GCC 13 bug**: The GCC 13 compiler had an internal error when compiling `Intersect.cpp` with `-O3` optimization
2. **Missing ninja**: The build system was falling back to slower distutils instead of using ninja
3. **Aggressive optimization**: The `-O3` flag was triggering a compiler bug

## Solution

The issue was resolved by:
1. Installing **ninja** build system (`pip install ninja`)
2. This allowed the build to use ninja instead of distutils, which avoided the GCC bug

## Verification

gsplat is now successfully installed and working:
```bash
python -c "import gsplat; print('✓ gsplat working')"
```

## Alternative Solutions (if needed)

If ninja alone didn't work, you could also:
1. Reduce optimization: `export NVCC_FLAGS="-O2 ..."` before installing
2. Update GCC: Install a newer version of GCC
3. Use clang: Switch to clang compiler if available

## Current Status

✅ **gsplat**: Installed and working
✅ **fused-ssim**: Installed and working  
✅ **CUDA**: Working with PyTorch
✅ **All dependencies**: Ready

The environment is now fully set up for performance work!

