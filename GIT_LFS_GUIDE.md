# Git LFS Configuration

This repository uses Git LFS (Large File Storage) to manage large binary files.

## Files Tracked by Git LFS

The following file types are tracked via Git LFS:

### Model Checkpoints
- `*.pt` - PyTorch model checkpoints
- `*.pth` - PyTorch model files
- `*.ckpt` - Checkpoint files
- `*.pkl` - Pickle files (models, data)

### Point Clouds and 3D Data
- `*.ply` - Point cloud files
- `*.bin` - Binary files (COLMAP, etc.)

### Large Data Files
- `*.h5` - HDF5 files
- `*.hdf5` - HDF5 files

### Synthetic Dataset Files (for Testing)
- `tests/synthetic_*/**/*.jpg` - Images in synthetic test datasets
- `tests/synthetic_*/**/*.png` - Images in synthetic test datasets
- `tests/synthetic_*/**/*.ply` - Point clouds in synthetic test datasets
- `tests/synthetic_*/**/*.bin` - Binary files in synthetic test datasets
- `synthetic_*/**/*.jpg` - Images in any synthetic dataset directory
- `synthetic_*/**/*.png` - Images in any synthetic dataset directory
- `synthetic_*/**/*.ply` - Point clouds in any synthetic dataset directory
- `synthetic_*/**/*.bin` - Binary files in any synthetic dataset directory

### Images (if needed in assets)
- `assets/**/*.jpg` - Large JPEG images
- `assets/**/*.png` - Large PNG images

## Usage

### Setting up Git LFS (One-time setup)

```bash
# Install Git LFS (if not already installed)
# On Ubuntu/Debian:
sudo apt-get install git-lfs

# On macOS:
brew install git-lfs

# Initialize Git LFS in this repository
git lfs install

# Track files (already configured in .gitattributes)
git add .gitattributes
git commit -m "Add Git LFS configuration"
```

### Working with LFS Files

Git LFS files are handled automatically:
- When you `git add` a file matching LFS patterns, it's automatically tracked
- When you `git clone`, LFS files are downloaded automatically
- `git lfs pull` - Explicitly download LFS files if needed
- `git lfs ls-files` - List files tracked by LFS

### Checking LFS Status

```bash
# List all LFS tracked files
git lfs ls-files

# Check LFS version
git lfs version

# Check LFS status
git lfs status
```

## Important Notes

1. **Large files already in Git**: If you've already committed large files to Git before enabling LFS, you'll need to migrate them:
   ```bash
   git lfs migrate import --include="*.pt,*.pth,*.ply" --everything
   ```

2. **Clone with LFS**: When cloning, LFS files are automatically downloaded. If you want to skip LFS files:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone <repo-url>
   ```

3. **Bandwidth**: LFS files are stored separately and may have bandwidth limits on GitHub (1GB/month free, then 1GB per $5/month)

4. **File Size Limits**: GitHub allows files up to 100MB in regular Git, and up to 2GB with Git LFS

## Configuration

The `.gitattributes` file defines which files are tracked by Git LFS. To modify:

```bash
# Add a new pattern
git lfs track "*.newformat"

# Remove a pattern
# Edit .gitattributes and remove the line
```

## Troubleshooting

If LFS files aren't downloading:

```bash
# Pull LFS files explicitly
git lfs pull

# Check if LFS is properly initialized
git lfs install

# Verify LFS is tracking files correctly
git lfs ls-files
```

