# Git LFS and Public Fork Issue

## Problem

GitHub does not allow Git LFS uploads to **public forks** for security reasons. The error message indicates:
```
@Shivam-Bhardwaj can not upload new objects to public fork Shivam-Bhardwaj/RAP
```

## Solutions

### Option 1: Convert Fork to Full Repository (Recommended)

1. Go to repository settings
2. Scroll to "Danger Zone"
3. Click "Unlink fork" or "Convert to full repository"
4. Then push normally:
   ```bash
   git push origin main
   git lfs push origin main --all
   ```

### Option 2: Remove Synthetic Dataset from Git (Recommended for Forks)

Remove the large dataset files from Git tracking:

```bash
git rm --cached -r tests/synthetic_test_dataset/images/
git commit -m "Remove synthetic dataset images (use Git LFS locally or generate on-demand)"
```

Then push:
```bash
git push origin main
```

The dataset can be generated locally using:
```bash
python tests/synthetic_dataset.py --source data/Cambridge/KingsCollege/colmap --output tests/synthetic_test_dataset --num_train 800 --num_test 200
```

### Option 3: Use External LFS Storage

Configure Git LFS to use a different storage backend (GitLab LFS, self-hosted, etc.):

```bash
git config lfs.url https://your-lfs-server.com/info/lfs
git push origin main
```

### Option 4: Push Code Only (Current State)

The code changes (`.gitattributes`, `.gitignore`, scripts) can be pushed without LFS files:

```bash
git rm --cached tests/synthetic_test_dataset/images/*.jpg
git commit -m "Remove images from Git (use Git LFS locally)"
git push origin main
```

## Recommendation

For a public fork, **Option 2** is best:
- Keep the code and configuration committed
- Generate the synthetic dataset locally or on CI/CD
- Document the generation process in README

The `.gitattributes` and `synthetic_dataset.py` are still useful for:
- Local development
- CI/CD pipelines
- Contributors who want to generate the dataset

