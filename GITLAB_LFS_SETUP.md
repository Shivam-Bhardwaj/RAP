# Using GitLab for Git LFS

## Overview

Since GitHub blocks LFS uploads to public forks, GitLab is an excellent alternative. GitLab offers:
- **Free Git LFS storage** (10GB per project)
- **No restrictions on forks**
- **Better LFS support** overall

## Setup Options

### Option 1: Migrate Repository to GitLab (Recommended)

1. **Create GitLab Account**
   - Go to https://gitlab.com
   - Sign up for a free account

2. **Create New Repository**
   - Click "New project"
   - Choose "Import project"
   - Or create empty repository first

3. **Push to GitLab**
   ```bash
   # Add GitLab remote
   git remote add gitlab https://gitlab.com/YOUR_USERNAME/RAP-ID.git
   
   # Push code
   git push gitlab main
   
   # Push LFS files
   git lfs push gitlab main --all
   ```

### Option 2: Use GitLab LFS Storage Only (Hybrid)

Keep GitHub for code, use GitLab only for LFS storage:

```bash
# Configure GitLab LFS endpoint
git config lfs.url https://gitlab.com/YOUR_USERNAME/RAP-ID.git/info/lfs

# Push LFS files to GitLab
git lfs push gitlab main --all

# Push code to GitHub (as before)
git push origin main
```

### Option 3: Dual Remote Setup

Push code to both GitHub and GitLab, LFS only to GitLab:

```bash
# Add GitLab remote
git remote add gitlab https://gitlab.com/YOUR_USERNAME/RAP-ID.git

# Configure LFS to use GitLab
git config lfs.url https://gitlab.com/YOUR_USERNAME/RAP-ID.git/info/lfs

# Push code to GitHub
git push origin main

# Push code and LFS to GitLab
git push gitlab main
git lfs push gitlab main --all
```

## Step-by-Step: Complete Migration to GitLab

### 1. Create GitLab Repository

```bash
# On GitLab.com:
# 1. Click "New project"
# 2. Choose "Import project"
# 3. Select "GitHub"
# 4. Authorize and select your repository
# 5. Or create empty repository manually
```

### 2. Configure Git Remote

```bash
cd /home/ubuntu/RAP

# Add GitLab remote
git remote add gitlab https://gitlab.com/YOUR_USERNAME/RAP-ID.git

# Or set as new origin
git remote set-url origin https://gitlab.com/YOUR_USERNAME/RAP-ID.git
```

### 3. Authenticate with GitLab

```bash
# Generate personal access token:
# 1. Go to GitLab -> Settings -> Access Tokens
# 2. Create token with "write_repository" and "write_api" scopes
# 3. Use token as password when pushing

# Or use SSH:
git remote set-url gitlab git@gitlab.com:YOUR_USERNAME/RAP-ID.git
```

### 4. Push Everything

```bash
# Push code
git push gitlab main

# Push LFS files (GitLab handles this automatically)
git lfs push gitlab main --all

# Or just push normally (GitLab detects LFS automatically)
git push gitlab main
```

## GitLab LFS Configuration

### Verify LFS Setup

```bash
# Check LFS configuration
git lfs env

# List LFS files
git lfs ls-files

# Verify GitLab LFS endpoint
git config lfs.url
```

### GitLab-Specific Settings

```bash
# Set GitLab LFS endpoint
git config lfs.url https://gitlab.com/YOUR_USERNAME/RAP-ID.git/info/lfs

# Enable LFS locking (optional)
git config lfs.https://gitlab.com/YOUR_USERNAME/RAP-ID.git/info/lfs.locksverify true
```

## Quick Setup Script

Create a script to automate setup:

```bash
#!/bin/bash
# setup_gitlab_lfs.sh

GITLAB_USER="YOUR_USERNAME"
GITLAB_REPO="RAP-ID"

echo "Setting up GitLab LFS..."

# Add GitLab remote
git remote add gitlab https://gitlab.com/${GITLAB_USER}/${GITLAB_REPO}.git 2>/dev/null || \
git remote set-url gitlab https://gitlab.com/${GITLAB_USER}/${GITLAB_REPO}.git

# Configure LFS
git config lfs.url https://gitlab.com/${GITLAB_USER}/${GITLAB_REPO}.git/info/lfs

# Push code
echo "Pushing code to GitLab..."
git push gitlab main

# Push LFS files
echo "Pushing LFS files to GitLab..."
git lfs push gitlab main --all

echo "Done! Repository available at: https://gitlab.com/${GITLAB_USER}/${GITLAB_REPO}"
```

## GitLab LFS Limits

- **Free tier**: 10GB storage per project
- **10GB bandwidth** per month
- **No restrictions** on forks
- **Better performance** than GitHub LFS

## Alternative: Use GitLab CI/CD to Generate Dataset

Instead of storing the dataset, generate it in CI/CD:

```yaml
# .gitlab-ci.yml
generate_dataset:
  script:
    - pip install -r requirements.txt
    - python tests/synthetic_dataset.py 
      --source $SOURCE_DATASET_PATH 
      --output tests/synthetic_test_dataset 
      --num_train 800 --num_test 200
  artifacts:
    paths:
      - tests/synthetic_test_dataset/
    expire_in: 1 week
```

## Troubleshooting

### LFS files not uploading

```bash
# Check LFS configuration
git lfs env

# Verify remote URL
git remote -v

# Try explicit push
git lfs push gitlab main --all
```

### Authentication issues

```bash
# Use personal access token
git config credential.helper store
git push gitlab main
# Enter username and token when prompted
```

### Large files still in Git

```bash
# Migrate existing files to LFS
git lfs migrate import --include="*.jpg" --everything

# Force push (be careful!)
git push gitlab main --force
```

## Comparison: GitHub vs GitLab LFS

| Feature | GitHub | GitLab |
|---------|--------|--------|
| Free Storage | 1GB | 10GB |
| Forks Support | ❌ Blocked | ✅ Supported |
| Bandwidth | 1GB/month | 10GB/month |
| Performance | Good | Excellent |
| CI/CD Integration | ✅ | ✅ |

## Recommendation

For this project, **GitLab is the better choice** because:
1. ✅ 10x more free storage (10GB vs 1GB)
2. ✅ No fork restrictions
3. ✅ Better LFS performance
4. ✅ Can generate dataset in CI/CD

## Next Steps

1. Create GitLab account and repository
2. Add GitLab remote
3. Push code and LFS files
4. Update documentation with GitLab URLs
5. Optionally keep GitHub as mirror

