#!/bin/bash
# Quick setup script for GitLab LFS

set -e

echo "=== GitLab LFS Setup ==="
echo ""

# Check if GitLab username/repo provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <gitlab_username> <repo_name>"
    echo "Example: $0 johndoe RAP-ID"
    exit 1
fi

GITLAB_USER="$1"
GITLAB_REPO="$2"
GITLAB_URL="https://gitlab.com/${GITLAB_USER}/${GITLAB_REPO}.git"

echo "GitLab User: $GITLAB_USER"
echo "Repository: $GITLAB_REPO"
echo "URL: $GITLAB_URL"
echo ""

# Check if remote exists
if git remote | grep -q "^gitlab$"; then
    echo "GitLab remote already exists, updating..."
    git remote set-url gitlab "$GITLAB_URL"
else
    echo "Adding GitLab remote..."
    git remote add gitlab "$GITLAB_URL"
fi

# Configure LFS to use GitLab
echo "Configuring Git LFS for GitLab..."
git config lfs.url "${GITLAB_URL}/info/lfs"

# Show current remotes
echo ""
echo "Current remotes:"
git remote -v

# Show LFS config
echo ""
echo "LFS configuration:"
git lfs env | grep -E "Endpoint|Remote"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Push code: git push gitlab main"
echo "2. Push LFS files: git lfs push gitlab main --all"
echo ""
echo "Or push everything at once:"
echo "   git push gitlab main && git lfs push gitlab main --all"
