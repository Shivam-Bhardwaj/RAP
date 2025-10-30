# Git LFS Authentication Fix

The server is set up correctly, but Git LFS needs authentication configuration.

## Quick Fix

Run these commands to configure Git for no-auth LFS:

```bash
cd /home/ubuntu/RAP

# Configure credential helper
git config credential.helper 'store --file=/dev/null'

# Or use empty credentials
echo "http://66.94.123.205" > ~/.git-credentials
echo "" >> ~/.git-credentials  
echo "" >> ~/.git-credentials

# Or use Git credential helper with empty credentials
git config credential.helper store
printf "http://\n\n" | git credential approve

# Disable batch to avoid credential issues
git config lfs.batch false

# Try push again
git lfs push origin main --all
```

## Alternative: Update Server Config

If authentication continues to be an issue, update the server config to use basic auth:

```bash
# On server
ssh root@66.94.123.205

# Install htpasswd
apt-get install -y apache2-utils

# Create password file
htpasswd -c /etc/nginx/.htpasswd gitlfs
# Enter password when prompted

# Update config
cat > ~/lfs-config/config.yaml << EOF
listen: "127.0.0.1:8080"
public: "http://66.94.123.205:8080"
storage: "local"
storage_path: "/var/lfs"
auth:
  type: "basic"
  htpasswd_file: "/etc/nginx/.htpasswd"
EOF

# Restart service
systemctl restart lfs-server
```

Then configure Git with credentials:
```bash
git config credential.helper store
printf "http://gitlfs:YOUR_PASSWORD@66.94.123.205\n" | git credential approve
```

