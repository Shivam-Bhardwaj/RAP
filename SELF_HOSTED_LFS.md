# Self-Hosting Git LFS Storage

## Overview

Self-hosting Git LFS gives you:
- **Full control** over storage and bandwidth
- **No size limits** (only limited by your server)
- **No fork restrictions**
- **Cost-effective** for large datasets
- **Privacy** - data stays on your infrastructure

## Self-Hosting Options

### Option 1: Git LFS Server (Official)

The official Git LFS server implementation.

#### Installation

```bash
# Install Go (required)
sudo apt-get update
sudo apt-get install -y golang-go

# Install Git LFS Server
go install github.com/git-lfs/lfs-test-server@latest

# Or clone and build
git clone https://github.com/git-lfs/lfs-test-server.git
cd lfs-test-server
go build -o lfs-server
```

#### Configuration

Create `config.yaml`:

```yaml
# config.yaml
listen: "0.0.0.0:8080"
public: "http://your-server.com:8080"
storage: "local"
storage_path: "/var/lfs"
auth:
  type: "none"  # or "basic", "jwt"
  htpasswd_file: "/path/to/.htpasswd"
```

#### Run Server

```bash
# Simple run
./lfs-server

# With config
./lfs-server --config config.yaml

# As systemd service (recommended)
sudo systemctl enable lfs-server
sudo systemctl start lfs-server
```

#### Configure Git Client

```bash
git config lfs.url http://your-server.com:8080/info/lfs
git lfs push origin main --all
```

### Option 2: MinIO (S3-Compatible Object Storage)

MinIO is excellent for self-hosted LFS storage using S3 protocol.

#### Installation

```bash
# Download MinIO
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/

# Create data directory
sudo mkdir -p /var/minio/data
sudo chown $USER:$USER /var/minio/data
```

#### Run MinIO

```bash
# Set access keys
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin123

# Run MinIO
minio server /var/minio/data --console-address ":9001"
```

#### Configure Git LFS for S3

```bash
# Install git-lfs-s3 (third-party tool)
# Or use git-lfs with S3 backend

# Configure environment
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin123
export AWS_ENDPOINT_URL=http://your-server.com:9000

# Use S3 backend
git config lfs.url "s3://your-bucket-name/info/lfs"
```

### Option 3: Nginx + Simple File Storage

Simple HTTP server with file storage.

#### Setup Nginx

```nginx
# /etc/nginx/sites-available/lfs
server {
    listen 8080;
    server_name your-server.com;
    
    root /var/lfs;
    autoindex on;
    
    location / {
        # Handle LFS API
        proxy_pass http://localhost:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # Direct file access
    location /objects/ {
        alias /var/lfs/objects/;
        sendfile on;
    }
}
```

#### Simple LFS Server Script

```python
#!/usr/bin/env python3
# simple_lfs_server.py
from flask import Flask, request, send_file, jsonify
import os
import hashlib

app = Flask(__name__)
STORAGE_ROOT = "/var/lfs"

@app.route('/<repo>/info/lfs/objects/batch', methods=['POST'])
def batch_api():
    data = request.json
    objects = []
    
    for obj in data.get('objects', []):
        oid = obj['oid']
        size = obj['size']
        
        # Store file info
        filepath = os.path.join(STORAGE_ROOT, oid[:2], oid[2:4], oid)
        
        objects.append({
            'oid': oid,
            'size': size,
            'authenticated': True,
            'actions': {
                'upload': {
                    'href': f'http://your-server.com/objects/{oid}',
                    'header': {}
                },
                'download': {
                    'href': f'http://your-server.com/objects/{oid}',
                    'header': {}
                }
            }
        })
    
    return jsonify({'objects': objects})

@app.route('/objects/<oid>', methods=['PUT', 'GET'])
def object_storage(oid):
    filepath = os.path.join(STORAGE_ROOT, oid[:2], oid[2:4], oid)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if request.method == 'PUT':
        request.data.save(filepath)
        return '', 200
    elif request.method == 'GET':
        return send_file(filepath)

if __name__ == '__main__':
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    app.run(host='0.0.0.0', port=8081)
```

### Option 4: Docker-Based Solutions

#### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  lfs-server:
    image: gitlab/gitlab-lfs-server:latest
    ports:
      - "8080:8080"
    volumes:
      - lfs-data:/var/lfs
    environment:
      - LFS_STORAGE_PATH=/var/lfs
      - LFS_LISTEN=:8080
    restart: unless-stopped

  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin123
    command: server /data --console-address ":9001"
    restart: unless-stopped

volumes:
  lfs-data:
  minio-data:
```

Run:
```bash
docker-compose up -d
```

## Complete Self-Hosted Setup Guide

### Step 1: Choose Your Server

**For small projects (< 100GB):**
- Git LFS Test Server (simple, lightweight)

**For medium projects (100GB - 1TB):**
- MinIO (S3-compatible, scalable)

**For large projects (> 1TB):**
- MinIO with distributed setup
- Or dedicated object storage (Ceph, Swift)

### Step 2: Server Setup

#### Basic Server Requirements

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y git nginx certbot python3-certbot-nginx

# Create storage directory
sudo mkdir -p /var/lfs
sudo chown $USER:$USER /var/lfs
```

#### Option A: Git LFS Test Server

```bash
# Install Go
sudo apt-get install -y golang-go

# Install lfs-test-server
go install github.com/git-lfs/lfs-test-server@latest

# Create config
cat > ~/lfs-config.yaml << EOF
listen: "127.0.0.1:8080"
public: "https://lfs.yourdomain.com"
storage: "local"
storage_path: "/var/lfs"
auth:
  type: "none"
EOF

# Run as service
sudo tee /etc/systemd/system/lfs-server.service << EOF
[Unit]
Description=Git LFS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=/home/$USER/go/bin/lfs-test-server --config /home/$USER/lfs-config.yaml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable lfs-server
sudo systemctl start lfs-server
```

#### Option B: MinIO Setup

```bash
# Download MinIO
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
sudo mv minio /usr/local/bin/

# Create service
sudo tee /etc/systemd/system/minio.service << EOF
[Unit]
Description=MinIO Object Storage
After=network.target

[Service]
Type=simple
User=$USER
Environment="MINIO_ROOT_USER=minioadmin"
Environment="MINIO_ROOT_PASSWORD=CHANGE_THIS_PASSWORD"
ExecStart=/usr/local/bin/minio server /var/minio/data --console-address ":9001"
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable minio
sudo systemctl start minio
```

### Step 3: Configure Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/lfs
server {
    listen 80;
    server_name lfs.yourdomain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name lfs.yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/lfs.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/lfs.yourdomain.com/privkey.pem;
    
    client_max_body_size 5G;  # Allow large file uploads
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable SSL:
```bash
sudo certbot --nginx -d lfs.yourdomain.com
```

### Step 4: Configure Git Client

```bash
cd /home/ubuntu/RAP

# Set LFS URL to your server
git config lfs.url https://lfs.yourdomain.com/info/lfs

# Verify
git lfs env

# Push LFS files
git lfs push origin main --all
```

### Step 5: Security Considerations

#### Authentication

**Option 1: Basic Auth (Simple)**

```bash
# Install htpasswd
sudo apt-get install -y apache2-utils

# Create password file
htpasswd -c /etc/nginx/.htpasswd username

# Update Nginx config
location / {
    auth_basic "Git LFS Storage";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://127.0.0.1:8080;
}
```

**Option 2: SSH Key Authentication**

Configure SSH access and use SSH URLs:
```bash
git config lfs.url ssh://git@your-server.com:/var/lfs
```

**Option 3: Token-Based (JWT)**

Use Git LFS server with JWT authentication.

## Automated Setup Script

Create a complete setup script:

```bash
#!/bin/bash
# setup_self_hosted_lfs.sh

set -e

SERVER_DOMAIN="${1:-lfs.yourdomain.com}"
STORAGE_PATH="${2:-/var/lfs}"

echo "=== Self-Hosted Git LFS Setup ==="
echo "Domain: $SERVER_DOMAIN"
echo "Storage: $STORAGE_PATH"
echo ""

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y golang-go nginx certbot python3-certbot-nginx

# Create storage directory
echo "Creating storage directory..."
sudo mkdir -p "$STORAGE_PATH"
sudo chown $USER:$USER "$STORAGE_PATH"

# Install Git LFS Server
echo "Installing Git LFS Server..."
go install github.com/git-lfs/lfs-test-server@latest

# Create config
cat > ~/lfs-config.yaml << EOF
listen: "127.0.0.1:8080"
public: "https://$SERVER_DOMAIN"
storage: "local"
storage_path: "$STORAGE_PATH"
auth:
  type: "none"
EOF

# Create systemd service
sudo tee /etc/systemd/system/lfs-server.service << EOF
[Unit]
Description=Git LFS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=$HOME/go/bin/lfs-test-server --config $HOME/lfs-config.yaml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable lfs-server
sudo systemctl start lfs-server

echo "Setup complete!"
echo "Configure DNS to point $SERVER_DOMAIN to this server"
echo "Then run: sudo certbot --nginx -d $SERVER_DOMAIN"
```

## Testing Your Setup

```bash
# Test LFS server
curl https://lfs.yourdomain.com/info/lfs

# Test from Git client
cd /home/ubuntu/RAP
git config lfs.url https://lfs.yourdomain.com/info/lfs
git lfs push origin main --all
```

## Monitoring and Maintenance

### Monitor Storage Usage

```bash
# Check storage size
du -sh /var/lfs

# Count files
find /var/lfs -type f | wc -l

# Monitor disk space
df -h /var/lfs
```

### Backup Strategy

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/lfs"
SOURCE_DIR="/var/lfs"
DATE=$(date +%Y%m%d)

# Create backup
tar -czf "$BACKUP_DIR/lfs-backup-$DATE.tar.gz" "$SOURCE_DIR"

# Keep only last 7 days
find "$BACKUP_DIR" -name "lfs-backup-*.tar.gz" -mtime +7 -delete
```

### Automated Cleanup

```bash
# Cleanup old/unused LFS objects (requires tracking)
# Keep only objects referenced in Git history
```

## Cost Comparison

| Solution | Setup Time | Monthly Cost | Storage Limit |
|----------|-----------|--------------|---------------|
| GitHub LFS | 0 min | $0-5 | 1GB free |
| GitLab LFS | 0 min | $0 | 10GB free |
| Self-Hosted (VPS) | 30 min | $5-20 | Unlimited |
| Self-Hosted (Dedicated) | 1 hour | $50-200 | Unlimited |

## Recommended Setup for This Project

For RAP-ID with ~235MB synthetic dataset:

**Option 1: Simple VPS (Recommended)**
- Provider: DigitalOcean, Linode, Hetzner
- Specs: 1 CPU, 2GB RAM, 20GB SSD
- Cost: ~$5/month
- Setup: Git LFS Test Server + Nginx

**Option 2: Shared Hosting**
- If you already have hosting
- Use existing server
- Cost: $0 (included)

**Option 3: Cloud Storage**
- AWS S3, Google Cloud Storage
- Pay per GB
- Cost: ~$0.01-0.05/GB/month

## Quick Start Commands

```bash
# 1. Install Git LFS Server
go install github.com/git-lfs/lfs-test-server@latest

# 2. Run server
~/go/bin/lfs-test-server --config lfs-config.yaml

# 3. Configure Git
git config lfs.url http://your-server:8080/info/lfs

# 4. Push LFS files
git lfs push origin main --all
```

## Next Steps

1. Choose your hosting (VPS, cloud, or existing server)
2. Run setup script or follow manual setup
3. Configure DNS and SSL
4. Update Git config to use self-hosted LFS
5. Push your 859 images!

