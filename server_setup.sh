#!/bin/bash
# Git LFS Server Setup Script
# Run this on your server: root@66.94.123.205

set -e

echo "=== Git LFS Server Setup ==="
echo "Server: 66.94.123.205"
echo ""

# Step 1: Update system
echo "Step 1: Updating system..."
apt-get update && apt-get upgrade -y

# Step 2: Install dependencies
echo "Step 2: Installing dependencies..."
apt-get install -y curl wget git vim ufw nginx

# Step 3: Configure firewall
echo "Step 3: Configuring firewall..."
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable
ufw status

# Step 4: Install Go
echo "Step 4: Installing Go..."
wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Verify Go installation
go version

# Step 5: Create storage directory
echo "Step 5: Creating storage directory..."
mkdir -p /var/lfs
chmod 755 /var/lfs
df -h /var/lfs

# Step 6: Install Git LFS Server
echo "Step 6: Installing Git LFS Server..."
go install github.com/git-lfs/lfs-test-server@latest

# Verify installation
ls -lh ~/go/bin/lfs-test-server

# Step 7: Create configuration
echo "Step 7: Creating configuration..."
mkdir -p ~/lfs-config
cat > ~/lfs-config/config.yaml << EOF
listen: "127.0.0.1:8080"
public: "http://66.94.123.205:8080"
storage: "local"
storage_path: "/var/lfs"
auth:
  type: "none"
EOF

cat ~/lfs-config/config.yaml

# Step 8: Create systemd service
echo "Step 8: Creating systemd service..."
cat > /etc/systemd/system/lfs-server.service << EOF
[Unit]
Description=Git LFS Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/root/go/bin/lfs-test-server --config /root/lfs-config/config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable lfs-server
systemctl start lfs-server

# Check status
sleep 2
systemctl status lfs-server --no-pager | head -15

# Step 9: Configure Nginx
echo "Step 9: Configuring Nginx..."
cat > /etc/nginx/sites-available/lfs << 'NGINXEOF'
server {
    listen 80;
    server_name 66.94.123.205;
    
    client_max_body_size 5G;
    client_body_timeout 300s;
    
    access_log /var/log/nginx/lfs-access.log;
    error_log /var/log/nginx/lfs-error.log;
    
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/lfs /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
nginx -t

# Reload Nginx
systemctl reload nginx
systemctl status nginx --no-pager | head -5

# Step 10: Test LFS server
echo ""
echo "Step 10: Testing LFS server..."
echo "Testing locally..."
curl -s http://localhost:8080/info/lfs || echo "Local test failed - server may need a moment to start"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Test from local machine: curl http://66.94.123.205/info/lfs"
echo "2. Configure Git: git config lfs.url http://66.94.123.205/info/lfs"
echo "3. Push LFS files: git lfs push origin main --all"
echo ""
echo "Check service status: systemctl status lfs-server"
echo "Check logs: journalctl -u lfs-server -f"
echo "Check storage: du -sh /var/lfs"

