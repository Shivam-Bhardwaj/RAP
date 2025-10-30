#!/bin/bash
# Self-Hosted Git LFS Setup Script

set -e

echo "=== Self-Hosted Git LFS Setup ==="
echo ""

# Check if domain provided
if [ -z "$1" ]; then
    echo "Usage: $0 <domain> [storage_path]"
    echo "Example: $0 lfs.yourdomain.com /var/lfs"
    exit 1
fi

SERVER_DOMAIN="$1"
STORAGE_PATH="${2:-/var/lfs}"

echo "Configuration:"
echo "  Domain: $SERVER_DOMAIN"
echo "  Storage: $STORAGE_PATH"
echo ""

# Check if running as root for some operations
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Don't run as root. Some steps require sudo."
    exit 1
fi

# Install dependencies
echo "Step 1: Installing dependencies..."
sudo apt-get update
sudo apt-get install -y golang-go nginx certbot python3-certbot-nginx || true

# Create storage directory
echo "Step 2: Creating storage directory..."
sudo mkdir -p "$STORAGE_PATH"
sudo chown $USER:$USER "$STORAGE_PATH"

# Install Git LFS Server
echo "Step 3: Installing Git LFS Server..."
if ! command -v go &> /dev/null; then
    echo "Go not found. Installing..."
    sudo apt-get install -y golang-go
fi

go install github.com/git-lfs/lfs-test-server@latest || {
    echo "Failed to install. Trying alternative method..."
    git clone https://github.com/git-lfs/lfs-test-server.git /tmp/lfs-server
    cd /tmp/lfs-server
    go build -o lfs-server
    sudo mv lfs-server /usr/local/bin/lfs-test-server
}

# Create config
echo "Step 4: Creating configuration..."
cat > ~/lfs-config.yaml << EOF
listen: "127.0.0.1:8080"
public: "https://$SERVER_DOMAIN"
storage: "local"
storage_path: "$STORAGE_PATH"
auth:
  type: "none"
EOF

# Create systemd service
echo "Step 5: Creating systemd service..."
LFS_BIN=$(which lfs-test-server || echo "$HOME/go/bin/lfs-test-server")
sudo tee /etc/systemd/system/lfs-server.service > /dev/null << EOF
[Unit]
Description=Git LFS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
ExecStart=$LFS_BIN --config $HOME/lfs-config.yaml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable lfs-server
sudo systemctl start lfs-server

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Configure DNS: Point $SERVER_DOMAIN to this server's IP"
echo "2. Setup SSL: sudo certbot --nginx -d $SERVER_DOMAIN"
echo "3. Configure Git: git config lfs.url https://$SERVER_DOMAIN/info/lfs"
echo "4. Push LFS: git lfs push origin main --all"
echo ""
echo "Server status:"
sudo systemctl status lfs-server --no-pager | head -5
