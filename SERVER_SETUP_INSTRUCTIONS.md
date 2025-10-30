# Quick Setup Instructions for Your Server

## Your Server Details
- **IP:** 66.94.123.205
- **User:** root
- **Password:** iouiouiou

## Option 1: Copy and Run Script (Easiest)

1. **Copy the setup script to your server:**

```bash
# On your local machine, copy the script
scp server_setup.sh root@66.94.123.205:/root/

# Or manually copy the contents of server_setup.sh
```

2. **SSH into your server:**

```bash
ssh root@66.94.123.205
# Enter password: iouiouiou
```

3. **Make executable and run:**

```bash
chmod +x server_setup.sh
./server_setup.sh
```

## Option 2: Run Commands Manually

If you prefer to run commands step by step, SSH into your server and run:

```bash
# 1. Update and install
apt-get update && apt-get upgrade -y
apt-get install -y curl wget git vim ufw nginx

# 2. Firewall
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable

# 3. Install Go
wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
rm -rf /usr/local/go
tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# 4. Install LFS Server
go install github.com/git-lfs/lfs-test-server@latest

# 5. Create storage and config
mkdir -p /var/lfs
mkdir -p ~/lfs-config
cat > ~/lfs-config/config.yaml << EOF
listen: "127.0.0.1:8080"
public: "http://66.94.123.205:8080"
storage: "local"
storage_path: "/var/lfs"
auth:
  type: "none"
EOF

# 6. Create systemd service
cat > /etc/systemd/system/lfs-server.service << EOF
[Unit]
Description=Git LFS Server
After=network.target

[Service]
Type=simple
User=root
ExecStart=/root/go/bin/lfs-test-server --config /root/lfs-config/config.yaml
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable lfs-server
systemctl start lfs-server

# 7. Configure Nginx
cat > /etc/nginx/sites-available/lfs << EOF
server {
    listen 80;
    server_name 66.94.123.205;
    client_max_body_size 5G;
    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host \$host;
    }
}
EOF

ln -sf /etc/nginx/sites-available/lfs /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# 8. Test
curl http://localhost:8080/info/lfs
curl http://66.94.123.205/info/lfs
```

## After Setup

Once setup is complete on your server, configure Git on your local machine:

```bash
cd /home/ubuntu/RAP

# Configure LFS URL
git config lfs.url http://66.94.123.205/info/lfs

# Disable locking (not supported by simple server)
git config lfs.http://66.94.123.205/info/lfs.locksverify false

# Verify
git lfs env

# Test connection
curl http://66.94.123.205/info/lfs

# Push LFS files
git lfs push origin main --all
```

## Troubleshooting

If connection fails:

```bash
# On server - check service
systemctl status lfs-server

# Check logs
journalctl -u lfs-server -n 50

# Test locally on server
curl http://localhost:8080/info/lfs

# Check Nginx
systemctl status nginx
nginx -t
```

