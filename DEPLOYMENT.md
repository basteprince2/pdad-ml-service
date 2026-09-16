# ML Service - Production Deployment Guide

Production deployment guide for the FastAPI ML service on Ubuntu Linux VPS (tested with 1GB RAM Hostinger tier).

Companion document to `docs/DEPLOYMENT.md` (Laravel deployment). This guide covers the FastAPI ML service specifically.

## Deployment Overview

The FastAPI ML service runs as a **systemd service** behind an **Nginx reverse proxy**. This provides:

- Automatic startup on server boot
- Automatic restart on crash
- HTTPS termination at Nginx layer
- Isolated Python virtual environment
- Structured JSON logging with daily rotation

## Prerequisites

Server should already have:

- Ubuntu 22.04 LTS or 24.04 LTS
- Nginx installed and running
- SSL certificate configured (Let's Encrypt via Certbot)
- Non-root sudo user (referred to as `deploy` below)
- Firewall configured (UFW) with ports 80, 443, 22 open only

## Step 1 - Install Python 3.12 or 3.13

Ubuntu 22.04 ships with Python 3.10 by default. Ubuntu 24.04 ships with 3.12. If neither is available, install via deadsnakes PPA:

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev
python3.13 --version
```

**Do NOT use Python 3.14.** scikit-learn 1.6.1 wheels are not available for 3.14.

## Step 2 - Deploy the ml_service Directory

Assuming the Laravel application is already deployed to `/var/www/techsource-pdad/`:

```bash
cd /var/www/techsource-pdad
# ml_service/ should already exist from git pull

sudo chown -R deploy:www-data ml_service
cd ml_service
```

## Step 3 - Create Virtual Environment

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional performance boost (Linux only, not Windows)
pip install uvloop httptools
```

Verify:

```bash
pip list | grep -E "fastapi|scikit-learn|uvicorn"
```

Expected: fastapi, scikit-learn 1.6.1, uvicorn.

## Step 4 - Place Model Artifacts

Upload the 8 `.pkl` files to `/var/www/techsource-pdad/ml_service/artifacts/`.

Options:

- **scp** from your local machine:

```bash
  scp -r artifacts/*.pkl deploy@your-server:/var/www/techsource-pdad/ml_service/artifacts/
```

- **wget from cloud storage** (if artifacts are on Google Drive with direct links, or on GitHub Releases)
- **rsync** for large batches

Verify:

```bash
ls -lh /var/www/techsource-pdad/ml_service/artifacts/
```

Expected 8 files totaling ~10MB.

Set correct permissions:

```bash
sudo chown -R deploy:www-data /var/www/techsource-pdad/ml_service/artifacts
sudo chmod 640 /var/www/techsource-pdad/ml_service/artifacts/*.pkl
```

## Step 5 - Configure Environment Variables

```bash
cd /var/www/techsource-pdad/ml_service
cp .env.example .env
nano .env
```

Set production values:

```
ARTIFACTS_DIR=/var/www/techsource-pdad/ml_service/artifacts
ALLOWED_ORIGINS=https://your-domain.ph
ML_TIMEOUT_SECONDS=2.0
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
LOG_DIR=/var/www/techsource-pdad/ml_service/logs
```

Secure the file:

```bash
sudo chmod 640 .env
sudo chown deploy:www-data .env
```

## Step 6 - Verify Manual Startup

Before creating the systemd service, verify the app starts correctly:

```bash
cd /var/www/techsource-pdad/ml_service
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8001
```

In another terminal:

```bash
curl http://127.0.0.1:8001/health
```

Expected response:

```json
{
    "status": "ok",
    "predictor_loaded": true,
    "recommender_loaded": true,
    "rate_limit_requests": 100,
    "rate_limit_window_seconds": 3600
}
```

Stop with Ctrl+C. Proceed only if the manual startup succeeded.

## Step 7 - Create Systemd Service

Create the service file:

```bash
sudo nano /etc/systemd/system/ml-service.service
```

Paste:

```ini
[Unit]
Description=TechSource PDAD FastAPI ML Service
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=deploy
Group=www-data
WorkingDirectory=/var/www/techsource-pdad/ml_service
Environment="PATH=/var/www/techsource-pdad/ml_service/.venv/bin"
EnvironmentFile=/var/www/techsource-pdad/ml_service/.env
ExecStart=/var/www/techsource-pdad/ml_service/.venv/bin/uvicorn main:app \
    --host 127.0.0.1 \
    --port 8001 \
    --workers 1 \
    --log-level warning \
    --access-log \
    --no-server-header

Restart=on-failure
RestartSec=5s

StandardOutput=append:/var/www/techsource-pdad/ml_service/logs/systemd-stdout.log
StandardError=append:/var/www/techsource-pdad/ml_service/logs/systemd-stderr.log

# Security hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/www/techsource-pdad/ml_service/logs

[Install]
WantedBy=multi-user.target
```

**Note on workers:** The service uses `--workers 1` for 1GB RAM VPS. If deploying to 2GB+ VPS, change to `--workers 2` for higher throughput.

Reload systemd, enable, and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-service
sudo systemctl start ml-service
sudo systemctl status ml-service
```

Expected status: `active (running)`.

Verify:

```bash
curl http://127.0.0.1:8001/health
```

Should return the same health JSON as Step 6.

## Step 8 - Configure Nginx Reverse Proxy

The Laravel app should already be configured in Nginx. Add an internal proxy for the ML service.

Edit the Nginx site config:

```bash
sudo nano /etc/nginx/sites-available/techsource-pdad
```

Inside the `server { ... }` block, add:

```nginx
    # Internal proxy for FastAPI ML service
    # Only Laravel backend should access this via 127.0.0.1
    location /ml/ {
        # Deny external access - only allow internal Laravel calls
        allow 127.0.0.1;
        deny all;

        proxy_pass http://127.0.0.1:8001/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
```

**Security note:** The ML service is not exposed externally. All requests must originate from Laravel (which runs on the same server). Update Laravel's `.env`:

```
ML_SERVICE_URL=http://127.0.0.1:8001
```

(Not `https://your-domain.ph/ml/` because Laravel talks directly to the local FastAPI, bypassing Nginx.)

Test and reload Nginx:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## Step 9 - Verify End-to-End

From the Laravel application directory:

```bash
cd /var/www/techsource-pdad
php artisan tinker --execute="echo config('services.ml_service.url');"
```

Should output `http://127.0.0.1:8001`.

Test the E2E flow:

```bash
php artisan test:setup-recommender-data
php artisan test:recommender
```

Expected output shows the three-layer filter progression:

```
Total jobs input:                    4
After employment type filter:        3
After disability compatibility:      2
Returned count:                      2
```

## Log Management

Structured JSON logs are written to `/var/www/techsource-pdad/ml_service/logs/ml_service.jsonl` with automatic daily rotation (14-day retention).

Systemd logs are in:

- `/var/www/techsource-pdad/ml_service/logs/systemd-stdout.log`
- `/var/www/techsource-pdad/ml_service/logs/systemd-stderr.log`

View live logs:

```bash
tail -f /var/www/techsource-pdad/ml_service/logs/ml_service.jsonl
```

View systemd journal:

```bash
sudo journalctl -u ml-service -f
```

## Maintenance Commands

```bash
# Restart the service
sudo systemctl restart ml-service

# Stop the service
sudo systemctl stop ml-service

# View status
sudo systemctl status ml-service

# View recent logs
sudo journalctl -u ml-service -n 100 --no-pager

# Reload after code changes (git pull, dependency updates)
cd /var/www/techsource-pdad/ml_service
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart ml-service
```

## Updating Model Artifacts

When retraining produces new `.pkl` files:

```bash
# 1. Upload new artifacts
scp -r new-artifacts/*.pkl deploy@your-server:/tmp/new-artifacts/

# 2. Stop the service
sudo systemctl stop ml-service

# 3. Backup existing artifacts
cd /var/www/techsource-pdad/ml_service/artifacts
mkdir -p ../artifacts-backup-$(date +%Y%m%d)
cp *.pkl ../artifacts-backup-$(date +%Y%m%d)/

# 4. Replace artifacts
cp /tmp/new-artifacts/*.pkl .

# 5. Update permissions
sudo chown deploy:www-data *.pkl
sudo chmod 640 *.pkl

# 6. Update artifacts/manifest.json with new version metadata

# 7. Restart service
sudo systemctl start ml-service
sudo systemctl status ml-service

# 8. Verify
curl http://127.0.0.1:8001/health
```

## Troubleshooting

### Service fails to start

Check status and logs:

```bash
sudo systemctl status ml-service
sudo journalctl -u ml-service -n 50 --no-pager
tail -50 /var/www/techsource-pdad/ml_service/logs/systemd-stderr.log
```

Common causes:

1. **Missing `.pkl` artifacts** - Health endpoint returns `predictor_loaded: false`. Check `ls -lh artifacts/`.
2. **Wrong Python version** - Recreate venv with `python3.13 -m venv .venv`.
3. **Permission denied on logs** - `sudo chown -R deploy:www-data logs/`.
4. **Port already in use** - Check `sudo lsof -i :8001` and stop conflicting process.

### 502 Bad Gateway from Nginx

FastAPI is down. Check `sudo systemctl status ml-service` and restart if needed.

### Slow predictions

Verify no swap thrashing (RAM exhaustion):

```bash
free -h
top
```

If RAM near limit, either upgrade VPS tier or reduce workers to 1 (if not already).

### Rate limit hits triggered unexpectedly

Check the rate limit configuration in `.env`:

```
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

Rate limiting is per-IP. Since Laravel is on the same server, all requests appear from 127.0.0.1. Verify localhost is whitelisted in `main.py` rate limit config.

## Rollback Procedure

If a deployment breaks the service:

```bash
# 1. Stop the service
sudo systemctl stop ml-service

# 2. Restore previous code
cd /var/www/techsource-pdad
git log --oneline -5
git checkout <previous-working-commit>

# 3. Reinstall dependencies (if requirements.txt changed)
cd ml_service
source .venv/bin/activate
pip install -r requirements.txt

# 4. Restore previous artifacts (if updated)
cp ../artifacts-backup-YYYYMMDD/*.pkl artifacts/

# 5. Restart
sudo systemctl start ml-service
sudo systemctl status ml-service
```

## Related Documentation

- `ml_service/README.md` - Development setup and local testing
- `docs/DEPLOYMENT.md` - Laravel application deployment
- `docs/MODEL_CARD.md` - RF predictor design and metrics
- `docs/RECOMMENDER_CARD.md` - Recommender architecture and three-layer filter
