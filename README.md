# ML Service - PDAD Employment Type Predictor + Job Recommender

FastAPI-based ML service for the TechSource PDAD Information System. Provides employment type prediction (Random Forest) and job recommendation (TF-IDF + Cosine Similarity with three-layer disability compatibility filter) via HTTP endpoints consumed by the Laravel application.

## Architecture

- **FastAPI** application (`main.py`)
- **Random Forest** predictor loaded from `.pkl` artifacts at startup
- **TF-IDF Cosine Similarity** recommender with request-time vectorization
- **Three-layer disability compatibility filter** in the recommender
- **Structured JSON logging** to `logs/ml_service.jsonl`
- **Rate limiting** (100 requests per 3600s per IP)
- **Timeout handling** (2s default, configurable)

## Endpoints

| Method | Path                       | Description                                       |
| ------ | -------------------------- | ------------------------------------------------- |
| GET    | `/health`                  | Health check, returns model load status           |
| POST   | `/predict/employment-type` | Predicts employment type from 11 profile features |
| POST   | `/recommend/jobs`          | Ranks jobs using TF-IDF + three-layer filter      |

Interactive API docs: `http://<host>:8001/docs` (Swagger UI)

## Directory Structure

```
ml_service/
├── main.py                 # FastAPI app + endpoint handlers
├── schemas.py              # Pydantic request/response models
├── config.py               # Environment configuration
├── logging_config.py       # Structured JSON logging setup
├── requirements.txt        # Pinned Python dependencies
├── .env.example            # Environment variable template
├── models/
│   ├── predictor.py        # Random Forest predictor class
│   └── recommender.py      # Three-layer TF-IDF recommender class
├── artifacts/              # ML model artifacts (.pkl files, gitignored)
├── training/               # Training scripts + notebooks reference
├── tests/                  # Endpoint tests
└── logs/                   # JSON logs (gitignored, rotated daily)
```

## Local Development Setup

### Prerequisites

- **Python 3.12 or 3.13** (NOT 3.14 — scikit-learn 1.6.1 wheels not available)
- pip
- Virtual environment support (`python -m venv`)

### Steps

```powershell
# 1. Clone repo and navigate to ml_service
cd ml_service

# 2. Create virtual environment
python -m venv .venv

# 3. Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
# source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template
Copy-Item .env.example .env

# 6. Place model artifacts in artifacts/ folder
#    Required files (obtain from team ML lead):
#    - rf_model.pkl
#    - training_columns.pkl
#    - model_classes.pkl
#    - raw_features.pkl
#    - categorical_cols.pkl
#    - tfidf_vectorizer.pkl
#    - job_tfidf_matrix.pkl
#    - job_ids.pkl

# 7. Run development server
uvicorn main:app --reload --port 8001
```

Verify: `curl http://localhost:8001/health` should return `{"status":"ok","predictor_loaded":true,"recommender_loaded":true,...}`

## Running Modes

### Development mode (local)

```bash
uvicorn main:app --reload --port 8001
```

- `--reload` — auto-reload on file changes
- Verbose INFO logs
- Single worker
- Suitable for local testing only

### Production mode (Linux VPS, 1GB RAM tier)

```bash
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 1 \
    --log-level warning \
    --access-log \
    --no-server-header
```

- `--host 0.0.0.0` — accept connections from any interface (required for external access via reverse proxy)
- `--port 8001` — same port as development for consistency
- `--workers 1` — single worker (~200MB RAM). Conservative for 1GB VPS shared with Laravel + MySQL + Nginx
- `--log-level warning` — hide INFO logs, reduce disk usage
- `--access-log` — retain access logs for audit trail
- `--no-server-header` — do not expose uvicorn version in response headers (security best practice)

**Do NOT use `--reload` in production.** It watches the filesystem and adds significant overhead. It is also unnecessary because production deployments do not modify code at runtime.

### Higher-throughput production (2GB+ RAM tier)

```bash
uvicorn main:app \
    --host 0.0.0.0 \
    --port 8001 \
    --workers 2 \
    --log-level warning \
    --access-log \
    --no-server-header
```

- `--workers 2` — ~2x throughput at 2x memory cost. Requires at least 2GB RAM to comfortably fit alongside Laravel and MySQL.

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable              | Default       | Description                                    |
| --------------------- | ------------- | ---------------------------------------------- |
| `ARTIFACTS_DIR`       | `./artifacts` | Path to `.pkl` model artifacts                 |
| `ALLOWED_ORIGINS`     | `*`           | CORS allowed origins (comma-separated in prod) |
| `ML_TIMEOUT_SECONDS`  | `2.0`         | Request timeout before returning 504           |
| `RATE_LIMIT_REQUESTS` | `100`         | Max requests per window per IP                 |
| `RATE_LIMIT_WINDOW`   | `3600`        | Rate limit window in seconds                   |
| `LOG_DIR`             | `./logs`      | Directory for JSON logs                        |

## Model Artifacts

Model `.pkl` files are **not committed to Git** (see `.gitignore`). They must be placed manually in the `artifacts/` folder. Total size: ~10MB.

Required artifacts:

| File                   | Size    | Purpose                              |
| ---------------------- | ------- | ------------------------------------ |
| `rf_model.pkl`         | ~9.5 MB | Trained RandomForestClassifier       |
| `training_columns.pkl` | ~4 KB   | 108 encoded column names for reindex |
| `model_classes.pkl`    | ~105 B  | 7 employment class labels            |
| `raw_features.pkl`     | ~205 B  | 11 raw feature names                 |
| `categorical_cols.pkl` | ~199 B  | 10 categorical column names          |
| `tfidf_vectorizer.pkl` | ~3 KB   | TF-IDF vectorizer for recommender    |
| `job_tfidf_matrix.pkl` | ~2 KB   | Pre-computed job TF-IDF matrix       |
| `job_ids.pkl`          | ~22 B   | Job ID mapping for recommender       |

Model card and recommender card documentation: `docs/MODEL_CARD.md`, `docs/RECOMMENDER_CARD.md`.

## Testing

End-to-end integration testing is performed via Laravel Artisan commands:

```powershell
# Setup minimal test fixture (1 PWD profile + 4 diverse jobs)
php artisan test:setup-recommender-data

# Run end-to-end recommender test with three-layer filter metadata
php artisan test:recommender
```

Expected output shows the three-layer filter progression:

```
Total jobs input:                    4
After employment type filter:        3
After disability compatibility:      2
Returned count:                      2
```

## Related Documentation

- `docs/MODEL_CARD.md` — Random Forest predictor metrics and design rationale
- `docs/RECOMMENDER_CARD.md` — Recommender architecture and three-layer filter details
- `ml_service/DEPLOYMENT.md` — Production deployment guide (Ubuntu VPS + Nginx + systemd)
- `docs/DEPLOYMENT.md` — Laravel application deployment guide
