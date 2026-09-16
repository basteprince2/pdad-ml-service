import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from config import (
    ARTIFACTS_DIR,
    ALLOWED_ORIGINS,
    ML_TIMEOUT_SECONDS,
    ML_LOG_DIR,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from models.predictor import EmploymentTypePredictor
from models.validator import VocabularyValidator
from models.recommender import JobRecommender
from schemas import (
    PredictionRequest,
    PredictionResponse,
    RecommendationRequest,
    RecommendationResponse,
)


predictor: EmploymentTypePredictor | None = None
validator: VocabularyValidator | None = None
recommender: JobRecommender | None = None
ml_logger = logging.getLogger("pdad_ml_service")

# In-memory rate limit store:
# key = client_ip:endpoint
# value = list of request timestamps
rate_limit_store: dict[str, list[float]] = {}


def setup_structured_logging() -> None:
    """
    Configure JSONL structured logging for ML endpoint calls.

    Logs are written to:
      - console
      - ./logs/ml_service.jsonl by default

    Raw input data is NOT logged. Only a SHA-256 input hash is stored.
    """
    os.makedirs(ML_LOG_DIR, exist_ok=True)

    ml_logger.setLevel(logging.INFO)
    ml_logger.handlers.clear()
    ml_logger.propagate = False

    log_file_path = os.path.join(ML_LOG_DIR, "ml_service.jsonl")

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    ml_logger.addHandler(file_handler)
    ml_logger.addHandler(stream_handler)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_payload(payload: dict) -> str:
    """
    Computes a deterministic SHA-256 hash of the request payload.

    This avoids storing sensitive raw PWD/profile fields in logs while still
    allowing request tracing and audit comparison.
    """
    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_ml_log(event: dict) -> None:
    ml_logger.info(
        json.dumps(
            event,
            ensure_ascii=False,
            default=str,
        )
    )


def get_client_ip(request: Request) -> str:
    """
    Gets client IP.

    In local development this is usually 127.0.0.1.
    Behind a reverse proxy, production can pass X-Forwarded-For.
    """
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


def enforce_rate_limit(request: Request, endpoint_name: str) -> None:
    """
    Simple in-memory IP + endpoint rate limiter.

    Default:
      100 requests per 3600 seconds per IP per endpoint.

    If exceeded:
      returns HTTP 429 Too Many Requests.
    """
    now = time.time()
    client_ip = get_client_ip(request)
    key = f"{client_ip}:{endpoint_name}"

    request_times = rate_limit_store.get(key, [])

    # Keep only timestamps inside the active window
    active_request_times = [
        request_time
        for request_time in request_times
        if now - request_time < RATE_LIMIT_WINDOW_SECONDS
    ]

    if len(active_request_times) >= RATE_LIMIT_REQUESTS:
        oldest_request_time = min(active_request_times)
        retry_after_seconds = int(
            RATE_LIMIT_WINDOW_SECONDS - (now - oldest_request_time)
        )

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "WARNING",
                "event": "ml_rate_limit_exceeded",
                "endpoint": endpoint_name,
                "client_ip": client_ip,
                "status": "rate_limited",
                "limit": RATE_LIMIT_REQUESTS,
                "window_seconds": RATE_LIMIT_WINDOW_SECONDS,
                "retry_after_seconds": retry_after_seconds,
            }
        )

        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded. Maximum {RATE_LIMIT_REQUESTS} requests "
                f"per {RATE_LIMIT_WINDOW_SECONDS} seconds."
            ),
            headers={"Retry-After": str(retry_after_seconds)},
        )

    active_request_times.append(now)
    rate_limit_store[key] = active_request_times


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, recommender, validator

    setup_structured_logging()

    predictor = EmploymentTypePredictor(ARTIFACTS_DIR)
    recommender = JobRecommender(ARTIFACTS_DIR)
    validator = VocabularyValidator(
        training_columns=predictor.training_columns,
        categorical_cols=predictor.categorical_cols,
    )

    print(f"[startup] Models loaded from {ARTIFACTS_DIR}")
    print(f"[startup] ML endpoint timeout set to {ML_TIMEOUT_SECONDS}s")
    print(f"[startup] Structured logs writing to {ML_LOG_DIR}/ml_service.jsonl")
    print(
        "[startup] Rate limit set to "
        f"{RATE_LIMIT_REQUESTS} requests per "
        f"{RATE_LIMIT_WINDOW_SECONDS}s per IP per endpoint"
    )

    yield

    print("[shutdown] ML service stopping")


app = FastAPI(
    title="PDAD ML Service",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


async def run_with_timeout(function, *args, endpoint_name: str):
    """
    Runs blocking ML inference in a worker thread and enforces timeout.

    If the ML call exceeds ML_TIMEOUT_SECONDS, the API returns 504
    instead of hanging indefinitely.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(function, *args),
            timeout=ML_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"{endpoint_name} timed out after "
                f"{ML_TIMEOUT_SECONDS} seconds"
            ),
        )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "predictor_loaded": predictor is not None,
        "recommender_loaded": recommender is not None,
        "rate_limit_requests": RATE_LIMIT_REQUESTS,
        "rate_limit_window_seconds": RATE_LIMIT_WINDOW_SECONDS,
    }


@app.post(
    "/predict/employment-type",
    response_model=PredictionResponse,
)
async def predict_employment_type(request: Request, req: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not loaded")

    endpoint_name = "/predict/employment-type"
    enforce_rate_limit(request, endpoint_name)

    request_id = str(uuid.uuid4())
    payload = req.model_dump()

    if validator is not None:
        validator.validate(payload)

    input_hash = hash_payload(payload)
    start_time = time.perf_counter()

    try:
        result = await run_with_timeout(
            predictor.predict,
            payload,
            endpoint_name=endpoint_name,
        )

        latency_seconds = time.perf_counter() - start_time

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "INFO",
                "event": "ml_prediction_completed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "success",
                "result": {
                    "predicted_type": result.get("predicted_type"),
                    "confidence": result.get("confidence"),
                },
            }
        )

        return result

    except HTTPException as error:
        latency_seconds = time.perf_counter() - start_time

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "ERROR",
                "event": "ml_prediction_failed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "error",
                "error_status_code": error.status_code,
                "error_detail": error.detail,
            }
        )

        raise

    except Exception as error:
        latency_seconds = time.perf_counter() - start_time

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "ERROR",
                "event": "ml_prediction_failed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "error",
                "error_detail": str(error),
            }
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {error}",
        )


@app.post(
    "/recommend/jobs",
    response_model=RecommendationResponse,
)
async def recommend_jobs(request: Request, req: RecommendationRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")

    endpoint_name = "/recommend/jobs"
    enforce_rate_limit(request, endpoint_name)

    request_id = str(uuid.uuid4())
    payload = req.model_dump()
    input_hash = hash_payload(payload)
    start_time = time.perf_counter()

    try:
        result = await run_with_timeout(
            recommender.recommend,
            payload,
            endpoint_name=endpoint_name,
        )

        latency_seconds = time.perf_counter() - start_time
        recommendations = result.get("recommendations", [])

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "INFO",
                "event": "ml_recommendation_completed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "success",
                "result": {
                    "recommendation_count": len(recommendations),
                    "top_job_post_id": (
                        recommendations[0].get("job_post_id")
                        if recommendations
                        else None
                    ),
                    "top_similarity_score": (
                        recommendations[0].get("similarity_score")
                        if recommendations
                        else None
                    ),
                },
            }
        )

        return result

    except HTTPException as error:
        latency_seconds = time.perf_counter() - start_time

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "ERROR",
                "event": "ml_recommendation_failed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "error",
                "error_status_code": error.status_code,
                "error_detail": error.detail,
            }
        )

        raise

    except Exception as error:
        latency_seconds = time.perf_counter() - start_time

        write_ml_log(
            {
                "timestamp": utc_now_iso(),
                "level": "ERROR",
                "event": "ml_recommendation_failed",
                "request_id": request_id,
                "endpoint": endpoint_name,
                "input_hash": input_hash,
                "latency_seconds": round(latency_seconds, 6),
                "status": "error",
                "error_detail": str(error),
            }
        )

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation error: {error}",
        )