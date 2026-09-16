import os
from dotenv import load_dotenv


load_dotenv()


ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

ML_TIMEOUT_SECONDS = float(os.getenv("ML_TIMEOUT_SECONDS", "2.0"))

ML_LOG_DIR = os.getenv("ML_LOG_DIR", "./logs")

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))