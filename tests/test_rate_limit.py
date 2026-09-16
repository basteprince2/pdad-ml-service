"""
Rate limiting verification script.

Simulates 105 rapid requests to /predict/employment-type from a single 
fake IP to verify the 100-per-hour rate limit triggers a 429 response.

Usage:
    # Ensure FastAPI is running:
    #   uvicorn main:app --port 8001
    #
    # Then run this script:
    python tests/test_rate_limit.py

Expected behavior:
    - Requests 1-100: HTTP 200 (or 422 if payload invalid — that's fine, 
      what matters is being under the limit)
    - Requests 101+: HTTP 429 Too Many Requests
"""
import time
import httpx


BASE_URL = "http://127.0.0.1:8001"
FAKE_IP = "10.99.99.99"  # non-localhost to bypass whitelist
TOTAL_REQUESTS = 105

# Valid payload matching PredictionRequest schema
PAYLOAD = {
    "pwd_profile_id": 1,
    "age": 28,
    "sex": "Male",
    "civil_status": "Single",
    "disability_type": "Visual Disability",
    "disability_visibility": "Apparent",
    "cause_of_disability": "Acquired",
    "educational_attainment": "College Graduate",
    "skills": "customer service",
    "mobility_status": "Independent",
    "current_assistive_device": "Eyeglasses",
    "occupation_group": "Clerical Support Workers",
}


def main():
    print("=" * 60)
    print("RATE LIMIT VERIFICATION TEST")
    print("=" * 60)
    print(f"Target:        {BASE_URL}/predict/employment-type")
    print(f"Fake IP:       {FAKE_IP} (via X-Forwarded-For header)")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Rate limit:    100 per hour per IP")
    print("=" * 60)
    print()

    headers = {
        "Content-Type": "application/json",
        "X-Forwarded-For": FAKE_IP,
        "X-Real-IP": FAKE_IP,
    }

    successful = 0
    rate_limited = 0
    other = 0
    first_429_at = None
    start = time.perf_counter()

    with httpx.Client(timeout=5.0) as client:
        for i in range(1, TOTAL_REQUESTS + 1):
            try:
                response = client.post(
                    f"{BASE_URL}/predict/employment-type",
                    json=PAYLOAD,
                    headers=headers,
                )
                status = response.status_code
            except httpx.RequestError as e:
                status = f"ERR ({type(e).__name__})"

            # Status classification
            if status == 200 or status == 422:
                # 200 = successful prediction
                # 422 = validation error (payload issue, but not rate limited)
                successful += 1
                marker = "OK"
            elif status == 429:
                rate_limited += 1
                if first_429_at is None:
                    first_429_at = i
                marker = "RATE LIMITED"
            else:
                other += 1
                marker = f"OTHER ({status})"

            # Print every 10th request + all 429s
            if i % 10 == 0 or status == 429 or i <= 3:
                print(f"  Request {i:3d}: HTTP {status:5} [{marker}]")

    duration = time.perf_counter() - start
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Duration:       {duration:.2f}s")
    print(f"Successful:     {successful} (200 or 422)")
    print(f"Rate limited:   {rate_limited} (429)")
    print(f"Other errors:   {other}")
    print(f"First 429 at:   request #{first_429_at}" if first_429_at else "First 429 at:   NEVER (rate limit not triggered!)")
    print()

    # Verdict
    if first_429_at is not None and 95 <= first_429_at <= 105:
        print("VERDICT: PASSED - Rate limit enforced at expected threshold.")
    elif first_429_at is None:
        print("VERDICT: FAILED - Rate limit did not trigger. Check whitelist config.")
    else:
        print(f"VERDICT: PARTIAL - Rate limit triggered but at unexpected point (#{first_429_at}).")


if __name__ == "__main__":
    main()