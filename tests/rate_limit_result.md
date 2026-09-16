# Rate Limit Verification Result

**Date:** 2026-07-10
**Endpoint tested:** `/predict/employment-type`
**Configured limit:** 100 requests/hour/IP
**Test method:** 105 rapid POST requests from a faked non-localhost IP (`10.99.99.99` via `X-Forwarded-For`)

## Result

| Metric             | Value        |
| ------------------ | ------------ |
| Total requests     | 105          |
| Successful (200)   | 100          |
| Rate limited (429) | 5            |
| First 429 at       | Request #101 |
| Duration           | 7.40s        |
| **Verdict**        | **PASSED**   |

## What this proves

The ML service enforces per-IP rate limiting at the configured threshold. The
101st request returned HTTP 429 Too Many Requests, confirming abuse protection
works. The test used a custom `X-Forwarded-For` IP to verify the limiter's key
function correctly distinguishes clients behind a reverse proxy — not just the
raw socket address. Localhost requests remain whitelisted for development.

## Reproduce

```powershell
# Terminal 1
uvicorn main:app --port 8001

# Terminal 2
python tests\test_rate_limit.py
```
