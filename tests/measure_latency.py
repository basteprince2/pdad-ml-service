import time
import statistics
import httpx


BASE_URL = "http://127.0.0.1:8002"
RUNS = 10


PREDICTION_PAYLOAD = {
    "pwd_profile_id": 1,
    "age": 28,
    "sex": "Male",
    "civil_status": "Single",
    "disability_type": "Visual Disability",
    "disability_visibility": "Visible",
    "cause_of_disability": "Congenital",
    "educational_attainment": "College Graduate",
    "skills": "Computer Literate",
    "mobility_status": "Independent",
    "current_assistive_device": "None",
    "occupation_group": "Clerical Support Workers"
}


RECOMMENDATION_PAYLOAD = {
    "pwd_profile_id": 1,
    "skills": "Computer Literate",
    "disability_type": "Visual Disability",
    "educational_attainment": "College Graduate",
    "predicted_employment_type": "Permanent",
    "top_k": 5
}


def measure_get(client, path):
    durations = []

    for _ in range(RUNS):
        start = time.perf_counter()
        response = client.get(f"{BASE_URL}{path}")
        elapsed = time.perf_counter() - start

        response.raise_for_status()
        durations.append(elapsed)

    return durations


def measure_post(client, path, payload):
    durations = []

    for _ in range(RUNS):
        start = time.perf_counter()
        response = client.post(f"{BASE_URL}{path}", json=payload)
        elapsed = time.perf_counter() - start

        response.raise_for_status()
        durations.append(elapsed)

    return durations


def summarize(name, durations):
    average = statistics.mean(durations)
    minimum = min(durations)
    maximum = max(durations)
    median = statistics.median(durations)

    passed = maximum < 2.0

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Runs:      {len(durations)}")
    print(f"Average:   {average:.4f}s")
    print(f"Median:    {median:.4f}s")
    print(f"Min:       {minimum:.4f}s")
    print(f"Max:       {maximum:.4f}s")
    print(f"Checklist: {'PASS' if passed else 'FAIL'} - max latency under 2 seconds")

    return passed


def main():
    print("PDAD ML Service Latency Measurement")
    print("=" * 40)
    print(f"Base URL: {BASE_URL}")
    print(f"Runs per endpoint: {RUNS}")

    with httpx.Client(timeout=15.0) as client:
        health_durations = measure_get(client, "/health")
        prediction_durations = measure_post(
            client,
            "/predict/employment-type",
            PREDICTION_PAYLOAD
        )
        recommendation_durations = measure_post(
            client,
            "/recommend/jobs",
            RECOMMENDATION_PAYLOAD
        )

    results = [
        summarize("GET /health", health_durations),
        summarize("POST /predict/employment-type", prediction_durations),
        summarize("POST /recommend/jobs", recommendation_durations),
    ]

    print("\nOverall Result")
    print("--------------")
    if all(results):
        print("PASS - All tested ML endpoints responded under 2 seconds.")
    else:
        print("FAIL - At least one endpoint exceeded the 2-second checklist threshold.")


if __name__ == "__main__":
    main()