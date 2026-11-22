"""Load test for POST /predict. Sends real heart sound recordings from the demo bundle."""
import glob
import json
import os
import random
from collections import Counter

from locust import HttpUser, between, events, task

DEMO_DIR = os.environ.get("DEMO_DIR", "/demo")
WAIT_MIN = float(os.environ.get("WAIT_MIN", "0.2"))
WAIT_MAX = float(os.environ.get("WAIT_MAX", "1.0"))

FILES = []
for p in sorted(glob.glob(os.path.join(DEMO_DIR, "*", "*.wav"))):
    with open(p, "rb") as f:
        FILES.append((os.path.basename(p), f.read()))
if not FILES:
    raise SystemExit(f"No .wav files found under {DEMO_DIR}/*/")

SERVED_BY = Counter()   # which API container answered, to show the load balancer spreading requests


class MurmurUser(HttpUser):
    wait_time = between(WAIT_MIN, WAIT_MAX)

    @task
    def predict(self):
        name, data = random.choice(FILES)
        with self.client.post("/predict", files={"file": (name, data, "audio/wav")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
                return
            try:
                body = r.json()
            except ValueError:
                r.failure("response is not JSON")
                return
            if "murmur_probability" not in body:
                r.failure("missing murmur_probability")
                return
            SERVED_BY[body.get("container", "unknown")] += 1
            r.success()


@events.test_stop.add_listener
def save_container_counts(environment, **kwargs):
    prefix = getattr(environment.parsed_options, "csv_prefix", None)
    print("Requests answered per container:", dict(SERVED_BY))
    if prefix:
        with open(f"{prefix}_containers.json", "w") as f:
            json.dump(dict(SERVED_BY), f)
        # Locust's own CSV can lag the final requests slightly, so also save the exact final numbers.
        e = environment.stats.get("/predict", "POST")
        with open(f"{prefix}_summary.json", "w") as f:
            json.dump({"requests": e.num_requests, "failures": e.num_failures,
                       "median_ms": e.median_response_time, "avg_ms": e.avg_response_time,
                       "p95_ms": e.get_response_time_percentile(0.95), "p99_ms": e.get_response_time_percentile(0.99),
                       "rps": e.total_rps}, f)
