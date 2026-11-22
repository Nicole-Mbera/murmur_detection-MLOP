#!/usr/bin/env bash
# Runs the same Locust load against 1, 2 and 3 API containers and builds a summary table.
# Settings can be overridden: USERS=40 DURATION=180s COUNTS="1 2 3 4" scripts/run_load_tests.sh
set -euo pipefail
USERS="${USERS:-30}"; SPAWN="${SPAWN:-5}"; DURATION="${DURATION:-120s}"; COUNTS="${COUNTS:-1 2 3}"
cd "$(dirname "$0")/.."
mkdir -p results
for N in $COUNTS; do
  echo "=== $N API container(s), $USERS users, $DURATION ==="
  scripts/scale.sh "$N"
  sleep 5
  docker compose --profile loadtest run --rm -T locust -f /locust/locustfile.py --host http://gateway \
    --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" --csv "/results/containers_$N" --only-summary
done
docker compose --profile loadtest run --rm -T --entrypoint python locust /locust/summarize.py /results
echo "Results are in ./results (summary.md, load_test_summary.png, containers_*_stats.csv)"
