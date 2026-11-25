#!/usr/bin/env bash
# Runs the same Locust load against 1, 2 and 3 API containers and builds a summary table.
# Settings can be overridden: USERS=150 DURATION=180s COUNTS="1 2 3 4" scripts/run_load_tests.sh
set -euo pipefail
USERS="${USERS:-100}"; SPAWN="${SPAWN:-10}"; DURATION="${DURATION:-120s}"; COUNTS="${COUNTS:-1 2 3}"
cd "$(dirname "$0")/.."
mkdir -p results
rm -f results/containers_*
for N in $COUNTS; do
  echo "=== $N API container(s), $USERS users, $DURATION ==="
  scripts/scale.sh "$N"
  sleep 5
  # --no-deps is essential: without it, "docker compose run" resets the api service to one container before the test starts.
  docker compose --profile loadtest run --rm --no-deps -T locust -f /locust/locustfile.py --host http://gateway \
    --headless -u "$USERS" -r "$SPAWN" -t "$DURATION" --csv "/results/containers_$N" --only-summary --exit-code-on-error 0
done
docker compose --profile loadtest run --rm --no-deps -T --entrypoint python locust /locust/summarize.py /results
echo "Results are in ./results (summary.md, load_test_summary.png, containers_*)"
echo "Check the last column of summary.md: 'Containers that answered' must equal the number of API containers in each row."
