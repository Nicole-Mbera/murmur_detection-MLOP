#!/usr/bin/env bash
# Usage: scripts/scale.sh N   -> run N API containers behind the nginx gateway, then check that all N answer.
set -euo pipefail
N="${1:?usage: scripts/scale.sh N}"
cd "$(dirname "$0")/.."
docker compose up -d --wait --scale api="$N" api gateway
docker compose restart gateway          # nginx reads the API container addresses at start
until curl -sf http://localhost/health >/dev/null; do sleep 1; done

# Check that the load balancer really reaches N different containers.
SEEN=$(for _ in $(seq 1 60); do curl -s http://localhost/health; echo; done | grep -o '"container":"[^"]*"' | sort -u | wc -l)
echo "Running with $N API container(s); the gateway reached $SEEN different container(s)."
docker compose ps api
if [ "$SEEN" -ne "$N" ]; then
  echo "WARNING: expected $N containers to answer but saw $SEEN. Check: docker compose logs gateway | tail" >&2
  exit 1
fi
