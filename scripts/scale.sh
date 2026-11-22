#!/usr/bin/env bash
# Usage: scripts/scale.sh N   -> run N API containers behind the nginx gateway.
set -euo pipefail
N="${1:?usage: scripts/scale.sh N}"
docker compose up -d --wait --scale api="$N" api gateway
docker compose restart gateway          # nginx reads the API container addresses at start
until curl -sf http://localhost/health >/dev/null; do sleep 1; done
echo "Running with $N API container(s):"
docker compose ps api
