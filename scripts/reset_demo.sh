#!/usr/bin/env bash
# Puts the model, uploaded data and retraining history back to the original v1 state.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f models/murmur_cnn_v[2-9]*.keras models/retrain_status.json models/retrain.lock
cp models/state.v1.json models/state.json
rm -f data/uploads/*.npz data/uploads/*.json
echo "Reset to model v1 with no uploaded data. Restart the API containers: docker compose restart api"
