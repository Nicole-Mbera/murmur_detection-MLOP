"""Small JSON state shared by all API containers through the mounted models/ and data/ folders."""
import json, os, time, tempfile

DEFAULT_STATE = {
    "version": 1,
    "model_file": "murmur_cnn_v1.h5",
    "threshold": 0.73,
    "history": [],        # one entry per retraining attempt
}


def _atomic_write(path, obj):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def load_state(models_dir):
    path = os.path.join(models_dir, "state.json")
    if not os.path.exists(path):
        _atomic_write(path, DEFAULT_STATE)
    with open(path) as f:
        return {**DEFAULT_STATE, **json.load(f)}


def save_state(models_dir, state):
    _atomic_write(os.path.join(models_dir, "state.json"), state)


def write_retrain_status(models_dir, **fields):
    _atomic_write(os.path.join(models_dir, "retrain_status.json"), {"updated_at": time.time(), **fields})


def read_retrain_status(models_dir):
    path = os.path.join(models_dir, "retrain_status.json")
    if not os.path.exists(path):
        return {"status": "idle"}
    with open(path) as f:
        return json.load(f)
