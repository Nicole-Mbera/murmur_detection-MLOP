"""FastAPI service: prediction, bulk upload, retraining trigger and status."""
import io, os, socket, tempfile, threading, time, zipfile

import tensorflow as tf

# Limit threads per container so that adding containers adds capacity (set TF_THREADS=1 in docker-compose).
_n = int(os.environ.get("TF_THREADS", "0"))
if _n > 0:
    tf.config.threading.set_intra_op_parallelism_threads(_n)
    tf.config.threading.set_inter_op_parallelism_threads(_n)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.data_store import ingest, list_batches, pending_summary
from src.prediction import Predictor
from src.retrain import is_retraining, retrain
from src.state import load_state, read_retrain_status

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
DATA_DIR = os.environ.get("DATA_DIR", "data")
AUTO_RETRAIN_AT = int(os.environ.get("RETRAIN_AUTO_THRESHOLD", "50"))   # new recordings that trigger retraining
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 500
MAX_ZIP_UNCOMPRESSED = 400 * 1024 * 1024

app = FastAPI(title="Heart murmur detection API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STARTED = time.time()
predictor = Predictor(MODELS_DIR)
predictor.warmup()
_stats = {"predictions": 0, "errors": 0, "latency_ms_sum": 0.0}
_stats_lock = threading.Lock()


def _save_temp(data: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _looks_like_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _start_retrain(trigger: str, epochs: int = 5):
    threading.Thread(target=retrain, kwargs=dict(models_dir=MODELS_DIR, data_dir=DATA_DIR, epochs=epochs,
                                                trigger=trigger), daemon=True).start()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"ok": True, "container": socket.gethostname()}


@app.get("/status")
def status():
    state = load_state(MODELS_DIR)
    with _stats_lock:
        n, errs, lat = _stats["predictions"], _stats["errors"], _stats["latency_ms_sum"]
    return {
        "container": socket.gethostname(),
        "uptime_seconds": round(time.time() - STARTED, 1),
        "started_at": STARTED,
        "model_version": state["version"],
        "model_file": state["model_file"],
        "threshold": state["threshold"],
        "predictions_served": n,
        "prediction_errors": errs,
        "avg_latency_ms": round(lat / n, 1) if n else None,
        "retraining": {**read_retrain_status(MODELS_DIR), "running": is_retraining(MODELS_DIR)},
        "pending_uploads": pending_summary(DATA_DIR, state["used_batches"]),
        "auto_retrain_at_recordings": AUTO_RETRAIN_AT,
        "retrain_history": state["history"][-5:],
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.time()
    data = file.file.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(413, "File larger than 20 MB")
    if not _looks_like_wav(data):
        raise HTTPException(400, "Upload a .wav file")
    path = _save_temp(data)
    try:
        result = predictor.predict_path(path)
    except ValueError as e:
        with _stats_lock:
            _stats["errors"] += 1
        raise HTTPException(422, str(e))
    except Exception as e:
        with _stats_lock:
            _stats["errors"] += 1
        raise HTTPException(500, f"Prediction failed: {type(e).__name__}")
    finally:
        os.remove(path)
    ms = (time.time() - t0) * 1000
    with _stats_lock:
        _stats["predictions"] += 1
        _stats["latency_ms_sum"] += ms
    return {**result, "latency_ms": round(ms, 1), "container": socket.gethostname(),
            "note": "Recording-level screening score, not a diagnosis."}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...), label: str = Form("")):
    """Bulk upload for retraining. Send .wav files with label=present|absent, or one .zip with
    present/ and absent/ folders (label then ignored)."""
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"At most {MAX_FILES} files per request")
    label = label.strip().lower()
    items, rejected, tmp_paths = [], [], []
    try:
        for f in files:
            data = f.file.read(MAX_FILE_BYTES * 20 + 1)
            name = f.filename or "unnamed"
            if name.lower().endswith(".zip"):
                try:
                    zf = zipfile.ZipFile(io.BytesIO(data))
                except zipfile.BadZipFile:
                    rejected.append({"file": name, "reason": "not a valid zip"}); continue
                if sum(i.file_size for i in zf.infolist()) > MAX_ZIP_UNCOMPRESSED:
                    rejected.append({"file": name, "reason": "zip too large when extracted"}); continue
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".wav") or "__MACOSX" in info.filename:
                        continue
                    parts = [p.lower() for p in info.filename.split("/")[:-1]]
                    lab = "present" if "present" in parts else "absent" if "absent" in parts else ""
                    if not lab:
                        rejected.append({"file": info.filename, "reason": "not inside a present/ or absent/ folder"}); continue
                    b = zf.read(info)
                    if not _looks_like_wav(b):
                        rejected.append({"file": info.filename, "reason": "not a wav file"}); continue
                    p = _save_temp(b); tmp_paths.append(p); items.append((info.filename, p, lab))
            else:
                if not _looks_like_wav(data):
                    rejected.append({"file": name, "reason": "not a wav file"}); continue
                p = _save_temp(data); tmp_paths.append(p); items.append((name, p, label))
        if not items:
            raise HTTPException(400, {"message": "No usable recordings", "rejected": rejected})
        meta, rej2 = ingest(DATA_DIR, items)
        rejected += rej2
    finally:
        for p in tmp_paths:
            try: os.remove(p)
            except FileNotFoundError: pass
    if meta is None:
        raise HTTPException(400, {"message": "No usable recordings", "rejected": rejected})
    state = load_state(MODELS_DIR)
    pending = pending_summary(DATA_DIR, state["used_batches"])
    auto = False
    if pending["recordings"] >= AUTO_RETRAIN_AT and not is_retraining(MODELS_DIR):
        _start_retrain("auto")
        auto = True
    return {"batch_id": meta["batch_id"], "accepted_recordings": meta["n_recordings"], "accepted_windows": meta["n_windows"],
            "rejected": rejected, "pending_recordings": pending["recordings"],
            "auto_retrain_started": auto, "auto_retrain_at_recordings": AUTO_RETRAIN_AT}


@app.post("/retrain")
def trigger_retrain(epochs: int = 5, force: bool = False):
    if is_retraining(MODELS_DIR):
        raise HTTPException(409, "A retraining run is already in progress")
    state = load_state(MODELS_DIR)
    pending = pending_summary(DATA_DIR, state["used_batches"])
    if pending["recordings"] == 0 and not force:
        raise HTTPException(400, "No new uploaded data since the last retraining. Upload data first or pass force=true.")
    if not 1 <= epochs <= 30:
        raise HTTPException(400, "epochs must be between 1 and 30")
    _start_retrain("manual", epochs)
    return {"started": True, "pending_recordings": pending["recordings"], "epochs": epochs}


@app.get("/data/summary")
def data_summary():
    import numpy as np
    out = {}
    for name, path in [("train_sample", "train/train_sample.npz"), ("validation_sample", "test/val_sample.npz")]:
        y = np.load(os.path.join(DATA_DIR, path))["y"]
        out[name] = {"windows": int(len(y)), "murmur_windows": int(y.sum())}
    batches = list_batches(DATA_DIR)
    out["uploads"] = {"batches": len(batches), "recordings": sum(b["n_recordings"] for b in batches),
                      "windows": sum(b["n_windows"] for b in batches),
                      "murmur_windows": sum(b["n_present_windows"] for b in batches)}
    return out
