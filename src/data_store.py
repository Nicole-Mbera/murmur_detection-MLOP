"""Storage for uploaded training data: one .npz of spectrogram windows plus a .json sidecar per batch."""
import os, json, time, uuid
import numpy as np

from src import preprocessing as P

LABELS = {"present": 1, "absent": 0}


def ingest(data_dir, items):
    """items: list of (name, wav_path, label_str). Returns (summary, rejected)."""
    X, y, kept, rejected = [], [], [], []
    for name, path, label in items:
        if label not in LABELS:
            rejected.append({"file": name, "reason": f"label must be present or absent, got {label!r}"})
            continue
        try:
            feats = P.wav_to_features(path, max_windows=8)
        except Exception as e:
            rejected.append({"file": name, "reason": f"unreadable audio ({type(e).__name__})"})
            continue
        if len(feats) == 0:
            rejected.append({"file": name, "reason": f"shorter than {P.MIN_SEC} s"})
            continue
        X.append(feats.astype(np.float16))
        y += [LABELS[label]] * len(feats)
        kept.append({"file": name, "label": label, "windows": int(len(feats))})
    if not kept:
        return None, rejected
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    folder = os.path.join(data_dir, "uploads")
    os.makedirs(folder, exist_ok=True)
    np.savez_compressed(os.path.join(folder, batch_id + ".npz"), X=np.concatenate(X), y=np.array(y, dtype="float32"))
    meta = {"batch_id": batch_id, "created_at": time.time(), "n_recordings": len(kept),
            "n_windows": int(len(y)), "n_present_windows": int(sum(y)), "files": kept}
    with open(os.path.join(folder, batch_id + ".json"), "w") as f:   # sidecar written last = batch is complete
        json.dump(meta, f)
    return meta, rejected


def list_batches(data_dir):
    folder = os.path.join(data_dir, "uploads")
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json"):
            with open(os.path.join(folder, fn)) as f:
                out.append(json.load(f))
    return out


def pending_summary(data_dir, used_batches):
    pend = [b for b in list_batches(data_dir) if b["batch_id"] not in set(used_batches)]
    return {"recordings": sum(b["n_recordings"] for b in pend), "windows": sum(b["n_windows"] for b in pend),
            "batches": [b["batch_id"] for b in pend]}


def load_uploads(data_dir):
    Xs, ys = [], []
    for b in list_batches(data_dir):
        d = np.load(os.path.join(data_dir, "uploads", b["batch_id"] + ".npz"))
        Xs.append(d["X"]); ys.append(d["y"])
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)
