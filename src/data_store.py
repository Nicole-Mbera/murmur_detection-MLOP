"""Database for uploaded training data (SQLite).

Each uploaded recording is preprocessed into spectrogram windows and saved as one row, together with its
label and filename. Retraining reads every row, and marks the rows it used. All API containers share the same
database file through the mounted data/ folder.
"""
import io
import os
import sqlite3
import time
import uuid

import numpy as np

from src import preprocessing as P

LABELS = {"present": 1, "absent": 0}
DB_FILE = "uploads.db"

SCHEMA = """CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    label TEXT NOT NULL,
    n_windows INTEGER NOT NULL,
    features BLOB NOT NULL,          -- float16 array (n_windows, 64, 157, 1) saved with numpy
    created_at REAL NOT NULL,
    used_in_run REAL                 -- time of the retraining run that used this row (NULL = not used yet)
)"""


def _connect(data_dir):
    folder = os.path.join(data_dir, "uploads")
    os.makedirs(folder, exist_ok=True)
    con = sqlite3.connect(os.path.join(folder, DB_FILE), timeout=30)
    con.execute(SCHEMA)
    return con


def ingest(data_dir, items):
    """Preprocess and store recordings. items: list of (name, wav_path, label_str).
    Returns (meta, rejected). meta is None when nothing could be stored."""
    rows, rejected, kept, preview = [], [], [], None
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    now = time.time()
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
        buf = io.BytesIO()
        np.save(buf, feats.astype(np.float16))
        rows.append((batch_id, name, label, int(len(feats)), buf.getvalue(), now))
        kept.append({"file": name, "label": label, "windows": int(len(feats))})
        if preview is None:
            preview = {"filename": name, "spectrogram": np.round(feats[0][..., 0], 2).tolist()}
    if not rows:
        return None, rejected
    con = _connect(data_dir)
    with con:
        con.executemany("INSERT INTO recordings (batch_id, filename, label, n_windows, features, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)", rows)
    con.close()
    meta = {"batch_id": batch_id, "n_recordings": len(kept), "n_windows": sum(k["windows"] for k in kept),
            "n_present_windows": sum(k["windows"] for k in kept if k["label"] == "present"),
            "files": kept, "preview": preview}
    return meta, rejected


def pending_summary(data_dir):
    con = _connect(data_dir)
    n, w = con.execute("SELECT COUNT(*), COALESCE(SUM(n_windows), 0) FROM recordings WHERE used_in_run IS NULL").fetchone()
    con.close()
    return {"recordings": int(n), "windows": int(w)}


def db_summary(data_dir):
    con = _connect(data_dir)
    n, w, p = con.execute("SELECT COUNT(*), COALESCE(SUM(n_windows), 0), "
                          "COALESCE(SUM(CASE WHEN label = 'present' THEN n_windows END), 0) FROM recordings").fetchone()
    con.close()
    return {"recordings": int(n), "windows": int(w), "murmur_windows": int(p)}


def list_recent(data_dir, limit=20):
    con = _connect(data_dir)
    rows = con.execute("SELECT id, filename, label, n_windows, created_at, used_in_run FROM recordings "
                       "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [{"id": r[0], "filename": r[1], "label": r[2], "windows": r[3], "uploaded_at": r[4],
             "used_in_retraining": r[5] is not None} for r in rows]


def load_uploads(data_dir):
    """Return (X, y, ids) for every stored recording, or (None, None, []) when the database is empty."""
    con = _connect(data_dir)
    rows = con.execute("SELECT id, label, n_windows, features FROM recordings ORDER BY id").fetchall()
    con.close()
    if not rows:
        return None, None, []
    Xs, ys, ids = [], [], []
    for rid, label, n, blob in rows:
        Xs.append(np.load(io.BytesIO(blob)))
        ys.append(np.full(n, LABELS[label], dtype="float32"))
        ids.append(rid)
    return np.concatenate(Xs), np.concatenate(ys), ids


def mark_used(data_dir, ids, run_time):
    if not ids:
        return
    con = _connect(data_dir)
    with con:
        con.executemany("UPDATE recordings SET used_in_run = ? WHERE id = ? AND used_in_run IS NULL",
                        [(run_time, i) for i in ids])
    con.close()
