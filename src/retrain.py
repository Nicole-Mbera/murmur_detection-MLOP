"""Retraining: fine-tune the current model on base samples plus all uploads, then promote only if not worse."""
import os, time, traceback
import numpy as np
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight

from src.model import compile_model, make_ds, evaluate_windows
from src.state import load_state, save_state, write_retrain_status
from src.data_store import load_uploads, list_batches

LOCK_MAX_AGE = 3600  # seconds before a lock file is treated as stale


def _lock_path(models_dir):
    return os.path.join(models_dir, "retrain.lock")


def is_retraining(models_dir):
    p = _lock_path(models_dir)
    return os.path.exists(p) and (time.time() - os.path.getmtime(p)) < LOCK_MAX_AGE


def _acquire(models_dir):
    p = _lock_path(models_dir)
    if os.path.exists(p) and (time.time() - os.path.getmtime(p)) >= LOCK_MAX_AGE:
        os.remove(p)
    try:
        os.close(os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def retrain(models_dir="models", data_dir="data", epochs=5, lr=1e-4, tolerance=0.01, trigger="manual"):
    if not _acquire(models_dir):
        return {"status": "skipped", "reason": "another retraining run is in progress"}
    started = time.time()
    try:
        write_retrain_status(models_dir, status="running", stage="loading data", epoch=0, epochs=epochs,
                             started_at=started, trigger=trigger)
        state = load_state(models_dir)
        base = np.load(os.path.join(data_dir, "train", "train_sample.npz"))
        val = np.load(os.path.join(data_dir, "test", "val_sample.npz"))
        Xu, yu = load_uploads(data_dir)
        X, y = base["X"], base["y"].astype("float32")
        n_up = 0 if Xu is None else len(Xu)
        if Xu is not None:
            X, y = np.concatenate([X, Xu]), np.concatenate([y, yu])
        order = np.random.default_rng(int(started)).permutation(len(X))
        X, y = X[order], y[order]
        Xv, yv = val["X"], val["y"].astype("float32")

        cw = compute_class_weight("balanced", classes=np.array([0., 1.]), y=y)
        class_weight = {0: float(cw[0]), 1: float(cw[1])}

        model = keras.models.load_model(os.path.join(models_dir, state["model_file"]), compile=False)
        compile_model(model, lr)
        before = evaluate_windows(model, Xv, yv, state["threshold"])

        write_retrain_status(models_dir, status="running", stage="training", epoch=0, epochs=epochs,
                             started_at=started, trigger=trigger)
        cb = keras.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: write_retrain_status(
            models_dir, status="running", stage="training", epoch=e + 1, epochs=epochs,
            started_at=started, trigger=trigger, loss=float(logs.get("loss", 0))))
        model.fit(make_ds(X, y, shuffle=True), epochs=epochs, class_weight=class_weight, callbacks=[cb], verbose=0)
        after = evaluate_windows(model, Xv, yv, state["threshold"])

        promoted = after["pr_auc"] >= before["pr_auc"] - tolerance
        state = load_state(models_dir)   # re-read: another container may have changed it
        entry = {"time": time.time(), "trigger": trigger, "train_windows": int(len(X)), "uploaded_windows": int(n_up),
                 "epochs": epochs, "lr": lr, "before": before, "after": after, "promoted": bool(promoted),
                 "tolerance": tolerance}
        if promoted:
            version = state["version"] + 1
            fname = f"murmur_cnn_v{version}.keras"
            model.save(os.path.join(models_dir, fname))
            state.update(version=version, model_file=fname)
            entry["new_version"] = version
        state["used_batches"] = [b["batch_id"] for b in list_batches(data_dir)]
        state["history"] = (state["history"] + [entry])[-20:]
        save_state(models_dir, state)
        write_retrain_status(models_dir, status="done", finished_at=time.time(), started_at=started,
                             trigger=trigger, result=entry)
        return {"status": "done", **entry}
    except Exception as e:
        write_retrain_status(models_dir, status="failed", error=f"{type(e).__name__}: {e}",
                             trace=traceback.format_exc()[-1500:], started_at=started)
        return {"status": "failed", "error": str(e)}
    finally:
        try:
            os.remove(_lock_path(models_dir))
        except FileNotFoundError:
            pass
