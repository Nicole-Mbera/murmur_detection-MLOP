"""Model loading and prediction. Reloads automatically when another container promotes a new model."""
import os, time, threading
import numpy as np
from tensorflow import keras

from src import preprocessing as P
from src.model import predict_proba
from src.state import load_state

MAX_WINDOWS = 8   # same cap as in training


class Predictor:
    def __init__(self, models_dir="models", check_every=5.0):
        self.models_dir = models_dir
        self.check_every = check_every
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.model = None
        self.version = None
        self.threshold = 0.5
        self.model_file = None
        self.load()

    def load(self):
        state = load_state(self.models_dir)
        model = keras.models.load_model(os.path.join(self.models_dir, state["model_file"]), compile=False)
        with self._lock:
            self.model, self.version = model, state["version"]
            self.threshold, self.model_file = state["threshold"], state["model_file"]

    def refresh_if_stale(self):
        now = time.time()
        if now - self._last_check < self.check_every:
            return
        self._last_check = now
        if load_state(self.models_dir)["version"] != self.version:
            self.load()

    def warmup(self):
        """Run one dummy prediction so first-request latency does not include library start-up."""
        import tempfile, soundfile as sf
        y = (np.random.default_rng(0).standard_normal(int(6 * P.SR)) * 0.05).astype("float32")
        fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        try:
            sf.write(path, y, P.SR)
            self.predict_path(path)
        finally:
            os.remove(path)

    def predict_path(self, path):
        """Predict one .wav recording. Window probabilities are averaged to one recording score."""
        self.refresh_if_stale()
        X = P.wav_to_features(path, max_windows=MAX_WINDOWS)
        if len(X) == 0:
            raise ValueError(f"Recording is shorter than {P.MIN_SEC} s")
        with self._lock:
            model, thr, ver = self.model, self.threshold, self.version
        probs = predict_proba(model, X)
        p = float(probs.mean())
        return {
            "label": "Present" if p >= thr else "Absent",
            "murmur_probability": round(p, 4),
            "threshold": thr,
            "n_windows": int(len(probs)),
            "window_probabilities": [round(float(v), 4) for v in probs],
            "model_version": ver,
        }
