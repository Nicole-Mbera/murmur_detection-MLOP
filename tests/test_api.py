import io, os, sys, time, zipfile, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
os.environ["RETRAIN_AUTO_THRESHOLD"] = "12"
from fastapi.testclient import TestClient
from api.main import app
c = TestClient(app)
rng = np.random.default_rng(1)

def wav_bytes(sec=8, sr=4000, f=60):
    t = np.arange(int(sec * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * f * t) * (np.sin(2 * np.pi * 1.2 * t) > 0) + 0.05 * rng.standard_normal(len(t))
    b = io.BytesIO(); sf.write(b, y.astype("float32"), sr, format="WAV"); return b.getvalue()

print("health", c.get("/health").json())
r = c.post("/predict", files={"file": ("a.wav", wav_bytes(), "audio/wav")}); print("predict", r.status_code, {k: v for k, v in r.json().items() if k != "window_probabilities"})
r = c.post("/predict", files={"file": ("a.wav", wav_bytes(0.5), "audio/wav")}); print("too short", r.status_code, r.json())
r = c.post("/predict", files={"file": ("a.txt", b"hello world" * 20, "text/plain")}); print("not wav", r.status_code, r.json())

# upload: 6 files with a label
files = [("files", (f"p{i}.wav", wav_bytes(f=90), "audio/wav")) for i in range(6)]
r = c.post("/upload", files=files, data={"label": "present"}); print("upload labelled", r.status_code, r.json())
# upload: zip with folders, 8 recordings -> crosses the auto threshold (12)
zb = io.BytesIO()
with zipfile.ZipFile(zb, "w") as z:
    for i in range(4): z.writestr(f"present/x{i}.wav", wav_bytes(f=90))
    for i in range(4): z.writestr(f"absent/y{i}.wav", wav_bytes(f=40))
    z.writestr("other/z.wav", wav_bytes()); z.writestr("absent/readme.txt", "x")
r = c.post("/upload", files=[("files", ("batch.zip", zb.getvalue(), "application/zip"))]); print("upload zip", r.status_code, r.json())
r = c.post("/upload", files=[("files", ("n.wav", wav_bytes(), "audio/wav"))], data={"label": "maybe"}); print("bad label", r.status_code, r.json())

for _ in range(120):
    s = c.get("/status").json()
    if s["retraining"].get("status") in ("done", "failed"): break
    time.sleep(2)
print("retrain status:", s["retraining"]["status"], s["retraining"].get("stage"))
res = s["retraining"].get("result") or s["retraining"]
print({k: res.get(k) for k in ("promoted", "train_windows", "uploaded_windows", "new_version", "error")})
if "before" in res: print("before", {k: round(v, 3) for k, v in res["before"].items()}); print("after ", {k: round(v, 3) for k, v in res["after"].items()})
print("version", s["model_version"], "pending", s["pending_uploads"]["recordings"])
r = c.post("/retrain"); print("retrain with nothing pending", r.status_code, r.json())
r = c.post("/predict", files={"file": ("a.wav", wav_bytes(), "audio/wav")}); print("predict after", r.status_code, r.json()["model_version"])
print(c.get("/data/summary").json())
