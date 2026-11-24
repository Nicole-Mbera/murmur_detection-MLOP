"""Send every demo recording in data/test/ to the running API and save the scores.

Usage (from the project root):   python3 scripts/score_demo.py http://localhost
Writes ui/assets/results/demo_predictions.csv and prints which recordings are safe to use in a demo.
Needs the 'requests' package (python3 -m pip install requests).
"""
import csv
import glob
import os
import sys

import requests

API = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost").rstrip("/")
rows = []
for path in sorted(glob.glob("data/test/*/*.wav")):
    truth = os.path.basename(os.path.dirname(path))          # present or absent
    with open(path, "rb") as f:
        r = requests.post(f"{API}/predict", files={"file": (os.path.basename(path), f, "audio/wav")}, timeout=60)
    r.raise_for_status()
    j = r.json()
    rows.append({"file": f"{truth}/{os.path.basename(path)}", "dataset_label": truth.capitalize(),
                 "probability": j["murmur_probability"], "predicted": j["label"], "windows": j["n_windows"]})
for r in rows:
    r["correct"] = r["dataset_label"] == r["predicted"]

os.makedirs("ui/assets/results", exist_ok=True)
with open("ui/assets/results/demo_predictions.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

for lab in ("Present", "Absent"):
    g = [r for r in rows if r["dataset_label"] == lab]
    print(f"{lab}: {sum(r['correct'] for r in g)} of {len(g)} correct")
good_p = [r for r in rows if r["dataset_label"] == "Present" and r["correct"] and r["probability"] >= 0.9]
good_a = [r for r in rows if r["dataset_label"] == "Absent" and r["correct"] and r["probability"] <= 0.1]
miss = [r for r in rows if not r["correct"]]
print("\nClear correct Present (probability >= 0.9), good for the demo:")
for r in good_p[:6]: print("  ", r["file"], r["probability"])
print("Clear correct Absent (probability <= 0.1), good for the demo:")
for r in good_a[:6]: print("  ", r["file"], r["probability"])
print("Mistakes (do not use as the main demo example):")
for r in miss: print("  ", r["file"], r["dataset_label"], "->", r["predicted"], r["probability"])
print("\nSaved ui/assets/results/demo_predictions.csv")
