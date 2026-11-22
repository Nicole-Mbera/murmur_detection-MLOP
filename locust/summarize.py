"""Turn the Locust results of each run (containers_<N>_summary.json) into a table and a chart."""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for path in sorted(glob.glob(os.path.join(folder, "containers_*_summary.json"))):
    n = int(re.search(r"containers_(\d+)_summary", path).group(1))
    r = json.load(open(path))
    seen_path = path.replace("_summary.json", "_containers.json")
    seen = json.load(open(seen_path)) if os.path.exists(seen_path) else {}
    rows.append({"containers": n, "requests": int(r["requests"]), "failures": int(r["failures"]),
                 "median_ms": float(r["median_ms"]), "avg_ms": float(r["avg_ms"]),
                 "p95_ms": float(r["p95_ms"]), "p99_ms": float(r["p99_ms"]), "rps": float(r["rps"]),
                 "containers_seen": len(seen)})
rows.sort(key=lambda x: x["containers"])
if not rows:
    raise SystemExit(f"No containers_*_summary.json files in {folder}")

lines = ["| API containers | Requests | Failures | Median (ms) | Mean (ms) | p95 (ms) | p99 (ms) | Requests/s | Containers that answered |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r['containers']} | {r['requests']} | {r['failures']} | {r['median_ms']:.0f} | {r['avg_ms']:.0f} | "
                 f"{r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['rps']:.1f} | {r['containers_seen']} |")
table = "\n".join(lines)
print(table)
with open(os.path.join(folder, "summary.md"), "w") as f:
    f.write(table + "\n")

x = [r["containers"] for r in rows]
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
axs[0].plot(x, [r["median_ms"] for r in rows], marker="o", label="median")
axs[0].plot(x, [r["p95_ms"] for r in rows], marker="o", label="p95")
axs[0].set_xlabel("API containers"); axs[0].set_ylabel("Response time (ms)"); axs[0].set_xticks(x); axs[0].legend()
axs[0].set_title("Latency")
axs[1].plot(x, [r["rps"] for r in rows], marker="o", color="tab:green")
axs[1].set_xlabel("API containers"); axs[1].set_ylabel("Requests per second"); axs[1].set_xticks(x)
axs[1].set_title("Throughput")
plt.tight_layout(); plt.savefig(os.path.join(folder, "load_test_summary.png"), dpi=150)
