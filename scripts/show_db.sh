#!/usr/bin/env bash
# Prints the uploads database (table "recordings") from inside the running API container. Useful in the demo video.
cd "$(dirname "$0")/.."
docker compose exec -T api python - <<'PY'
import sqlite3, time
con = sqlite3.connect("/app/data/uploads/uploads.db")
print("Database: data/uploads/uploads.db | table: recordings (the spectrogram features are stored as a BLOB and not printed)")
print(f'{"id":>4}  {"file":<24}{"label":<9}{"windows":>7}  {"uploaded":<20}used in retraining')
for rid, name, label, n, created, used in con.execute(
        "SELECT id, filename, label, n_windows, created_at, used_in_run FROM recordings ORDER BY id DESC LIMIT 15"):
    print(f'{rid:>4}  {name[:23]:<24}{label:<9}{n:>7}  {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created)):<20}{"yes" if used else "no"}')
rows, windows = con.execute("SELECT COUNT(*), COALESCE(SUM(n_windows), 0) FROM recordings").fetchone()
print(f"total: {rows} recordings, {windows} windows")
PY
