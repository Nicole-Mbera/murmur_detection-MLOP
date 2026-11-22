# Heart murmur detection from heart sound recordings (CirCor DigiScope)

Binary classification of phonocardiogram audio (murmur present / absent), with an end-to-end pipeline:
data processing, model training and evaluation, a prediction API, retraining on uploaded data, a web UI, Docker deployment
and load testing.

> TODO before submission: add the video demo link, the deployed URL, and the load test results (see sections below).

## Project structure
```
notebook/heart_murmur_pipeline.ipynb   preprocessing, EDA, experiments, evaluation
src/        preprocessing.py, model.py, prediction.py, retrain.py, data_store.py, state.py
api/        FastAPI service (predict, upload, retrain, status)
ui/         Streamlit interface
data/       train/ and test/ small samples used for retraining; uploads/ holds user-uploaded data
models/     model file (.keras) and state.json (version, threshold)
assets/     figures, result tables and recordings.csv used by the UI
demo_bundle/ 60 example recordings (present/ and absent/) from held-out test patients
locust/     load test script and summarizer
nginx/      load balancer configuration
scripts/    scale.sh, run_load_tests.sh, reset_demo.sh
```

## Run it
See [DEPLOY.md](DEPLOY.md) for AWS EC2. On any machine with Docker: `docker compose up -d --build`, then open
`http://localhost:8501` (UI) and `http://localhost/docs` (API).

## Results
TODO: model results from the notebook, and the Locust table from `results/summary.md`.
