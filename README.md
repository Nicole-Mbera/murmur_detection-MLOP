# Heart murmur detection from heart sound recordings (CirCor DigiScope)

An end-to-end machine learning pipeline for audio processing and traing a convolutional neural network classifier to detect whether a
heart sound recording has features of murmur **present** or **absent**. The project covers data acquisition and preprocessing, model
training and evaluation, a prediction API, a database for uploaded data, retraining, a web interface, Docker packaging,
cloud deployment and load testing.

The model is a screening tool. It extends my previous assignment (ECG arrhythmia classification) in
the same cardiac screening theme, for resource limited settings where specialist cardiologists are scarce.

- **Video demo:** https://youtu.be/3HfbucBVWFQ
- **Deployed application:** http://13.62.246.159:8501 (UI) and http://13.62.246.159/docs
- **Dataset:** CirCor DigiScope Phonocardiogram Dataset (PhysioNet): https://www.kaggle.com/datasets/bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2

## What the system does

| Requirement | Where |
|---|---|
| Data acquisition, preprocessing, model creation, testing | `notebook/heart_murmur.ipynb`, `src/preprocessing.py`, `src/model.py` |
| Prediction for one recording (.wav) | UI page **Predict**, API `POST /predict`, `src/prediction.py` |
| Visualizations with interpretations | UI page **Data visualizations**, notebook Sections 4 and 6 |
| Upload bulk data, saved to a database | UI page **Upload and retrain**, API `POST /upload`, SQLite database (`src/data_store.py`) |
| Preprocessing of the uploaded data | Done at upload time with the same `preprocessing.py`; a preview is shown in the UI |
| Retraining (button and automatic trigger) | API `POST /retrain`, automatic once enough new recordings are saved, `src/retrain.py` |
| Model up-time | UI page **Overview**, API `GET /status` |
| Load test with different numbers of containers | `locust/`, `scripts/run_load_tests.sh`, results below |

## Project structure
```
README.md
DEPLOY.md                         AWS deployment and load test instructions
WALKTHROUGH.md                    plain-language explanation of the project and the demo plan
notebook/heart_murmur.ipynb       EDA, preprocessing, three experiments, evaluation, saving, demos
src/
  preprocessing.py                audio -> log-mel spectrogram windows (shared by notebook, API, retraining)
  model.py                        CNN definition, training helpers, evaluation metrics
  prediction.py                   loads the model and predicts one recording
  retrain.py                      fine-tuning with a promotion check
  data_store.py                   SQLite database of uploaded recordings
  state.py                        model version and retraining state shared by containers
data/
  train/                          train_sample.npz and val_sample.npz (small samples for retraining)
  test/present, test/absent       60 example recordings from held-out test patients
  uploads/                        uploads.db (created at run time)
models/
  murmur_cnn_v1.h5                trained model
  config.json, state.json         threshold and preprocessing settings; current model version
api/main.py                       FastAPI service
ui/                               Streamlit web interface (ui/assets holds figures and result tables)
nginx.conf                        load balancer configuration
locust/                           load test script and summary script
scripts/                          scale.sh, run_load_tests.sh, reset_demo.sh, score_demo.py, show_db.sh
docker-compose.yml, Dockerfile.*  containers: api, gateway (nginx), ui, locust
results/                          load test output (summary.md, chart and raw Locust files) once the tests have run
tests/test_api.py                 automated end-to-end test of the API (predict, bad inputs, upload, retrain, promotion)
```

## Run it

### With Docker (recommended)
```bash
docker compose up -d --build
```
Open `http://localhost:8501` for the web interface and `http://localhost/docs` for the API documentation.
To run several API containers behind the load balancer: `scripts/scale.sh 3`.
To stop: `docker compose down`. For AWS, see [DEPLOY.md](DEPLOY.md).

### Without Docker
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r ui/requirements.txt
PYTHONPATH=. uvicorn api.main:app --port 8000            # terminal 1
API_URL=http://localhost:8000 streamlit run ui/app.py    # terminal 2
```

### Reproduce the model
Open `notebook/heart_murmur.ipynb` in Kaggle with the CirCor DigiScope dataset attached
(`bjoernjostein/the-circor-digiscope-phonocardiogram-dataset-v2`) and run all cells. The last cells create `export.zip` with the
model, samples, demo recordings and figures in the layout of this repository.

## API
| Endpoint | Purpose |
|---|---|
| `POST /predict` | one .wav file; returns label, murmur probability, per-window probabilities, model version, latency |
| `POST /upload` | several .wav files with `label=present|absent`, or one .zip with `present/` and `absent/` folders |
| `POST /retrain` | start retraining in the background (`epochs` optional) |
| `GET /status` | up-time, model version, request counts, retraining progress, uploads waiting, history |
| `GET /uploads` | latest rows of the uploads database |
| `GET /data/summary`, `GET /health` | data counts and health check |

## How it works
1. **Preprocessing.** Each recording is resampled to 4 kHz, band-pass filtered (25 to 500 Hz), cut into 5 s windows and converted to a
   64 x 157 log-mel spectrogram, standardised per window.
2. **Model.** A small CNN (batch normalisation, dropout, global average pooling) outputs a murmur probability per window. The
   probabilities of all windows of a recording are averaged and compared with a threshold chosen on validation data.
3. **Splits.** Patients are split 70/15/15 into train, validation and test. No patient appears in two splits.
4. **Retraining.** Uploaded recordings are preprocessed and stored in a SQLite database. Retraining fine-tunes the current model on the base
   sample plus all uploads, compares old and new model on a fixed validation sample, and promotes the new model only if it is not
   clearly worse. All containers see the new model after promotion.

## Results

### Model (held-out test patients)
Test set: 132 patients, 459 recordings (80 with a murmur). The model is the selected experiment 3 (deeper CNN with L2 and SpecAugment,
98,001 parameters), with the decision threshold 0.38 chosen on validation recordings. Metrics are per recording, with 95% intervals from
resampling patients.

| Metric | Value | 95% interval |
|---|---|---|
| Accuracy | 0.948 | 0.920 to 0.972 |
| Loss (log loss) | 0.156 | 0.112 to 0.211 |
| Precision (murmur present) | 0.850 | 0.716 to 0.942 |
| Recall (murmur present) | 0.850 | 0.736 to 0.952 |
| F1 (murmur present) | 0.850 | 0.747 to 0.924 |
| ROC-AUC | 0.960 | 0.918 to 0.992 |
| PR-AUC | 0.923 | 0.842 to 0.976 |

The model finds 68 of 80 murmur recordings and flags 12 of 379 absent recordings. Validation results are lower (recording-level
ROC-AUC 0.871, PR-AUC 0.752), so the test figures are best read as an optimistic estimate.

Experiments (validation set, recording level):

| Experiment | Parameters | Validation ROC-AUC | Validation PR-AUC |
|---|---|---|---|
| 1 baseline CNN | 4,833 | 0.726 | 0.532 |
| 2 regularised CNN (batch norm, dropout, class weights, early stopping) | 23,697 | 0.833 | 0.705 |
| 3 deeper CNN with L2 and SpecAugment (selected) | 98,001 | 0.871 | 0.752 |

Full tables, figures and interpretations are in the notebook (Sections 7 and 8).

### Load test (Locust)
<img width="1487" height="566" alt="image" src="https://github.com/user-attachments/assets/5c3c6b51-2db8-4187-be0b-1ccc42aeafd9" />


| API containers | Requests | Failures | Median (ms) | Mean (ms) | p95 (ms) | p99 (ms) | Requests/s | Containers that answered |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 3525 | 0 | 2700 | 2635 | 3100 | 3300 | 29.4 | 1 |
| 2 | 4822 | 1 | 1600 | 1769 | 3800 | 4500 | 40.2 | 2 |
| 3 | 4734 | 3 | 1100 | 1810 | 5100 | 6100 | 39.5 | 3 |


## Limitations
- The label rule keeps a recording from a murmur patient only where a murmur was annotated at that site, so performance is measured on those
  recordings and not on every recording from a murmur patient.
- Results come from one data split and one training run per experiment.
- The demo recordings come from test patients, so after the retraining demo the test set is no longer untouched for the retrained model.
- The API has no authentication, and the model has been validated on one pediatric dataset only.
