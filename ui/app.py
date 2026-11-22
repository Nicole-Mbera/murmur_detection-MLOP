"""Streamlit UI: uptime, prediction, data visualizations, model evaluation, upload and retraining."""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
ASSETS = os.environ.get("ASSETS_DIR", "assets")
DATA = os.environ.get("DATA_DIR", "data")
DEMO_DIR = os.environ.get("DEMO_DIR", "demo_bundle")

st.set_page_config(page_title="Heart murmur screening", layout="wide")


# ---------- API helpers ----------
def _detail(r):
    try:
        d = r.json().get("detail", r.text)
        return d if isinstance(d, str) else str(d)
    except ValueError:
        return r.text[:300]


def api_get(path, timeout=10):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {_detail(r)}"
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: cannot reach the API at {API_URL}"


def api_post(path, timeout=300, **kw):
    try:
        r = requests.post(f"{API_URL}{path}", timeout=timeout, **kw)
        if r.status_code >= 400:
            return None, f"HTTP {r.status_code}: {_detail(r)}"
        return r.json(), None
    except Exception as e:
        return None, f"{type(e).__name__}: cannot reach the API at {API_URL}"


def fmt_duration(sec):
    sec = int(sec)
    d, rem = divmod(sec, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return (f"{d}d " if d else "") + f"{h}h {m}m {s}s"


# ---------- cached data for the visualizations ----------
@st.cache_data
def load_recordings():
    return pd.read_csv(os.path.join(ASSETS, "recordings.csv"))


@st.cache_data
def load_spectrogram_means():
    d = np.load(os.path.join(DATA, "train", "train_sample.npz"))
    X, y = d["X"].astype("float32")[..., 0], d["y"]
    return X[y == 0].mean(0), X[y == 1].mean(0), int((y == 0).sum()), int((y == 1).sum())


# ---------- pages ----------
def page_overview():
    st.header("Model up-time and service status")
    t0 = time.time()
    health, herr = api_get("/health")
    ping_ms = (time.time() - t0) * 1000
    status, serr = api_get("/status")
    if herr or serr:
        st.error(f"API offline. {herr or serr}")
        return
    st.success(f"API online (health check {ping_ms:.0f} ms)")
    c = st.columns(5)
    c[0].metric("Up-time", fmt_duration(status["uptime_seconds"]))
    c[1].metric("Model version", f"v{status['model_version']}")
    c[2].metric("Decision threshold", status["threshold"])
    c[3].metric("Predictions served", status["predictions_served"])
    avg = status["avg_latency_ms"]
    c[4].metric("Avg latency (ms)", "n/a" if avg is None else avg)
    st.caption(f"Answered by container `{status['container']}`. Up-time and request counters belong to that container; "
               "with several containers behind the load balancer they differ from one refresh to the next. "
               "Model version, retraining state and uploads are shared by all containers.")

    ret = status["retraining"]
    st.subheader("Retraining")
    st.write(f"State: **{ret.get('status', 'idle')}**"
             + (" (running now)" if ret.get("running") else ""))
    pend = status["pending_uploads"]
    st.write(f"New uploaded recordings waiting to be used: **{pend['recordings']}** "
             f"(automatic retraining starts at {status['auto_retrain_at_recordings']}).")

    hist = status["retrain_history"]
    if hist:
        rows = [{"time": time.strftime("%Y-%m-%d %H:%M", time.localtime(h["time"])), "trigger": h["trigger"],
                 "train windows": h["train_windows"], "uploaded windows": h["uploaded_windows"],
                 "PR-AUC before": round(h["before"]["pr_auc"], 3), "PR-AUC after": round(h["after"]["pr_auc"], 3),
                 "recall before": round(h["before"]["recall"], 3), "recall after": round(h["after"]["recall"], 3),
                 "promoted": h["promoted"], "new version": h.get("new_version")} for h in hist]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Before and after are measured on the same fixed 400-window validation sample.")
    else:
        st.info("No retraining runs yet.")
    st.button("Refresh")


def page_predict():
    st.header("Predict from one heart sound recording")
    st.write("Upload a .wav recording from a digital stethoscope. The model returns a murmur score for the recording. "
             "It is a screening aid, not a diagnosis.")
    data, name = None, None
    up = st.file_uploader("Recording (.wav)", type=["wav"])
    if up is not None:
        data, name = up.getvalue(), up.name
    samples = []
    if os.path.isdir(DEMO_DIR):
        for lab in ("present", "absent"):
            d = os.path.join(DEMO_DIR, lab)
            if os.path.isdir(d):
                samples += [f"{lab}/{f}" for f in sorted(os.listdir(d)) if f.endswith(".wav")]
    if samples and data is None:
        pick = st.selectbox("Or use a demo recording from the held-out test patients "
                            "(folder name is the dataset label)", ["(none)"] + samples)
        if pick != "(none)":
            with open(os.path.join(DEMO_DIR, pick), "rb") as f:
                data, name = f.read(), pick
    if data is None:
        return
    st.audio(data, format="audio/wav")
    if st.button("Analyse recording", type="primary"):
        with st.spinner("Running the model..."):
            res, err = api_post("/predict", files={"file": (os.path.basename(name), data, "audio/wav")}, timeout=60)
        if err:
            st.error(err)
            return
        left, right = st.columns([1, 2])
        with left:
            if res["label"] == "Present":
                st.error(f"Murmur likely PRESENT")
            else:
                st.success("Murmur likely ABSENT")
            st.metric("Murmur probability", f"{res['murmur_probability']:.2f}",
                      help=f"Recording is called Present when the probability is at least {res['threshold']}")
            st.progress(min(max(res["murmur_probability"], 0.0), 1.0))
            st.caption(f"Threshold {res['threshold']} | model v{res['model_version']} | "
                       f"{res['n_windows']} five-second window(s) | {res['latency_ms']} ms | container {res['container']}")
        with right:
            st.write("Probability per five-second window")
            st.bar_chart(pd.DataFrame({"probability": res["window_probabilities"]}))
        st.caption(res["note"])


def page_visualizations():
    st.header("Data visualizations")
    st.write("Source: CirCor DigiScope heart sound recordings, split by patient. "
             "Only recordings kept by the labelling rule are shown (2,890 of 3,163).")
    try:
        rec = load_recordings()
        m0, m1, n0, n1 = load_spectrogram_means()
    except Exception as e:
        st.error(f"Missing data files: {e}")
        return

    # 1. class balance
    st.subheader("1. Murmur prevalence per split")
    ct = rec.groupby(["split", "label"]).size().unstack(fill_value=0).loc[["train", "val", "test"]]
    ct.columns = ["Absent", "Present"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ct.plot.bar(ax=ax, rot=0)
    for i, s in enumerate(ct.index):
        ax.text(i, ct.loc[s].max() * 1.02, f"{ct.loc[s, 'Present'] / ct.loc[s].sum():.1%} Present", ha="center", fontsize=9)
    ax.set_ylabel("Recordings"); ax.set_xlabel("Split (by patient)")
    a, b = st.columns([1, 1])
    a.pyplot(fig)
    b.markdown(
        "**What it shows.** About 17% of recordings have a murmur in every split (17.3% train, 16.9% validation, 17.4% test), "
        "so the splits are comparable.\n\n"
        "**Why it matters.** A model that always answered Absent would be about 83% accurate and find no murmurs. "
        "Accuracy is therefore not a useful score here, which is why the evaluation uses PR-AUC, recall and precision.")
    plt.close(fig)

    # 2. site
    st.subheader("2. Murmur rate by auscultation site")
    site = rec.groupby("site")["label"].agg(rate="mean", n="count")
    site = site[site["n"] >= 20].sort_values("n", ascending=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(site.index, site["rate"] * 100)
    for i, s in enumerate(site.index):
        ax.text(i, site.loc[s, "rate"] * 100 + 0.3, f"n={int(site.loc[s, 'n'])}", ha="center", fontsize=9)
    ax.set_ylabel("Murmur-present recordings (%)"); ax.set_xlabel("Site (aortic, mitral, pulmonary, tricuspid)")
    a, b = st.columns([1, 1])
    a.pyplot(fig)
    b.markdown(
        "**What it shows.** Murmur rates sit in a narrow band across the four main valve sites (15.3% aortic, 17.2% pulmonary, "
        "18.0% mitral, 18.3% tricuspid), each with 689 to 777 recordings.\n\n"
        "**Why it matters.** The recording site does not separate murmur from no murmur, so the model has to find the murmur in the sound itself. "
        "A site with only 4 recordings was left out. Rates reflect the labelling rule: a recording from a murmur patient counts as Present only "
        "if its site is listed as a murmur location.")
    plt.close(fig)

    # 3. recordings per patient
    st.subheader("3. Recordings per patient")
    per = rec.groupby("patient_id").size().value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(per.index.astype(str), per.values)
    ax.set_xlabel("Recordings from one patient"); ax.set_ylabel("Patients")
    a, b = st.columns([1, 1])
    a.pyplot(fig)
    b.markdown(
        f"**What it shows.** {rec['patient_id'].nunique()} patients contribute {len(rec):,} recordings, "
        f"{len(rec) / rec['patient_id'].nunique():.1f} per patient on average (most have 3 or 4).\n\n"
        "**Why it matters.** Recordings from one child are alike. If they were split at random, the same child would appear in training and test "
        "and the score would be inflated. All splits here are by patient, the same lesson as the inter-patient ECG result.")
    plt.close(fig)

    # 4. spectrograms
    st.subheader("4. What murmur looks like in the sound")
    hz = np.linspace(25, 500, m0.shape[0])
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, img, title, cmap in [(axs[0], m0, "Absent (mean)", "viridis"), (axs[1], m1, "Present (mean)", "viridis"),
                                 (axs[2], m1 - m0, "Present minus Absent", "coolwarm")]:
        im = ax.imshow(img, origin="lower", aspect="auto", extent=[0, 5, 25, 500], cmap=cmap)
        ax.set_title(title); ax.set_xlabel("Time in window (s)"); ax.set_ylabel("Frequency (Hz)")
        plt.colorbar(im, ax=ax)
    st.pyplot(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(hz, m0.mean(1), label="Absent"); ax.plot(hz, m1.mean(1), label="Present")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Mean standardised log-mel level"); ax.legend()
    a, b = st.columns([1, 1])
    a.pyplot(fig)
    b.markdown(
        f"**What it shows.** Averaged over {n0} Absent and {n1} Present training windows (a balanced sample), murmur windows carry relatively less energy "
        "below about 80 Hz and slightly more between roughly 90 and 400 Hz, with the largest gap near 150 Hz. Each window is standardised, so this is a "
        "difference in spectral shape, not loudness. The difference is small (at most 0.1 on a scale that spans about 2), and faint vertical "
        "banding appears roughly once per second.\n\n"
        "**Why it matters.** The signal is subtle on average, which fits a model that finds the murmur with moderate recall, not perfectly. "
        "The averages cannot show how individual murmurs vary.")
    plt.close(fig)


def page_evaluation():
    st.header("Model evaluation (held-out test patients)")
    st.write("Numbers come from the training notebook. The decision threshold was chosen on validation data, then applied once to the test set. "
             "Intervals are 95% bootstrap intervals over patients.")
    p = os.path.join(ASSETS, "results", "test_metrics.csv")
    if os.path.exists(p):
        st.subheader("Test metrics (per recording)")
        st.dataframe(pd.read_csv(p), hide_index=True, width="stretch")
    p = os.path.join(ASSETS, "results", "experiments.csv")
    if os.path.exists(p):
        st.subheader("Experiments (validation)")
        st.dataframe(pd.read_csv(p), hide_index=True, width="stretch")
        st.caption("Single seed. Differences between the variants are small.")
    for f, cap in [("fig5_test_roc_pr_confusion.png", "ROC, precision-recall and confusion matrix on the test set"),
                   ("fig6_training_and_threshold.png", "Training curves and the precision/recall trade-off by threshold")]:
        path = os.path.join(ASSETS, "figures", f)
        if os.path.exists(path):
            st.image(path, caption=cap)


def page_upload_retrain():
    st.header("Upload new data and retrain")
    status, err = api_get("/status")
    if err:
        st.error(err)
        return
    pend = status["pending_uploads"]
    st.write(f"New recordings waiting: **{pend['recordings']}**. Retraining starts automatically at "
             f"**{status['auto_retrain_at_recordings']}**, or when you press the button below.")

    st.subheader("1. Upload labelled recordings")
    mode = st.radio("Format", ["Several .wav files with one label", "One .zip with present/ and absent/ folders"], horizontal=True)
    files, label = [], ""
    if mode.startswith("Several"):
        label = st.selectbox("Label for all selected files", ["present", "absent"])
        files = st.file_uploader("Recordings", type=["wav"], accept_multiple_files=True)
    else:
        z = st.file_uploader("Zip file", type=["zip"])
        files = [z] if z is not None else []
    if st.button("Upload", disabled=not files):
        payload = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
        with st.spinner("Uploading and extracting features..."):
            res, err = api_post("/upload", files=payload, data={"label": label})
        if err:
            st.error(err)
        else:
            st.session_state["last_upload"] = res
    res = st.session_state.get("last_upload")
    if res:
        st.success(f"Accepted {res['accepted_recordings']} recordings ({res['accepted_windows']} windows). "
                   f"Pending: {res['pending_recordings']}." + (" Automatic retraining started." if res["auto_retrain_started"] else ""))
        if res["rejected"]:
            st.warning(f"{len(res['rejected'])} file(s) rejected")
            st.dataframe(pd.DataFrame(res["rejected"]), hide_index=True)

    st.subheader("2. Retrain")
    epochs = st.slider("Fine-tuning epochs", 1, 15, 5)
    running = status["retraining"].get("running", False)
    if st.button("Retrain now", type="primary", disabled=running):
        out, err = api_post("/retrain", params={"epochs": epochs}, timeout=30)
        if err:
            st.error(err)
        else:
            st.info("Retraining started.")
            time.sleep(1)
            st.rerun()

    ret = status["retraining"]
    if running:
        ep, tot = ret.get("epoch", 0), max(ret.get("epochs", 1), 1)
        st.progress(min(ep / tot, 1.0), text=f"{ret.get('stage', 'running')}: epoch {ep} of {tot}")
        time.sleep(2)
        st.rerun()
    elif ret.get("status") == "done":
        r = ret["result"]
        st.write("**Last run**")
        st.dataframe(pd.DataFrame({"before": r["before"], "after": r["after"]}).T.round(3))
        if r["promoted"]:
            st.success(f"Promoted to model v{r['new_version']}.")
        else:
            st.warning("Not promoted: the new model scored clearly worse on the validation sample, so the current model was kept.")
        st.caption(f"Trained on {r['train_windows']} windows ({r['uploaded_windows']} uploaded). "
                   "A promotion means the new model is not clearly worse, which is a weaker claim than better.")
    elif ret.get("status") == "failed":
        st.error(f"Last run failed: {ret.get('error')}")


PAGES = {"Overview": page_overview, "Predict": page_predict, "Data visualizations": page_visualizations,
         "Model evaluation": page_evaluation, "Upload and retrain": page_upload_retrain}

st.sidebar.title("Heart murmur screening")
choice = st.sidebar.radio("Page", list(PAGES))
st.sidebar.caption(f"API: {API_URL}")
PAGES[choice]()
