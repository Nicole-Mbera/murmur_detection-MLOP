"""CNN definition, training and window-level evaluation."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score


def build_cnn(input_shape=(64, 157, 1), filters=(16, 32, 64), dropout=0.3):
    inp = keras.Input(shape=input_shape)
    x = inp
    for f in filters:
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out)


def compile_model(model, lr=1e-3):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.AUC(curve="PR", name="pr_auc")])
    return model


def make_ds(X, y, shuffle=False, bs=64, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(4096, seed=seed)
    return ds.batch(bs).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(tf.data.AUTOTUNE)


def predict_proba(model, X, bs=64):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].astype("float32"), training=False).numpy().ravel())
    return np.concatenate(out) if out else np.empty(0, dtype="float32")


def evaluate_windows(model, X, y, threshold=0.5):
    p = predict_proba(model, X)
    pred = (p >= threshold).astype(int)
    y = np.asarray(y).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "n_windows": int(len(y)),
    }
