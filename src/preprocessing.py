"""Heart sound preprocessing. One module shared by the notebook, the API and retraining."""
import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

SR = 4000                  # target sample rate (Hz)
LOW_HZ, HIGH_HZ = 25, 500  # band-pass range for heart sounds and murmurs
WIN_SEC = 5.0              # window length
HOP_SEC = 5.0              # hop between windows (non-overlapping)
MIN_SEC = 1.5              # recordings shorter than this are skipped
N_MELS, N_FFT, HOP_LENGTH = 64, 512, 128

_SOS = butter(4, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=SR, output="sos")


def load_audio(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def bandpass(y):
    return sosfiltfilt(_SOS, y).astype(np.float32)


def split_windows(y, max_windows=None):
    """Cut a recording into fixed-length windows. The last partial window is zero-padded
    only if it holds at least MIN_SEC of signal. Returns an array (n_windows, win_samples)."""
    win, hop, min_len = int(WIN_SEC * SR), int(HOP_SEC * SR), int(MIN_SEC * SR)
    if len(y) < min_len:
        return np.empty((0, win), dtype=np.float32)
    windows = []
    for start in range(0, len(y), hop):
        seg = y[start:start + win]
        if len(seg) < win:
            if len(seg) < min_len and windows:
                break
            seg = np.pad(seg, (0, win - len(seg)))
        windows.append(seg)
        if max_windows and len(windows) >= max_windows:
            break
    return np.stack(windows).astype(np.float32)


def to_logmel(window):
    """One window -> standardised log-mel spectrogram of shape (N_MELS, frames)."""
    m = librosa.feature.melspectrogram(
        y=window, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=LOW_HZ, fmax=HIGH_HZ, power=2.0)
    db = librosa.power_to_db(m, ref=1.0, top_db=80.0)
    return ((db - db.mean()) / (db.std() + 1e-6)).astype(np.float32)


def wav_to_features(path, max_windows=None):
    """.wav path -> array (n_windows, N_MELS, frames, 1). Empty array if too short."""
    y = bandpass(load_audio(path))
    wins = split_windows(y, max_windows)
    if len(wins) == 0:
        return np.empty((0, N_MELS, 1 + int(WIN_SEC * SR) // HOP_LENGTH, 1), dtype=np.float32)
    return np.stack([to_logmel(w) for w in wins])[..., None]
