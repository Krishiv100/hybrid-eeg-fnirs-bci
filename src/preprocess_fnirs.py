import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
from src.config import FNIRS_RAW_ROOT


def modified_beer_lambert_law(raw_intensity: np.ndarray) -> np.ndarray:
    """
    Converts raw fNIRS light intensity (n_samples, 72) at 760nm (0:36) and 850nm (36:72)
    into HbO (0:36) and HbR (36:72) concentration changes using the Modified Beer-Lambert Law.
    """
    eps = 1e-10
    I = np.clip(raw_intensity.astype(np.float64), eps, None)
    I0 = np.mean(I, axis=0, keepdims=True)

    dod = -np.log10(I / I0)
    dod_760 = dod[:, :36]
    dod_850 = dod[:, 36:]

    E = np.array([
        [1.4866, 3.8437],
        [2.5264, 1.7986]
    ])
    dpf = 6.0
    dist = 3.0
    E_inv = np.linalg.inv(E) / (dpf * dist)

    hbo = E_inv[0, 0] * dod_760 + E_inv[0, 1] * dod_850
    hbr = E_inv[1, 0] * dod_760 + E_inv[1, 1] * dod_850

    return np.concatenate([hbo, hbr], axis=1)


def bandpass_filter_fnirs(data: np.ndarray, fs: float = 10.0, lowcut: float = 0.01, highcut: float = 0.1) -> np.ndarray:
    """
    Zero-phase 3rd-order Butterworth bandpass filter (0.01 - 0.1 Hz).
    """
    nyq = 0.5 * fs
    b, a = butter(3, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_and_preprocess_subject_fnirs(subject: str, task_type: str = 'mi', fnirs_tmax: float = 15.0):
    """
    Loads and preprocesses all 3 sessions of fNIRS for a given subject and task.
    Uses a 0 to 15s window (150 samples at 10 Hz) to capture the delayed hemodynamic peak.
    Returns:
        X_fnirs: np.ndarray of shape (60, 72, 150), dtype=float32
        y: np.ndarray of shape (60,), dtype=int64
    """
    sub_folder = FNIRS_RAW_ROOT / f"subject {int(subject):02d}"
    cnt_mat = sio.loadmat(sub_folder / "cnt.mat", squeeze_me=True, struct_as_record=False)['cnt']
    mrk_mat = sio.loadmat(sub_folder / "mrk.mat", squeeze_me=True, struct_as_record=False)['mrk']

    session_indices = [0, 2, 4] if task_type == 'mi' else [1, 3, 5]
    all_epochs = []
    all_labels = []

    for idx in session_indices:
        cnt = cnt_mat[idx]
        mrk = mrk_mat[idx]
        fs = float(cnt.fs)  # 10.0 Hz

        hb_continuous = modified_beer_lambert_law(cnt.x)
        hb_filtered = bandpass_filter_fnirs(hb_continuous, fs=fs)

        n_samples = int(fnirs_tmax * fs)   # 150 samples (15s post-cue)
        base_samples = int(2.0 * fs)       # 20 samples (-2s to 0s pre-cue baseline)

        times_ms = mrk.time
        labels = np.argmax(mrk.y, axis=0).astype(np.int64)

        session_epochs = []
        for t_ms in times_ms:
            onset_sample = int(round(t_ms * fs / 1000.0))
            base_start = max(0, onset_sample - base_samples)
            baseline = np.mean(hb_filtered[base_start:onset_sample, :], axis=0, keepdims=True)

            end_sample = onset_sample + n_samples
            if end_sample <= hb_filtered.shape[0]:
                trial_segment = hb_filtered[onset_sample:end_sample, :] - baseline
            else:
                # Pad edge case if trial is at the very end of recording
                valid = hb_filtered[onset_sample:, :] - baseline
                pad_len = n_samples - valid.shape[0]
                trial_segment = np.pad(valid, ((0, pad_len), (0, 0)), mode='edge')

            session_epochs.append(trial_segment.T.astype(np.float32))

        all_epochs.append(np.stack(session_epochs, axis=0))
        all_labels.append(labels)

    X_fnirs = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X_fnirs, y