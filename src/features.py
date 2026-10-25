import numpy as np
from scipy.signal import welch
from src.config import EEG_SFREQ, FREQ_BANDS


def extract_eeg_psd_features(X_eeg: np.ndarray, sfreq: float = EEG_SFREQ) -> np.ndarray:
    """
    Computes log-bandpower features across theta, alpha, beta, and gamma bands.
    Input:  X_eeg of shape (n_trials, 30, n_samples)
    Output: eeg_feats of shape (n_trials, 120)
    """
    freqs, psd = welch(X_eeg, fs=sfreq, nperseg=int(sfreq * 2), axis=-1)

    band_powers = []
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
        bp = np.log10(np.mean(psd[:, :, idx_band], axis=-1) + 1e-12)
        band_powers.append(bp)

    eeg_feats = np.concatenate(band_powers, axis=1)
    return eeg_feats.astype(np.float32)


def _window_mean_and_slope(seg: np.ndarray) -> np.ndarray:
    """Helper to compute mean and linear slope for a time window (n_trials, n_channels, n_win_samples)."""
    n_win = seg.shape[-1]
    mean_val = np.mean(seg, axis=-1)

    t = np.linspace(0.0, 1.0, n_win, dtype=np.float32)
    t_c = t - np.mean(t)
    slope_val = np.sum(seg * t_c[None, None, :], axis=-1) / (np.sum(t_c ** 2) + 1e-12)
    return np.concatenate([mean_val, slope_val], axis=1)


def extract_fnirs_stat_features(X_fnirs: np.ndarray, fs: float = 10.0) -> np.ndarray:
    """
    Extracts hemodynamic mean and slope across three delayed physiological windows
    (2-7s, 7-12s, and 5-15s post-cue) to account for the ~2-5s BOLD/HbO response lag.
    Input:  X_fnirs of shape (n_trials, 72, 150)
    Output: fnirs_feats of shape (n_trials, 72 * 6 = 432)
    """
    w1 = X_fnirs[:, :, int(2 * fs):int(7 * fs)]    # Early rise window (2s - 7s)
    w2 = X_fnirs[:, :, int(7 * fs):int(12 * fs)]   # Peak plateau window (7s - 12s)
    w3 = X_fnirs[:, :, int(5 * fs):int(15 * fs)]   # Full active response window (5s - 15s)

    f1 = _window_mean_and_slope(w1)
    f2 = _window_mean_and_slope(w2)
    f3 = _window_mean_and_slope(w3)

    fnirs_feats = np.concatenate([f1, f2, f3], axis=1)
    return fnirs_feats.astype(np.float32)