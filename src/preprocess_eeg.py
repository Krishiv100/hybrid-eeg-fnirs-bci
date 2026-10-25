import gc
import warnings
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from src.config import (
    EEG_BIDS_ROOT, MI_SESSIONS, MA_SESSIONS,
    EEG_LOW_CUT, EEG_HIGH_CUT, EPOCH_TMIN, EPOCH_TMAX
)

# Silence MNE and BIDS RuntimeWarnings for clean console output
warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")


def load_and_preprocess_subject_eeg(subject: str, task_type: str = 'mi'):
    """
    Loads and preprocesses all 3 sessions of EEG for a given subject and task.
    task_type: 'mi' (Motor Imagery) or 'ma' (Mental Arithmetic)
    Returns:
        X_eeg: np.ndarray of shape (n_trials, 30, n_samples), dtype=float32
        y: np.ndarray of shape (n_trials,), dtype=int64
    """
    sessions = MI_SESSIONS if task_type == 'mi' else MA_SESSIONS
    all_epochs_data = []
    all_labels = []

    for ses in sessions:
        bids_path = BIDSPath(
            subject=subject,
            session=ses,
            task='imagery',
            run='0',
            datatype='eeg',
            root=EEG_BIDS_ROOT
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = read_raw_bids(bids_path=bids_path, verbose=False)

        # 1. Pick only the 30 EEG channels (excludes VEOG and HEOG cleanly)
        eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
        raw.pick(eeg_picks)
        raw.load_data(verbose=False)

        # 2. Common Average Reference (CAR)
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # 3. Bandpass filter (0.5 - 45 Hz)
        raw.filter(l_freq=EEG_LOW_CUT, h_freq=EEG_HIGH_CUT, verbose=False)

        # 4. Extract events and map class labels to match fNIRS MATLAB order:
        #    MI: 0 = left_hand, 1 = right_hand
        #    MA: 0 = mental_arithmetic (or active task), 1 = baseline / rest
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=EPOCH_TMIN,
            tmax=EPOCH_TMAX,
            baseline=None,
            preload=True,
            verbose=False
        )

        data = epochs.get_data(copy=True).astype(np.float32)

        # Map MNE event codes to exact 0/1 labels matching fNIRS mrk.y
        inv_map = {code: name.lower() for name, code in event_id.items()}
        raw_codes = epochs.events[:, -1]

        if task_type == 'mi':
            labels = np.array([0 if 'left' in inv_map[c] else 1 for c in raw_codes], dtype=np.int64)
        else:
            # In MA sessions, baseline/rest is class 1 in fNIRS mrk.y, arithmetic is class 0
            labels = np.array(
                [1 if ('base' in inv_map[c] or 'rest' in inv_map[c]) else 0 for c in raw_codes],
                dtype=np.int64
            )

        all_epochs_data.append(data)
        all_labels.append(labels)

        del raw, epochs
        gc.collect()

    X_eeg = np.concatenate(all_epochs_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X_eeg, y