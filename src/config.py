from pathlib import Path

# 1. Directory Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EEG_BIDS_ROOT = PROJECT_ROOT / "nm000267_v1.0.3"
FNIRS_RAW_ROOT = PROJECT_ROOT / "NIRS_01-29"  # Where you will extract NIRS_01-29.zip

PROCESSED_DIR = PROJECT_ROOT / "processed"
EEG_EPOCHS_DIR = PROCESSED_DIR / "eeg_epochs"
FNIRS_EPOCHS_DIR = PROCESSED_DIR / "fnirs_epochs"
FEATURES_DIR = PROCESSED_DIR / "features"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create processed directories if they don't exist
for d in [EEG_EPOCHS_DIR, FNIRS_EPOCHS_DIR, FEATURES_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 2. Dataset Metadata
SUBJECTS = [str(i) for i in range(1, 30)]  # Subjects '1' to '29'
MI_SESSIONS = ['0imagery', '2imagery', '4imagery']
MA_SESSIONS = ['1arithmetic', '3arithmetic', '5arithmetic']

# 3. EEG Preprocessing Parameters
EEG_SFREQ = 200.0
EEG_LOW_CUT = 0.5    # High-pass filter cutoff (Hz)
EEG_HIGH_CUT = 45.0  # Low-pass filter cutoff (Hz)
EPOCH_TMIN = 0.0     # Start of task execution (seconds relative to cue)
EPOCH_TMAX = 10.0    # Shin et al. task duration is 10 seconds per trial

# Standard EEG frequency bands for PSD feature extraction
FREQ_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0)
}