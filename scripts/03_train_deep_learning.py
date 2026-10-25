import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config import SUBJECTS, FEATURES_DIR
from src.preprocess_eeg import load_and_preprocess_subject_eeg
from src.preprocess_fnirs import load_and_preprocess_subject_fnirs
from src.features import extract_eeg_psd_features, extract_fnirs_stat_features
from src.deep_models import evaluate_deep_subject


def run_deep_benchmark(subjects_list, task_type='ma'):
    task_label = "Mental Arithmetic (MA)" if task_type == 'ma' else "Motor Imagery (MI)"
    print(f"\n=======================================================")
    print(f" Running Deep Learning (GatedHybridBCINet): {task_label}")
    print(f"=======================================================")

    eeg_accs, fnirs_accs, hybrid_accs = [], [], []

    for sub in subjects_list:
        cache_file = FEATURES_DIR / f"sub-{int(sub):02d}_{task_type}_features.npz"

        if cache_file.exists():
            data = np.load(cache_file)
            X_fnirs = data["X_fnirs_raw"]
            eeg_feats = data["eeg_feats"]
            fnirs_feats = data["fnirs_feats"]
            y = data["y"]
        else:
            X_eeg, y = load_and_preprocess_subject_eeg(sub, task_type=task_type)
            X_fnirs, _ = load_and_preprocess_subject_fnirs(sub, task_type=task_type)
            eeg_feats = extract_eeg_psd_features(X_eeg)
            fnirs_feats = extract_fnirs_stat_features(X_fnirs)

        res = evaluate_deep_subject(eeg_feats, fnirs_feats, X_fnirs, y)
        print(f"Sub-{int(sub):02d} (GatedHybridBCINet) -> EEG: {res['EEG']:5.1f}% | fNIRS: {res['fNIRS']:5.1f}% | Hybrid: {res['Hybrid']:5.1f}%")

        eeg_accs.append(res["EEG"])
        fnirs_accs.append(res["fNIRS"])
        hybrid_accs.append(res["Hybrid"])

    print(f"\n--- Deep Learning Average Across {len(subjects_list)} Subjects ({task_label}) ---")
    print(f"{'Architecture':<20} | {'EEG-Only':<10} | {'fNIRS-Only':<10} | {'Hybrid (EEG+fNIRS)':<18}")
    print("-" * 66)
    print(f"{'GatedHybridBCINet':<20} | {np.mean(eeg_accs):8.2f}% | {np.mean(fnirs_accs):9.2f}% | {np.mean(hybrid_accs):16.2f}%")


if __name__ == "__main__":
    test_subjects = SUBJECTS[:5]
    run_deep_benchmark(test_subjects, task_type='ma')
    run_deep_benchmark(test_subjects, task_type='mi')