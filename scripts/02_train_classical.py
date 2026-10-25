import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from src.config import SUBJECTS, FEATURES_DIR
from src.preprocess_eeg import load_and_preprocess_subject_eeg
from src.preprocess_fnirs import load_and_preprocess_subject_fnirs
from src.features import extract_eeg_psd_features, extract_fnirs_stat_features
from src.classical_ml import evaluate_subject_modalities


def run_benchmark(subjects_list, task_type='ma'):
    task_label = "Mental Arithmetic (MA)" if task_type == 'ma' else "Motor Imagery (MI)"
    print(f"\n=======================================================")
    print(f" Running Classical ML Benchmark: {task_label}")
    print(f"=======================================================")

    summary = {m: {"EEG": [], "fNIRS": [], "Hybrid": []} for m in ["Shrinkage_LDA", "SVM_RBF", "Random_Forest"]}

    for sub in subjects_list:
        X_eeg, y_eeg = load_and_preprocess_subject_eeg(sub, task_type=task_type)
        X_fnirs, y_fnirs = load_and_preprocess_subject_fnirs(sub, task_type=task_type)

        assert np.array_equal(y_eeg, y_fnirs), f"Label mismatch for subject {sub}!"

        eeg_feats = extract_eeg_psd_features(X_eeg)
        fnirs_feats = extract_fnirs_stat_features(X_fnirs)

        # Save extracted features for fast reloading later
        np.savez_compressed(
            FEATURES_DIR / f"sub-{int(sub):02d}_{task_type}_features.npz",
            X_eeg_raw=X_eeg,
            X_fnirs_raw=X_fnirs,
            eeg_feats=eeg_feats,
            fnirs_feats=fnirs_feats,
            y=y_eeg
        )

        res = evaluate_subject_modalities(eeg_feats, fnirs_feats, y_eeg)
        lda_res = res["Shrinkage_LDA"]
        print(f"Sub-{int(sub):02d} (Shrinkage LDA) -> EEG: {lda_res['EEG']:5.1f}% | fNIRS: {lda_res['fNIRS']:5.1f}% | Hybrid: {lda_res['Hybrid']:5.1f}%")

        for m in summary:
            for mod in ["EEG", "fNIRS", "Hybrid"]:
                summary[m][mod].append(res[m][mod])

    print(f"\n--- Average Accuracy Across {len(subjects_list)} Subjects ({task_label}) ---")
    print(f"{'Model':<16} | {'EEG-Only':<10} | {'fNIRS-Only':<10} | {'Hybrid (EEG+fNIRS)':<18}")
    print("-" * 62)
    for m in summary:
        mean_eeg = np.mean(summary[m]["EEG"])
        mean_fnirs = np.mean(summary[m]["fNIRS"])
        mean_hyb = np.mean(summary[m]["Hybrid"])
        print(f"{m:<16} | {mean_eeg:8.2f}% | {mean_fnirs:9.2f}% | {mean_hyb:16.2f}%")


if __name__ == "__main__":
    # Test on the first 5 subjects first for a fast benchmark (~30-45 seconds)
    test_subjects = SUBJECTS[:5]
    run_benchmark(test_subjects, task_type='ma')
    run_benchmark(test_subjects, task_type='mi')