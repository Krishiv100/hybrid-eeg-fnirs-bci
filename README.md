# Hybrid EEG-fNIRS Brain-Computer Interface (BCI)

An end-to-end multimodal Brain-Computer Interface pipeline combining **Electroencephalography (EEG)** for high temporal resolution neural dynamics and **functional Near-Infrared Spectroscopy (fNIRS)** for localized hemodynamic responses ($\text{HbO}$ / $\text{HbR}$).

---

## Project Architecture

```text
bids_data/
│
├── src/
│   ├── config.py              # Central hyperparameters & paths
│   ├── preprocess_eeg.py      # Bandpass filtering, CAR, & trial epoching
│   ├── preprocess_fnirs.py    # Modified Beer-Lambert Law (MBLL) & Butterworth filtering
│   ├── features.py            # EEG Welch PSD & fNIRS multi-window hemodynamic features
│   ├── classical_ml.py        # Shrinkage LDA, SVM, Random Forest & Stacking Fusion
│   └── deep_models.py         # PyTorch GatedHybridBCINet Attention Fusion Network
│
├── scripts/
│   ├── 01_inspect_data.py     # Inspects BIDS dataset structure and raw streams
│   ├── 02_train_classical.py  # Runs classical ML benchmarks (LDA, SVM, RF)
│   ├── 03_train_deep_learning.py # Trains and evaluates PyTorch GatedHybridBCINet
│   └── 04_generate_plots.py   # Generates publication-grade performance charts & fNIRS curves
│
└── outputs/
    └── figures/               # Saved performance plots & hemodynamic response curves