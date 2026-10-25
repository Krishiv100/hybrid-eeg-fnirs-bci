import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OUTPUTS_DIR, FEATURES_DIR
from src.preprocess_fnirs import load_and_preprocess_subject_fnirs

def generate_comparison_plots():
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Accuracy Comparison Bar Chart (Mental Arithmetic & Motor Imagery)
    tasks = ['Mental Arithmetic (MA)', 'Motor Imagery (MI)']
    eeg_accs = [92.33, 63.33]
    fnirs_accs = [84.00, 58.67]
    hybrid_accs = [96.67, 68.33]

    x = np.arange(len(tasks))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    rects1 = ax.bar(x - width, eeg_accs, width, label='EEG-Only', color='#4C72B0')
    rects2 = ax.bar(x, fnirs_accs, width, label='fNIRS-Only', color='#DD8452')
    rects3 = ax.bar(x + width, hybrid_accs, width, label='Hybrid (EEG+fNIRS)', color='#55A868')

    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Hybrid EEG-fNIRS BCI Performance (GatedHybridBCINet)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.set_ylim(40, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plot_path = figures_dir / "model_performance_comparison.png"
    plt.savefig(plot_path)
    print(f"Saved performance chart to: {plot_path}")
    plt.close()

    # 2. Hemodynamic Response Curve (HbO vs HbR)
    print("Generating fNIRS hemodynamic response curve from Subject 1...")
    X_fnirs, y_fnirs = load_and_preprocess_subject_fnirs('1', task_type='ma')
    
    # Separate trials by class (Class 0: Mental Arithmetic, Class 1: Baseline)
    class_0_trials = X_fnirs[y_fnirs == 0]
    
    # Average across trials and select a representative channel (e.g., channel 0 for HbO, channel 36 for HbR)
    time_axis = np.linspace(0, 15, class_0_trials.shape[2])
    hbo_mean = np.mean(class_0_trials[:, 0, :], axis=0)       # HbO channel 0
    hbr_mean = np.mean(class_0_trials[:, 36, :], axis=0)     # HbR channel 0

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(time_axis, hbo_mean, label='Oxyhemoglobin (HbO)', color='#D95F02', linewidth=2.5)
    ax.plot(time_axis, hbr_mean, label='Deoxyhemoglobin (HbR)', color='#1B9E77', linewidth=2.5, linestyle='--')
    ax.axvline(0, color='gray', linestyle=':', label='Task Onset')
    
    ax.set_title('Task-Evoked Hemodynamic Response (fNIRS - Sub-01)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time Post-Stimulus (seconds)', fontsize=11)
    ax.set_ylabel('Concentration Change (mmol/L)', fontsize=11)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    fnirs_plot_path = figures_dir / "fnirs_hemodynamic_response.png"
    plt.savefig(fnirs_plot_path)
    print(f"Saved fNIRS response plot to: {fnirs_plot_path}")
    plt.close()

if __name__ == "__main__":
    generate_comparison_plots()