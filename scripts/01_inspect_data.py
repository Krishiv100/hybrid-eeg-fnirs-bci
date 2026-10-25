from pathlib import Path
import mne
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

PROJECT_ROOT = Path(__file__).resolve().parent.parent
bids_root = PROJECT_ROOT / "nm000267_v1.0.3"

# 1. Get all subject IDs in the dataset
subjects = get_entity_vals(bids_root, 'subject')
print(f"Total subjects ({len(subjects)}): {subjects}")

# 2. Pick the first subject and print all files inside its folder
first_sub = subjects[0]
sub_dir = bids_root / f"sub-{first_sub}"
print(f"\nFiles inside sub-{first_sub}:")
for p in sorted(sub_dir.rglob("*")):
    if p.is_file():
        print("  -", p.relative_to(bids_root))

# 3. Load the first session ('0imagery') for the first subject
bids_path = BIDSPath(
    subject=first_sub,
    session='0imagery',
    task='imagery',
    run='0',
    datatype='eeg',
    root=bids_root
)

print(f"\nLoading file: {bids_path.fpath}")
raw = read_raw_bids(bids_path=bids_path, verbose=False)
raw.load_data()

# 4. Inspect channels, sampling rate, and event markers
print("\n--- Summary Info ---")
print(f"Sampling Frequency: {raw.info['sfreq']} Hz")
print(f"Total Channels ({len(raw.ch_names)}): {raw.ch_names}")
print(f"Channel Types: {set(raw.get_channel_types())}")

events, event_id = mne.events_from_annotations(raw, verbose=False)
print(f"\nEvent Markers (event_id): {event_id}")
print(f"First 5 events:\n{events[:5]}")