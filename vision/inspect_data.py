"""
Run this FIRST, before anything else in vision/. It inspects the actual
downloaded data and reports what it finds -- trial folders, frame counts,
and the raw content of any labels file -- rather than assuming a
structure and failing confusingly three steps later.

This exists because the exact folder layout and label format weren't
knowable in advance (real external data, not something generated here) --
same reasoning as Phase 5's Bosch notebook.
"""
import os
import sys

# ---- EDIT THIS if your data isn't here ----
DATA_DIR = r"ha4m-demo-data\Demo"


def find_trials(data_dir):
    """A 'trial' is any folder containing a Color/ subfolder with PNGs."""
    trials = {}
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "Color":
            trial_name = os.path.basename(os.path.dirname(root))
            png_count = 0
            for sub_root, _, sub_files in os.walk(root):
                png_count += sum(1 for f in sub_files if f.lower().endswith(".png"))
            trials[trial_name] = {"color_path": root, "n_frames": png_count}
    return trials


def find_labels(trial_root):
    """Looks for a labels file anywhere directly under the trial's own
    folder (not inside Color/) -- common names tried, case-insensitive."""
    candidates = ["labels.txt", "label.txt", "annotations.txt", "annotation.txt"]
    for f in os.listdir(trial_root):
        if f.lower() in candidates:
            return os.path.join(trial_root, f)
    return None


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"FAIL: '{DATA_DIR}' does not exist relative to your current directory.")
        print(f"Current directory: {os.getcwd()}")
        print("Fix DATA_DIR at the top of this file, or run this from the linesight root.")
        sys.exit(1)

    trials = find_trials(DATA_DIR)
    if not trials:
        print(f"FAIL: no 'Color' folder containing PNGs found anywhere under '{DATA_DIR}'.")
        print("Walked the tree and found these directories instead:")
        for root, dirs, files in os.walk(DATA_DIR):
            depth = root.replace(DATA_DIR, "").count(os.sep)
            if depth <= 3:
                print("  " * depth + os.path.basename(root) or root)
        sys.exit(1)

    print(f"Found {len(trials)} trial(s) under '{DATA_DIR}':\n")
    for name, info in trials.items():
        print(f"  {name}: {info['n_frames']} frames in {info['color_path']}")

    print()
    for name, info in trials.items():
        # Color is a direct child of the trial folder -- one level up only
        trial_root = os.path.dirname(info["color_path"])
        labels_path = find_labels(trial_root)
        print(f"--- {name} ---")
        if labels_path:
            print(f"  Labels file found: {labels_path}")
            with open(labels_path) as f:
                lines = [next(f, "").rstrip() for _ in range(8)]
            print("  First lines (raw, exactly as stored):")
            for line in lines:
                if line:
                    print(f"    {line}")
        else:
            print(f"  No labels file found directly under {trial_root}")
            print(f"  Contents of that folder: {os.listdir(trial_root)}")
        print()

    print("If the label lines above don't look like 'action_id,start_frame,end_frame'")
    print("(or something clearly equivalent), paste this whole output back and the")
    print("parser in frame_classifier.py will be adjusted to match the real format")
    print("before any training is attempted.")


if __name__ == "__main__":
    main()
