"""
The actual classifier. Run inspect_data.py first -- this assumes its
output matched the expected label format (action_id,start_frame,end_frame).
If your real labels file looks different, this parser needs adjusting
before running further; the diagnostic output tells you exactly what to
change.

Frozen backbone, not the full pipeline: a pretrained ResNet18 with every
convolutional weight frozen, used purely as a feature extractor. Only a
linear classifier on top of those features is actually fit -- which is
why this is fast enough to run on CPU in minutes, not hours. Almost
nothing is being trained; the pretrained features do the real work.
"""
import os
import glob
import csv

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---- EDIT THIS if your data isn't here ----
DATA_DIR = r"ha4m-demo-data\Demo"

# Keep 1 in every N frames -- consecutive frames barely differ, and a
# frozen-backbone linear classifier needs nowhere near every frame to
# learn a good decision boundary. Lower this if a trial has few frames
# after subsampling; raise it if training feels slow.
SUBSAMPLE_EVERY = 5


def find_trials(data_dir):
    trials = {}
    for root, dirs, files in os.walk(data_dir):
        if os.path.basename(root) == "Color":
            trial_name = os.path.basename(os.path.dirname(root))
            trials[trial_name] = root
    return trials


def list_frames(color_dir):
    """Sorted by FrameID, which is embedded in the filename -- not by
    filesystem order, which isn't guaranteed to match temporal order."""
    paths = glob.glob(os.path.join(color_dir, "**", "FrameID*.png"), recursive=True)

    def frame_id(path):
        name = os.path.basename(path)
        return int(name.split("_")[0].replace("FrameID", ""))

    return sorted(paths, key=frame_id)


def load_labels(labels_path):
    """Parses 'action_id,start_frame,end_frame' rows into a
    {frame_index: action_id} lookup. If your real file's format differs,
    this is the function to change -- inspect_data.py's raw output tells
    you what to change it to."""
    frame_to_action = {}
    with open(labels_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            action_id, start, end = int(row[0]), int(row[1]), int(row[2])
            for frame_idx in range(start, end + 1):
                frame_to_action[frame_idx] = action_id
    return frame_to_action


def build_dataset(trial_root, subsample_every=SUBSAMPLE_EVERY):
    frames = list_frames(trial_root)
    labels_path = os.path.join(os.path.dirname(trial_root), "Labels.txt")
    frame_to_action = load_labels(labels_path)

    paths, labels = [], []
    for i, path in enumerate(frames):
        if i % subsample_every != 0:
            continue
        if i in frame_to_action:
            paths.append(path)
            labels.append(frame_to_action[i])
    return paths, labels


_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_backbone(pretrained=True):
    """Frozen ResNet18, classification head removed -- outputs a 512-dim
    feature vector per image. Every conv weight has requires_grad=False;
    nothing in the backbone itself is ever updated."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_features(paths, backbone, batch_size=16):
    features = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = torch.stack([_preprocess(Image.open(p).convert("RGB")) for p in batch_paths])
            feats = backbone(imgs)
            features.append(feats.numpy())
    return np.concatenate(features, axis=0)


def main():
    trials = find_trials(DATA_DIR)
    if len(trials) < 1:
        print(f"No trials found under {DATA_DIR} -- run inspect_data.py first.")
        return

    trial_names = sorted(trials.keys())
    print(f"Found trials: {trial_names}")

    if len(trials) >= 2:
        train_trial, test_trial = trial_names[0], trial_names[1]
        print(f"Train on {train_trial}, test on {test_trial} (different trial -- "
              f"the honest split, not a random shuffle of the same footage).")
    else:
        train_trial = test_trial = trial_names[0]
        print(f"Only one trial found -- will split {train_trial} by time instead "
              f"(first 70% train, last 30% test). Less rigorous than a held-out "
              f"trial, but still a real train/test separation.")

    train_paths, train_labels = build_dataset(trials[train_trial])
    print(f"Train: {len(train_paths)} frames after subsampling (1 in {SUBSAMPLE_EVERY})")

    if train_trial == test_trial:
        cutoff = int(len(train_paths) * 0.7)
        test_paths, test_labels = train_paths[cutoff:], train_labels[cutoff:]
        train_paths, train_labels = train_paths[:cutoff], train_labels[:cutoff]
    else:
        test_paths, test_labels = build_dataset(trials[test_trial])
    print(f"Test: {len(test_paths)} frames")

    print("\nLoading pretrained ResNet18 (frozen backbone)...")
    backbone = build_backbone(pretrained=True)

    print("Extracting features (no training happening yet, just a forward pass)...")
    X_train = extract_features(train_paths, backbone)
    X_test = extract_features(test_paths, backbone)

    print(f"\nTraining the linear head on {X_train.shape[0]} x {X_train.shape[1]}-dim features...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, train_labels)

    pred = clf.predict(X_test)
    acc = accuracy_score(test_labels, pred)

    print(f"\n=== Held-out accuracy: {acc:.3f} ===")
    print("\nPer-action breakdown:")
    print(classification_report(test_labels, pred, zero_division=0))


if __name__ == "__main__":
    main()
