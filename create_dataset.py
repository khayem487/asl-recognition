import argparse
import os
import pickle
import string
from collections import defaultdict
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_LABEL_ORDER = list(string.ascii_uppercase) + ["del", "nothing", "space"]


def normalize_label(raw_label: str):
    if raw_label is None:
        return None

    label = raw_label.strip()
    if not label:
        return None

    if len(label) == 1 and label.isalpha():
        return label.upper()

    lower = label.lower()
    aliases = {
        "delete": "del",
        "del": "del",
        "nothing": "nothing",
        "space": "space",
        "blank": "nothing",
    }
    return aliases.get(lower)


def is_image_file(path: str):
    _, ext = os.path.splitext(path)
    return ext.lower() in IMAGE_EXTENSIONS


def discover_label_directories(dataset_root: str):
    """Find directories named like ASL labels that actually contain image files."""
    label_dirs = defaultdict(list)

    for dirpath, _, filenames in os.walk(dataset_root):
        if not filenames:
            continue

        has_image = any(is_image_file(name) for name in filenames)
        if not has_image:
            continue

        label = normalize_label(os.path.basename(dirpath))
        if label is None:
            continue

        label_dirs[label].append(dirpath)

    return dict(label_dirs)


def extract_features_from_image(image_bgr, hands):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    min_x = min(x_coords)
    min_y = min(y_coords)

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x - min_x)
        features.append(lm.y - min_y)

    return features


def list_images_for_label(label_dirs, label, max_per_class=None):
    files = []
    for directory in label_dirs.get(label, []):
        for name in os.listdir(directory):
            full_path = os.path.join(directory, name)
            if os.path.isfile(full_path) and is_image_file(full_path):
                files.append(full_path)

    files = sorted(set(files))
    if max_per_class and max_per_class > 0:
        files = files[:max_per_class]

    return files


def build_dataset(dataset_root: str, max_per_class=None, min_detection_confidence=0.5, labels=None):
    dataset_root = os.path.abspath(dataset_root)
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

    selected_labels = labels or DEFAULT_LABEL_ORDER
    selected_labels = [normalize_label(lbl) for lbl in selected_labels]
    selected_labels = [lbl for lbl in selected_labels if lbl is not None]

    label_dirs = discover_label_directories(dataset_root)

    data = []
    y = []
    per_label_stats = {}

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        for label in selected_labels:
            image_paths = list_images_for_label(label_dirs, label, max_per_class=max_per_class)
            processed = 0
            kept = 0

            if not image_paths:
                per_label_stats[label] = {
                    "images_found": 0,
                    "samples_kept": 0,
                    "dropped_no_hand": 0,
                }
                continue

            print(f"[DATASET] Label={label} images={len(image_paths)}")

            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    continue

                processed += 1
                features = extract_features_from_image(image, hands)
                if features is None or len(features) != 42:
                    continue

                data.append(features)
                y.append(label)
                kept += 1

            per_label_stats[label] = {
                "images_found": len(image_paths),
                "samples_kept": kept,
                "dropped_no_hand": max(0, processed - kept),
            }

    present_labels = [lbl for lbl in selected_labels if any(item == lbl for item in y)]

    meta = {
        "dataset_root": dataset_root,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "labels_requested": selected_labels,
        "labels_present": present_labels,
        "max_per_class": max_per_class,
        "min_detection_confidence": min_detection_confidence,
        "stats": per_label_stats,
        "total_samples": len(data),
    }

    return data, y, present_labels, meta


def save_dataset_pickle(path: str, data, labels, label_order, meta: dict):
    payload = {
        "data": data,
        "labels": labels,
        "label_order": label_order,
        "meta": meta,
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build landmark dataset from ASL image folders (Kaggle ASL Alphabet compatible)."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to extracted dataset root (e.g. .../asl-alphabet).",
    )
    parser.add_argument(
        "--output",
        default="data.pickle",
        help="Output pickle path (default: data.pickle).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Limit images per class (0 = all).",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels subset (e.g. A,B,C,space,del,nothing).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()] if args.labels else None
    max_per_class = args.max_per_class if args.max_per_class > 0 else None

    data, labels_out, label_order, meta = build_dataset(
        dataset_root=args.dataset_root,
        max_per_class=max_per_class,
        min_detection_confidence=args.min_detection_confidence,
        labels=labels,
    )

    if not data:
        raise RuntimeError(
            "No usable hand-landmark samples were extracted. "
            "Check dataset path / labels / image quality."
        )

    save_dataset_pickle(args.output, data, labels_out, label_order, meta)

    print("\n[OK] Dataset pickle generated")
    print(f"Path: {os.path.abspath(args.output)}")
    print(f"Samples: {len(data)}")
    print(f"Labels: {', '.join(label_order)}")


if __name__ == "__main__":
    main()
