import argparse
import json
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from create_dataset import build_dataset, save_dataset_pickle


def load_dataset_pickle(path: str):
    with open(path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Invalid dataset pickle format: expected dict payload")

    data = payload.get("data", [])
    labels = payload.get("labels", [])
    label_order = payload.get("label_order") or sorted(set(labels))
    meta = payload.get("meta", {})

    return data, labels, label_order, meta


def keep_consistent_feature_size(data, labels):
    lengths = Counter(len(row) for row in data)
    if not lengths:
        return [], [], None

    expected = 42 if 42 in lengths else lengths.most_common(1)[0][0]

    kept_x = []
    kept_y = []
    for row, label in zip(data, labels):
        if len(row) == expected:
            kept_x.append(row)
            kept_y.append(label)

    return kept_x, kept_y, expected


def ensure_min_samples_per_class(labels, min_count=2):
    counts = Counter(labels)
    too_small = [label for label, count in counts.items() if count < min_count]
    return counts, too_small


def save_labels(path: str, labels):
    with open(path, "w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ASL classifier (RandomForest) from landmark features."
    )
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Optional dataset root (Kaggle ASL Alphabet extracted folder). If set, dataset pickle is generated on the fly.",
    )
    parser.add_argument(
        "--data-pickle",
        default="data.pickle",
        help="Input (or generated) dataset pickle path.",
    )
    parser.add_argument(
        "--output-model",
        default="model.p",
        help="Output model path (default: model.p).",
    )
    parser.add_argument(
        "--labels-path",
        default="aa.txt",
        help="Output labels text file path (default: aa.txt).",
    )
    parser.add_argument(
        "--report-path",
        default="training_report.json",
        help="Training report JSON output.",
    )
    parser.add_argument(
        "--dataset-tag",
        default="kaggle.com/datasets/grassknoted/asl-alphabet",
        help="Dataset identifier stored in model metadata.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="RandomForest n_estimators.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=0,
        help="RandomForest max_depth (0 = None).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="When --dataset-root is used: cap samples per class (0 = all).",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated labels subset for dataset build.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe min detection confidence for dataset build.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Build or load dataset
    if args.dataset_root:
        print(f"[INFO] Building dataset from: {args.dataset_root}")
        labels_filter = [lbl.strip() for lbl in args.labels.split(",") if lbl.strip()] if args.labels else None
        max_per_class = args.max_per_class if args.max_per_class > 0 else None

        data, labels, label_order, dataset_meta = build_dataset(
            dataset_root=args.dataset_root,
            max_per_class=max_per_class,
            min_detection_confidence=args.min_detection_confidence,
            labels=labels_filter,
        )

        if not data:
            raise RuntimeError("No samples extracted from dataset root")

        save_dataset_pickle(args.data_pickle, data, labels, label_order, dataset_meta)
        print(f"[INFO] Saved intermediate dataset pickle: {os.path.abspath(args.data_pickle)}")
    else:
        print(f"[INFO] Loading dataset pickle: {args.data_pickle}")
        data, labels, label_order, dataset_meta = load_dataset_pickle(args.data_pickle)

    # 2) Cleanup consistency
    x, y, feature_size = keep_consistent_feature_size(data, labels)
    if not x:
        raise RuntimeError("No valid samples after feature-size filtering")

    class_counts, too_small = ensure_min_samples_per_class(y, min_count=2)
    if too_small:
        raise RuntimeError(
            f"At least one class has <2 samples (cannot stratify): {too_small}. "
            "Add more samples or train on a subset."
        )

    x = np.asarray(x)
    y = np.asarray(y)

    # 3) Split and train
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        shuffle=True,
        stratify=y,
        random_state=args.random_state,
    )

    max_depth = None if args.max_depth <= 0 else args.max_depth
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    # 4) Evaluate
    y_pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    labels_for_metrics = sorted(list(set(y.tolist())))
    conf = confusion_matrix(y_test, y_pred, labels=labels_for_metrics).tolist()
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=labels_for_metrics,
        output_dict=True,
        zero_division=0,
    )

    print(f"[RESULT] accuracy={accuracy:.4f} f1_weighted={f1_weighted:.4f}")

    # 5) Save model bundle
    model_bundle = {
        "model": model,
        "labels": list(model.classes_),
        "feature_size": int(feature_size),
        "dataset": args.dataset_tag,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
        },
    }

    with open(args.output_model, "wb") as f:
        pickle.dump(model_bundle, f)

    save_labels(args.labels_path, list(model.classes_))

    report = {
        "dataset": args.dataset_tag,
        "trained_at": model_bundle["trained_at"],
        "feature_size": int(feature_size),
        "samples_total": int(len(x)),
        "samples_train": int(len(x_train)),
        "samples_test": int(len(x_test)),
        "classes": list(model.classes_),
        "class_counts": dict(class_counts),
        "metrics": {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
        },
        "confusion_matrix": {
            "labels": labels_for_metrics,
            "matrix": conf,
        },
        "classification_report": cls_report,
        "dataset_meta": dataset_meta,
        "artifacts": {
            "model": os.path.abspath(args.output_model),
            "labels": os.path.abspath(args.labels_path),
            "data_pickle": os.path.abspath(args.data_pickle),
        },
    }

    with open(args.report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Model saved: {os.path.abspath(args.output_model)}")
    print(f"[OK] Labels saved: {os.path.abspath(args.labels_path)}")
    print(f"[OK] Report saved: {os.path.abspath(args.report_path)}")


if __name__ == "__main__":
    main()
