# ASL Recognition

Web-enabled American Sign Language recognition demo using Python, MediaPipe and scikit-learn.

This repo includes a **deployable Flask app** and a reproducible training flow for the Kaggle dataset:
- https://www.kaggle.com/datasets/grassknoted/asl-alphabet

## Features

- Browser webcam capture
- Real-time hand landmark extraction (MediaPipe when available)
- Prediction endpoint (`/process_frame`)
- Supports a real trained model artifact (`model.p`)
- OpenCV fallback hand-presence detection when MediaPipe is unavailable in runtime

## One-click deploy (Render)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/khayem487/asl-recognition)

Render free instance may sleep after inactivity.

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## Training with Kaggle ASL Alphabet

### 1) Install training dependencies

```bash
pip install -r requirements-train.txt
```

### 2) Download + unzip dataset

Use this dataset:
- https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Unzip it somewhere on disk (example path below).

### 3) Train and generate `model.p`

```bash
python train_classifier.py \
  --dataset-root "C:/path/to/asl-alphabet" \
  --dataset-tag "kaggle.com/datasets/grassknoted/asl-alphabet" \
  --output-model model.p \
  --labels-path aa.txt \
  --report-path training_report.json
```

Optional speed control (smaller sample):

```bash
python train_classifier.py --dataset-root "C:/path/to/asl-alphabet" --max-per-class 1200
```

This command will:
- extract MediaPipe hand landmarks from dataset images
- build/update `data.pickle`
- train RandomForest classifier
- save `model.p`, `aa.txt`, and `training_report.json`

### 4) Use model in the web app

Keep `model.p` and `aa.txt` in repo root (same level as `app.py`), then run:

```bash
python app.py
```

The app auto-loads labels from `model.p` metadata (or falls back to `aa.txt`).

## Model behavior

- If `model.p` exists and hand landmarks are available, the app returns model predictions.
- If `model.p` is missing, the app returns hand-presence predictions.
- If MediaPipe is unavailable, the app still runs using a lightweight OpenCV fallback detector.

## Project structure

```text
asl-recognition/
├── app.py
├── requirements.txt
├── requirements-train.txt
├── Dockerfile
├── render.yaml
├── templates/
├── static/
├── create_dataset.py
├── train_classifier.py
└── test_classifier.py
```
