# ASL Recognition

Web-enabled American Sign Language recognition demo using Python, MediaPipe and scikit-learn.

This repo now includes a **deployable Flask app** so the project can run as a hosted browser demo.

## Features

- Browser webcam capture
- Real-time hand landmark extraction (MediaPipe when available)
- Prediction endpoint (`/process_frame`)
- Works with optional trained `model.p`
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

## Optional full training stack

For training scripts (`train_classifier.py`, plots, etc.):

```bash
pip install -r requirements-train.txt
```

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
