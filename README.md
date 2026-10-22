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

## Deploy the actual application on Vercel

**Live application: https://khayem-asl.vercel.app/**

This Vercel project is connected to `khayem487/asl-recognition`. A push to
`master` rebuilds production automatically. `vercel.json` explicitly routes
requests to `Dockerfile.vercel`, including its trained-model startup checks.
Deployment previews remain protected. Local edits take effect only after a push.
The browser policy allows the application's own camera workflow and disables
microphone access; this project only needs video frames for hand recognition.

`Dockerfile.vercel` deploys the same Flask webcam application, MediaPipe detector
and committed 28-class RandomForest artifact. Camera frames go to
`POST /process_frame`; this is server-side inference, rather than a static
simulation or a hand-presence substitute. The classifier is kept unchanged.

```bash
npx vercel@64.7.0 deploy --prod
```

The image uses Python 3.12, Linux OpenCV system libraries, and scikit-learn 1.8.0
to match the version that serialized `model.p`. The build checks that both
MediaPipe and the model load. `REQUIRE_ASL_MODEL=1` prevents publishing a broken
inference runtime as a successful fallback. Concurrent frame extraction is
serialized because MediaPipe's graph is shared by the process.

The Vercel Container Runtime is currently beta and scales idle production
instances down after five minutes. A cold start remains possible; deploying on
Vercel does not promise an always-warm server. See the official
[container runtime documentation](https://vercel.com/docs/functions/runtimes/container).
Render remains available as a separate deployment.

To verify the running server without activating a webcam:

```bash
python tests/smoke_deployment.py https://YOUR_PROJECT.vercel.app
# Optional: exercise MediaPipe and the actual classifier using a public hand image.
python tests/smoke_deployment.py https://YOUR_PROJECT.vercel.app --hand-image /path/to/hand.jpg
```

The fixture test requires `mode: "model"`; a hand-presence fallback fails the
check. It establishes that the trained inference pipeline runs, and does not
measure live webcam recognition accuracy.

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

## No-local-training option (GitHub Actions cloud)

If your PC is too slow, use the built-in workflow:

1. Go to **Actions** tab in this repo.
2. Run workflow: **"Train ASL model (cloud)"**.
3. Keep defaults (or tune `max_per_class` / `n_estimators`).
4. At the end:
   - artifacts (`model.p`, `aa.txt`, `training_report.json`) are uploaded, and
   - if `model.p` is not too large, it is auto-committed to the repo.

Then trigger a Render redeploy so production loads the new `model.p`.

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
