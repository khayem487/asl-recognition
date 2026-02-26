import base64
import os
import pickle
import string

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

try:
    import mediapipe as mp
except Exception:
    mp = None

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.p")
LABELS_PATH = os.path.join(BASE_DIR, "aa.txt")


def create_hands_detector():
    if mp is None:
        return None

    mp_hands = None
    try:
        # Common API path
        mp_hands = mp.solutions.hands
    except Exception:
        try:
            # Some mediapipe builds expose solutions only under python package path
            from mediapipe.python.solutions import hands as mp_hands  # type: ignore
        except Exception:
            return None

    try:
        return mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception:
        return None


hands = create_hands_detector()

DEFAULT_LABELS = list(string.ascii_uppercase) + ["space", "del", "nothing"]
MODEL = None
LABELS = DEFAULT_LABELS.copy()


def load_model_bundle(path: str):
    if not os.path.exists(path):
        return None, None

    with open(path, "rb") as f:
        model_obj = pickle.load(f)

    labels = None
    model = model_obj

    if isinstance(model_obj, dict):
        if "model" in model_obj:
            model = model_obj["model"]
        labels_obj = model_obj.get("labels")
        if isinstance(labels_obj, (list, tuple)):
            labels = [str(x) for x in labels_obj if str(x).strip()]

    return model, labels


def load_labels(path: str):
    if not os.path.exists(path):
        return DEFAULT_LABELS.copy()

    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f.readlines() if line.strip()]

    return labels if labels else DEFAULT_LABELS.copy()


def refresh_runtime_artifacts():
    global MODEL, LABELS

    model, labels_from_model = load_model_bundle(MODEL_PATH)
    MODEL = model

    if labels_from_model:
        LABELS = labels_from_model
    else:
        LABELS = load_labels(LABELS_PATH)


refresh_runtime_artifacts()

print(
    f"[INFO] ASL demo started. model_loaded={MODEL is not None}, "
    f"hands_detector_loaded={hands is not None}, labels_count={len(LABELS)}"
)


def extract_landmarks(frame_bgr):
    if hands is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

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


def normalize_features(features, expected_size):
    if len(features) == expected_size:
        return features

    if len(features) > expected_size:
        return features[:expected_size]

    return features + [0.0] * (expected_size - len(features))


def decode_prediction(prediction_value):
    # Numeric-like value -> label index
    try:
        idx = int(prediction_value)
        if 0 <= idx < len(LABELS):
            return LABELS[idx]
    except Exception:
        pass

    # Already a string label
    return str(prediction_value)


def detect_hand_fallback(frame_bgr):
    """Very lightweight fallback when MediaPipe is unavailable.

    This is not ASL classification — only rough hand-presence detection.
    """
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Broad skin-color mask (works reasonably for demo conditions)
    lower = np.array([0, 20, 40], dtype=np.uint8)
    upper = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]

    # Require meaningful foreground area for a positive signal
    return area > 0.02 * frame_area


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_loaded=MODEL is not None,
        detector_loaded=hands is not None,
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "detector_loaded": hands is not None,
            "labels_count": len(LABELS),
        }
    )


@app.route("/process_frame", methods=["POST"])
def process_frame():
    payload = request.get_json(silent=True) or {}
    frame_data = payload.get("frame")

    if not frame_data or "," not in frame_data:
        return jsonify(success=False, error="invalid_frame"), 400

    try:
        frame_b64 = frame_data.split(",", 1)[1]
        frame_bytes = base64.b64decode(frame_b64)
        np_frame = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify(success=False, error="decode_failed"), 400

    if frame is None:
        return jsonify(success=False, error="empty_frame"), 400

    features = extract_landmarks(frame)
    if features is None:
        # Fallback path when MediaPipe is unavailable or no landmarks are produced.
        if detect_hand_fallback(frame):
            return jsonify(success=True, prediction="HAND_DETECTED", mode="fallback-opencv")
        return jsonify(success=False, prediction="NO_HAND", mode="none")

    if MODEL is None:
        return jsonify(success=True, prediction="HAND_DETECTED", mode="fallback")

    expected_size = getattr(MODEL, "n_features_in_", len(features))
    model_input = normalize_features(features, int(expected_size))

    try:
        pred = MODEL.predict([np.asarray(model_input)])[0]
        label = decode_prediction(pred)
        return jsonify(success=True, prediction=label, mode="model")
    except Exception:
        return jsonify(success=True, prediction="HAND_DETECTED", mode="fallback")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
