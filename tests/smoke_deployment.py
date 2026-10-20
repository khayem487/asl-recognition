"""Probe a real deployment without opening a webcam or using personal images.

python tests/smoke_deployment.py https://example.vercel.app --hand-image /path/to/public-fixture.jpg
The optional fixture must contain a detectable hand. The test verifies the real
model path, not the accuracy of a particular sign or a webcam session.
"""

import argparse
import base64
import json
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import zlib


def call(base, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read()
        return response.status, raw, round(time.monotonic() - started, 3)


def blank_png():
    def chunk(kind, content):
        value = kind + content
        return struct.pack(">I", len(content)) + value + struct.pack(">I", zlib.crc32(value))

    header = struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0)
    pixels = zlib.compress((b"\x00" + b"\x00" * 64 * 3) * 64)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--hand-image", type=Path)
    args = parser.parse_args()
    checks = []

    status, raw, elapsed = call(args.base_url, "/health")
    health = json.loads(raw)
    assert status == 200, (status, raw)
    assert health["model_loaded"] and health["detector_loaded"], health
    assert health["labels_count"] == 28, health
    checks.append({"check": "real_model_and_mediapipe_ready", "seconds": elapsed, "result": health})

    status, raw, elapsed = call(args.base_url, "/")
    assert status == 200 and b"Start Camera" in raw
    checks.append({"check": "original_camera_interface", "seconds": elapsed})

    status, raw, elapsed = call(args.base_url, "/process_frame", {})
    assert status == 400 and json.loads(raw)["error"] == "invalid_frame"
    checks.append({"check": "invalid_frame_rejected", "seconds": elapsed})

    frame = "data:image/png;base64," + base64.b64encode(blank_png()).decode()
    status, raw, elapsed = call(args.base_url, "/process_frame", {"frame": frame})
    result = json.loads(raw)
    assert status == 200 and result["prediction"] == "NO_HAND", result
    checks.append({"check": "blank_frame_has_no_hand", "seconds": elapsed, "result": result})

    if args.hand_image:
        frame = "data:image/jpeg;base64," + base64.b64encode(args.hand_image.read_bytes()).decode()
        status, raw, elapsed = call(args.base_url, "/process_frame", {"frame": frame})
        result = json.loads(raw)
        assert status == 200 and result.get("mode") == "model" and result.get("success"), result
        checks.append({"check": "public_hand_fixture_runs_actual_classifier", "seconds": elapsed, "result": result})

    print(json.dumps({"base_url": args.base_url, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
