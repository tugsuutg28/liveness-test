from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.pipeline import predict_attack

DATA_DIR = Path("datasets/webcam_eval/real")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

models = {
    "baseline": tf.keras.models.load_model(
        "models/deepfake_baseline.keras",
        compile=False
    ),
    "finetuned": tf.keras.models.load_model(
        "models/deepfake_finetuned.keras",
        compile=False
    ),
}

PADDINGS = [0.10, 0.30, 0.60]


def crop_face(frame, padding):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    height, width = frame.shape[:2]

    cx = x + w / 2
    cy = y + h / 2

    side = max(w, h) * (1 + 2 * padding)

    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(width, int(cx + side / 2))
    y2 = min(height, int(cy + side / 2))

    return frame[y1:y2, x1:x2]


images = sorted(DATA_DIR.glob("*.jpg"))

for padding in PADDINGS:
    crops = []

    for path in images:
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        crop = crop_face(frame, padding)

        if crop is not None:
            crops.append(crop)

    print(f"\nPADDING: {padding}")
    print(f"Images evaluated: {len(crops)}")

    for name, model in models.items():
        scores = np.array([
            predict_attack(model, crop)
            for crop in crops
        ])

        accepted = np.sum(scores < 0.5)

        print(
            f"{name}: "
            f"Accepted {accepted}/{len(scores)} "
            f"({accepted / len(scores):.2%}), "
            f"Average fake score: {scores.mean():.4f}"
        )
        