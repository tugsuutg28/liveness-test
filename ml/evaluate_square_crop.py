from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.face_crop import crop_face
from ml.pipeline import predict_attack

DATA_DIR = Path("datasets/webcam_eval/real")
PADDINGS = [0.10, 0.30, 0.60]

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Face detector could not be loaded.")

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

images = sorted(DATA_DIR.glob("*.jpg"))

# Detect faces once so all padding settings use identical boxes.
samples = []

for path in images:
    frame = cv2.imread(str(path))

    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 1:
        samples.append((frame, faces[0]))

print(f"Total images: {len(images)}")
print(f"Usable images: {len(samples)}")

for padding in PADDINGS:
    crops = [
        crop_face(frame, bbox, padding=padding)
        for frame, bbox in samples
    ]

    crops = [crop for crop in crops if crop is not None]

    print(f"\nSQUARE CROP — PADDING {padding}")
    print("-" * 40)

    for name, model in models.items():
        scores = np.array([
            predict_attack(model, crop)
            for crop in crops
        ])

        accepted = np.sum(scores < 0.5)
        rejected = np.sum(scores >= 0.5)

        print(f"{name.upper()}")
        print(f"  Genuine accepted: {accepted}/{len(scores)}")
        print(f"  False positives: {rejected}/{len(scores)}")
        print(f"  Genuine acceptance: {accepted / len(scores):.2%}")
        print(f"  Average fake score: {scores.mean():.4f}")