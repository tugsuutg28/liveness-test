from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.pipeline import predict_attack

IMAGE_PATH = Path("datasets/webcam_eval/real/real_001.jpg")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError("Image not found.")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
)

if len(faces) != 1:
    raise RuntimeError(
        f"Expected one face, detected {len(faces)}."
    )

x, y, w, h = faces[0]


def crop_with_padding(frame, padding):
    height, width = frame.shape[:2]

    cx = x + w / 2
    cy = y + h / 2

    side = max(w, h) * (1 + 2 * padding)

    x1 = max(0, int(cx - side / 2))
    y1 = max(0, int(cy - side / 2))
    x2 = min(width, int(cx + side / 2))
    y2 = min(height, int(cy + side / 2))

    return frame[y1:y2, x1:x2]


variants = {
    "original_full_frame": image,
    "tight_crop": crop_with_padding(image, 0.10),
    "medium_crop": crop_with_padding(image, 0.30),
    "wide_crop": crop_with_padding(image, 0.60),
}

output = Path("reports/crop_sensitivity")
output.mkdir(parents=True, exist_ok=True)

models = {
    "baseline": tf.keras.models.load_model(
        "models/deepfake_baseline.keras",
        compile=False,
    ),
    "finetuned": tf.keras.models.load_model(
        "models/deepfake_finetuned.keras",
        compile=False,
    ),
}

for name, frame in variants.items():
    cv2.imwrite(str(output / f"{name}.jpg"), frame)

    print(f"\n{name}")

    for model_name, model in models.items():
        score = predict_attack(model, frame)

        print(f"  {model_name}: {score:.4f}")
        