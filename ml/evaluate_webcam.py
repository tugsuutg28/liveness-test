"""Evaluate baseline and fine-tuned models on genuine webcam faces."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.pipeline import predict_attack

DATA_DIR = Path("datasets/webcam_eval/real")
MODEL_DIR = Path("models")

# OpenCV's built-in face detector.
cascade_path = (
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load face detector.")


def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    # Reject images without exactly one detected face.
    if len(faces) != 1:
        return None

    x, y, w, h = faces[0]

    # Square crop with 10% padding.
    side = int(max(w, h) * 1.2)
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(frame.shape[1], cx + side // 2)
    y2 = min(frame.shape[0], cy + side // 2)

    return frame[y1:y2, x1:x2]


images = sorted(DATA_DIR.glob("*.jpg"))

if not images:
    raise RuntimeError("No webcam images found.")

print(f"Found {len(images)} webcam images.")

# Prepare face crops once so both models receive identical inputs.
face_crops = []

for image_path in images:
    frame = cv2.imread(str(image_path))

    if frame is None:
        continue

    face = extract_face(frame)

    if face is not None:
        face_crops.append((image_path.name, face))

print(f"Successfully detected faces in {len(face_crops)} images.")

if not face_crops:
    raise RuntimeError("No usable face crops found.")

for model_name in ("deepfake_baseline", "deepfake_finetuned"):
    model_path = MODEL_DIR / f"{model_name}.keras"

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    scores = []

    for filename, face in face_crops:
        score = predict_attack(model, face)
        scores.append(score)

    scores = np.asarray(scores)

    correct = np.sum(scores < 0.5)
    false_positives = np.sum(scores >= 0.5)

    print(f"\n{model_name.upper()}")
    print("-" * 40)
    print(f"Images evaluated: {len(scores)}")
    print(f"Correctly classified REAL: {correct}")
    print(f"Incorrectly classified FAKE: {false_positives}")
    print(f"Genuine acceptance rate: {correct / len(scores):.2%}")
    print(f"False-positive rate: {false_positives / len(scores):.2%}")
    print(f"Average fake score: {scores.mean():.4f}")
    