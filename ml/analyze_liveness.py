from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.face_crop import crop_face
from ml.pipeline import predict_attack

model = tf.keras.models.load_model(
    "models/liveness.keras",
    compile=False
)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def evaluate(folder):
    scores = []

    for path in sorted(Path(folder).glob("*.jpg")):
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        if len(faces) != 1:
            continue

        face = crop_face(frame, faces[0], padding=0.60)

        if face is not None:
            scores.append(predict_attack(model, face))

    return np.asarray(scores)


real = evaluate("datasets/webcam_eval/real")
attack = evaluate("datasets/webcam_eval/screen_attack")

for name, scores in [("GENUINE", real), ("SCREEN ATTACK", attack)]:
    print(f"\n{name}")
    print(f"Samples: {len(scores)}")

    if len(scores):
        print(f"Mean: {scores.mean():.4f}")
        print(f"Minimum: {scores.min():.4f}")
        print(f"Maximum: {scores.max():.4f}")
        print(f"Median: {np.median(scores):.4f}")
        print(f"10th percentile: {np.percentile(scores, 10):.4f}")
        print(f"90th percentile: {np.percentile(scores, 90):.4f}")