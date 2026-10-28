from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.face_crop import crop_face
from ml.pipeline import predict_attack

DATA = Path("datasets/webcam_eval")
PADDING = 0.60

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Face detector could not be loaded.")

models = {
    "liveness": tf.keras.models.load_model(
        "models/liveness.keras",
        compile=False
    ),
    "deepfake_baseline": tf.keras.models.load_model(
        "models/deepfake_baseline.keras",
        compile=False
    ),
    "deepfake_finetuned": tf.keras.models.load_model(
        "models/deepfake_finetuned.keras",
        compile=False
    ),
}


def load_faces(folder):
    samples = []
    total = 0

    for path in sorted(folder.glob("*.jpg")):
        total += 1
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

        if len(faces) != 1:
            continue

        face = crop_face(frame, faces[0], padding=PADDING)

        if face is not None:
            samples.append(face)

    return total, samples


for category in ("real", "screen_attack"):
    total, faces = load_faces(DATA / category)

    print(f"\n{'=' * 45}")
    print(f"CATEGORY: {category.upper()}")
    print(f"Total images: {total}")
    print(f"Usable images: {len(faces)}")

    if not faces:
        continue

    for name, model in models.items():
        scores = np.asarray([
            predict_attack(model, face)
            for face in faces
        ])

        predicted_real = int(np.sum(scores < 0.5))
        predicted_attack = int(np.sum(scores >= 0.5))

        print(f"\n{name.upper()}")
        print(f"  Predicted REAL: {predicted_real}/{len(scores)}")
        print(f"  Predicted ATTACK: {predicted_attack}/{len(scores)}")
        print(f"  Average attack score: {scores.mean():.4f}")

        if category == "real":
            print(
                f"  Genuine acceptance: "
                f"{predicted_real / len(scores):.2%}"
            )
        else:
            print(
                f"  Attack detection: "
                f"{predicted_attack / len(scores):.2%}"
            )
            print(
                f"  Attack false acceptance: "
                f"{predicted_real / len(scores):.2%}"
            )