from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.face_crop import crop_face
from ml.pipeline import predict_attack

DATA = Path("datasets/webcam_eval")
PADDING = 0.60
THRESHOLD = 0.5

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

models = {
    "liveness": tf.keras.models.load_model(
        "models/liveness.keras",
        compile=False
    ),
    "deepfake": tf.keras.models.load_model(
        "models/deepfake_finetuned.keras",
        compile=False
    ),
}


def evaluate(category):
    folder = DATA / category
    results = []

    for path in sorted(folder.glob("*.jpg")):
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 5)

        if len(faces) != 1:
            continue

        face = crop_face(frame, faces[0], padding=PADDING)

        if face is None:
            continue

        liveness_score = predict_attack(
            models["liveness"], face
        )

        deepfake_score = predict_attack(
            models["deepfake"], face
        )

        # Accept only if BOTH models predict genuine.
        accepted = (
            liveness_score < THRESHOLD
            and deepfake_score < THRESHOLD
        )

        results.append(accepted)

    return np.asarray(results, dtype=bool)


real = evaluate("real")
attack = evaluate("screen_attack")

print("\nFUSION EVALUATION")
print("=" * 40)

print(f"Genuine samples: {len(real)}")
print(f"Attack samples: {len(attack)}")

if len(real) and len(attack):
    genuine_accepted = int(real.sum())
    attacks_accepted = int(attack.sum())

    print(
        f"Genuine acceptance: "
        f"{genuine_accepted}/{len(real)} "
        f"({real.mean():.2%})"
    )

    print(
        f"Genuine rejection: "
        f"{len(real) - genuine_accepted}/{len(real)} "
        f"({1 - real.mean():.2%})"
    )

    print(
        f"Attack detection: "
        f"{len(attack) - attacks_accepted}/{len(attack)} "
        f"({1 - attack.mean():.2%})"
    )

    print(
        f"Attack false acceptance: "
        f"{attacks_accepted}/{len(attack)} "
        f"({attack.mean():.2%})"
    )