from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.pipeline import predict_attack

PADDING = 0.60

model = tf.keras.models.load_model(
    "models/liveness_baseline.keras",
    compile=False
)

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


def crop_with_border(frame, bbox, border_type):
    height, width = frame.shape[:2]
    x, y, w, h = map(int, bbox)

    cx = x + w / 2
    cy = y + h / 2

    side = max(1, round(max(w, h) * (1 + 2 * PADDING)))

    left = round(cx - side / 2)
    top = round(cy - side / 2)
    right = left + side
    bottom = top + side

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)

    padded = cv2.copyMakeBorder(
        frame,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        border_type,
        value=(127, 127, 127),
    )

    left += pad_left
    right += pad_left
    top += pad_top
    bottom += pad_top

    return padded[top:bottom, left:right]


def evaluate(folder, border_type):
    scores = []

    for path in sorted(folder.glob("*.jpg")):
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            1.1,
            5,
            minSize=(60, 60),
        )

        if len(faces) != 1:
            continue

        crop = crop_with_border(
            frame,
            faces[0],
            border_type
        )

        scores.append(predict_attack(model, crop))

    return np.asarray(scores)


methods = {
    "REPLICATE": cv2.BORDER_REPLICATE,
    "CONSTANT": cv2.BORDER_CONSTANT,
}

for method_name, border_type in methods.items():
    print(f"\n{method_name}")
    print("=" * 40)

    for category in ("real", "screen_attack"):
        folder = Path("datasets/webcam_eval") / category

        scores = evaluate(folder, border_type)

        if len(scores) == 0:
            print(f"{category}: No usable images")
            continue

        accepted = np.sum(scores < 0.5)
        detected = np.sum(scores >= 0.5)

        print(f"\n{category.upper()}")
        print(f"Samples: {len(scores)}")
        print(f"Predicted REAL: {accepted}")
        print(f"Predicted ATTACK: {detected}")
        print(f"Average attack score: {scores.mean():.4f}")

        if category == "real":
            print(f"Genuine acceptance: {accepted / len(scores):.2%}")
        else:
            print(f"Attack detection: {detected / len(scores):.2%}")
            print(f"Attack false acceptance: {accepted / len(scores):.2%}")