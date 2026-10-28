"""Compare baseline and fine-tuned liveness models."""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from ml.face_crop import crop_face


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("datasets/webcam_eval")

MODELS = {
    "BASELINE": Path("models/liveness_baseline.keras"),
    "FINETUNED": Path("models/liveness_finetuned.keras"),
}

IMAGE_SIZE = (160, 160)
THRESHOLD = 0.50


# --------------------------------------------------
# Load face detector
# --------------------------------------------------

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Face detector could not be loaded.")


# --------------------------------------------------
# Prepare images
# --------------------------------------------------

def prepare_images(folder, border_type):

    images = []

    for path in sorted(folder.glob("*.jpg")):

        frame = cv2.imread(str(path))

        if frame is None:
            continue

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) != 1:
            continue

        face = crop_face(
            frame,
            faces[0],
            padding=0.60,
            border_type=border_type,
        )

        if face is None:
            continue

        face = cv2.resize(
            face,
            IMAGE_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )   

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        images.append(face.astype(np.float32))

    if not images:
        raise RuntimeError(
            f"No usable images found in {folder}"
        )

    return np.stack(images)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, images, label):

    scores = model.predict(
        images,
        batch_size=8,
        verbose=0,
    ).reshape(-1)

    predicted_attack = scores >= THRESHOLD

    print(f"\n{label}")
    print("-" * 35)
    print(f"Samples: {len(scores)}")
    print(f"Predicted REAL: {(~predicted_attack).sum()}")
    print(f"Predicted ATTACK: {predicted_attack.sum()}")
    print(f"Average attack score: {scores.mean():.4f}")

    if label == "REAL":
        print(
            "Genuine acceptance: "
            f"{(~predicted_attack).mean() * 100:.2f}%"
        )

    else:
        print(
            "Attack detection: "
            f"{predicted_attack.mean() * 100:.2f}%"
        )
        print(
            "Attack false acceptance: "
            f"{(~predicted_attack).mean() * 100:.2f}%"
        )


def main():

    # Compare the baseline using its original border method.
    # Compare the fine-tuned model using the constant padding
    # applied to its custom training images.
    configurations = [
        ("BASELINE", cv2.BORDER_REPLICATE),
        ("FINETUNED", cv2.BORDER_CONSTANT),
    ]

    for name, border_type in configurations:

        path = MODELS[name]

        if not path.exists():
            raise FileNotFoundError(path)

        model = tf.keras.models.load_model(
            path,
            compile=False,
        )

        print("\n" + "=" * 45)
        print(name)
        print("=" * 45)

        real_images = prepare_images(
            DATA_DIR / "real",
            border_type,
        )

        attack_images = prepare_images(
            DATA_DIR / "screen_attack",
            border_type,
        )

        evaluate(model, real_images, "REAL")
        evaluate(model, attack_images, "SCREEN_ATTACK")


if __name__ == "__main__":
    main()