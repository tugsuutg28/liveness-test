from pathlib import Path

import cv2
import numpy as np

from ml.evaluate_webcam import extract_face

ROOT = Path("datasets")
OUTPUT = Path("reports/dataset_comparison.jpg")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

SIZE = (220, 220)


def load_images(folder, count=3, crop=False):
    paths = sorted(
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.png"))
    )

    result = []

    for path in paths:
        image = cv2.imread(str(path))

        if image is None:
            continue

        if crop:
            image = extract_face(image)

            if image is None:
                continue

        image = cv2.resize(image, SIZE)
        result.append(image)

        if len(result) == count:
            break

    return result


groups = [
    (
        "Kaggle REAL",
        load_images(ROOT / "deepfake/train/real"),
    ),
    (
        "Kaggle FAKE",
        load_images(ROOT / "deepfake/train/fake"),
    ),
    (
        "Webcam ORIGINAL",
        load_images(ROOT / "webcam_eval/real"),
    ),
    (
        "Webcam FACE CROP",
        load_images(ROOT / "webcam_eval/real", crop=True),
    ),
]

rows = []

for label, images in groups:
    while len(images) < 3:
        images.append(np.zeros((220, 220, 3), dtype=np.uint8))

    row = np.hstack(images)

    cv2.putText(
        row,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    rows.append(row)

comparison = np.vstack(rows)

cv2.imwrite(str(OUTPUT), comparison)

print(f"Saved comparison image: {OUTPUT}")