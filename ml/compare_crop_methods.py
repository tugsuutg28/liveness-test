"""Compare square face-cropping methods without changing training data."""

from pathlib import Path

import cv2
import numpy as np

SOURCE = Path("datasets/liveness_custom")
OUTPUT = Path("reports/crop_method_comparison.jpg")

PADDING = 0.60
SIZE = 160

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Could not load face detector.")


def square_crop(frame, bbox, border_type):
    height, width = frame.shape[:2]
    x, y, w, h = map(int, bbox)

    center_x = x + w / 2
    center_y = y + h / 2

    side = max(1, round(max(w, h) * (1 + 2 * PADDING)))

    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
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

    crop = padded[top:bottom, left:right]

    return cv2.resize(
        crop,
        (SIZE, SIZE),
        interpolation=cv2.INTER_AREA
    )


def get_sample(label):
    folder = SOURCE / "train" / label

    for path in sorted(folder.rglob("*.jpg")):
        frame = cv2.imread(str(path))

        if frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        if len(faces) == 1:
            return frame, faces[0]

    raise RuntimeError(f"No usable image found for {label}")


methods = {
    "REPLICATE": cv2.BORDER_REPLICATE,
    "REFLECT": cv2.BORDER_REFLECT_101,
    "CONSTANT": cv2.BORDER_CONSTANT,
}

tiles = []

for label in ("real", "fake"):
    frame, bbox = get_sample(label)

    for method_name, border_type in methods.items():
        crop = square_crop(frame, bbox, border_type)

        tile = cv2.copyMakeBorder(
            crop,
            30, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

        cv2.putText(
            tile,
            f"{label.upper()} - {method_name}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
        )

        tiles.append(tile)

rows = [
    np.hstack(tiles[0:3]),
    np.hstack(tiles[3:6]),
]

comparison = np.vstack(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

if not cv2.imwrite(str(OUTPUT), comparison):
    raise RuntimeError("Could not save comparison image.")

print(f"Comparison saved to: {OUTPUT}")