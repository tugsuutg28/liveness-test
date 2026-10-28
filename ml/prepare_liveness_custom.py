"""Prepare face crops for custom liveness fine-tuning."""

from pathlib import Path
import cv2

from ml.face_crop import crop_face

SOURCE = Path("datasets/liveness_custom")
DEST = Path("datasets/liveness_prepared")

PADDING = 0.60

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Could not load face detector.")

for split in ("train", "val"):
    for label in ("real", "fake"):
        source_dir = SOURCE / split / label
        output_dir = DEST / split / label
        output_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(source_dir.rglob("*.jpg"))

        saved = 0
        skipped = 0

        for path in files:
            frame = cv2.imread(str(path))

            if frame is None:
                skipped += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) != 1:
                skipped += 1
                continue

            face = crop_face(
                frame,
                faces[0],
                padding=PADDING
            )

            if face is None:
                skipped += 1
                continue

            face = cv2.resize(
                face,
                (160, 160),
                interpolation=cv2.INTER_AREA
            )

            # Unique filename based on session and original image.
            filename = f"{path.parent.name}_{path.stem}.jpg"

            if not cv2.imwrite(str(output_dir / filename), face):
                skipped += 1
                continue

            saved += 1

        print(
            f"{split}/{label}: "
            f"{saved} saved, {skipped} skipped"
        )

print("\nDataset preparation complete.")
print(f"Output directory: {DEST}")