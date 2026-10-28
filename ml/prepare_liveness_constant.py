"""Prepare face crops with constant padding for liveness fine-tuning."""

from pathlib import Path

import cv2

from ml.face_crop import crop_face


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SOURCE = Path("datasets/liveness_custom")
DEST = Path("datasets/liveness_prepared_constant")

PADDING = 0.60
IMAGE_SIZE = (160, 160)

SPLITS = ("train", "val")
LABELS = ("real", "fake")


# --------------------------------------------------
# Initialize face detector
# --------------------------------------------------

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if detector.empty():
    raise RuntimeError("Could not load OpenCV face detector.")


# --------------------------------------------------
# Prepare dataset
# --------------------------------------------------

def prepare_dataset():

    total_saved = 0
    total_skipped = 0

    for split in SPLITS:
        for label in LABELS:

            source_dir = SOURCE / split / label
            output_dir = DEST / split / label

            if not source_dir.exists():
                print(f"WARNING: Missing directory: {source_dir}")
                continue

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            files = sorted(source_dir.rglob("*.jpg"))

            saved = 0
            skipped = 0

            for path in files:

                # Read image
                frame = cv2.imread(str(path))

                if frame is None:
                    skipped += 1
                    print(f"Unreadable image: {path}")
                    continue

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                # Detect faces
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )

                # Require exactly one face
                if len(faces) != 1:
                    skipped += 1
                    continue

                # Square crop with constant gray padding
                face = crop_face(
                    frame,
                    faces[0],
                    padding=PADDING,
                    border_type=cv2.BORDER_CONSTANT
                )

                if face is None:
                    skipped += 1
                    continue

                # Resize to CNN input size
                face = cv2.resize(
                    face,
                    IMAGE_SIZE,
                    interpolation=cv2.INTER_AREA
                )

                # Create unique filename using session folder
                filename = (
                    f"{path.parent.name}_{path.stem}.jpg"
                )

                output_path = output_dir / filename

                # Save processed image
                success = cv2.imwrite(
                    str(output_path),
                    face
                )

                if not success:
                    skipped += 1
                    print(f"Failed to save: {output_path}")
                    continue

                saved += 1

            total_saved += saved
            total_skipped += skipped

            print(
                f"{split}/{label}: "
                f"{saved} saved, "
                f"{skipped} skipped"
            )

    print("\n" + "=" * 45)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 45)

    print(f"Total saved: {total_saved}")
    print(f"Total skipped: {total_skipped}")
    print(f"Output directory: {DEST}")


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    prepare_dataset()