"""Collect screen presentation attack images using a webcam."""

import time
from pathlib import Path

import cv2

OUTPUT_DIR = Path("datasets/webcam_eval/screen_attack")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same camera index as collect_webcam_real.py
CAMERA_INDEX = 0

TARGET_IMAGES = 50
INTERVAL_SECONDS = 1.0

camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not camera.isOpened():
    raise RuntimeError("Could not open camera.")

print("SCREEN ATTACK COLLECTION")
print("------------------------")
print("Display a face photograph on your iPhone.")
print("Hold the iPhone in front of your MacBook camera.")
print("Press S to start collecting.")
print("Press Q to quit.")

collecting = False
saved = 0
last_save = 0

try:
    while saved < TARGET_IMAGES:
        success, frame = camera.read()

        if not success:
            print("Failed to read camera frame.")
            break

        status = (
            f"Collecting: {saved}/{TARGET_IMAGES}"
            if collecting
            else "Press S to start"
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Screen Attack Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            collecting = True
            print("Collection started.")

        if collecting and time.time() - last_save >= INTERVAL_SECONDS:
            filename = OUTPUT_DIR / f"screen_{saved:03d}.jpg"

            # Read an unannotated frame for saving.
            success, clean_frame = camera.read()

            if success:
                cv2.imwrite(str(filename), clean_frame)

                saved += 1
                last_save = time.time()

                print(f"Saved {saved}/{TARGET_IMAGES}")

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"\nCollection complete: {saved} images.")
print(f"Saved in: {OUTPUT_DIR}")