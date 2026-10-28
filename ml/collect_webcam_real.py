"""Collect genuine webcam face crops for model evaluation."""

import time
from pathlib import Path

import cv2

OUTPUT = Path("datasets/webcam_eval/real")
OUTPUT.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 1
TARGET = 50
INTERVAL = 1.0

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Collecting genuine webcam samples.")
print("Use different angles, distances, and lighting.")
print("Press Q to stop.")

saved = 0
last_save = 0

try:
    while saved < TARGET:
        ok, frame = camera.read()

        if not ok:
            break

        cv2.imshow("Webcam Collection - Press Q to stop", frame)

        if time.time() - last_save >= INTERVAL:
            path = OUTPUT / f"real_{saved:03d}.jpg"

            cv2.imwrite(str(path), frame)

            saved += 1
            last_save = time.time()

            print(f"Saved {saved}/{TARGET}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"\nCollected {saved} images.")