"""Collect genuine and presentation-attack images for liveness training."""

import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--label", choices=["real", "fake"], required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--session", required=True)

    args = parser.parse_args()

    if args.count <= 0 or args.interval <= 0:
        parser.error("--count and --interval must be positive")

    output_dir = (
        Path("datasets/liveness_custom")
        / args.split
        / args.label
        / args.session
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        raise RuntimeError("Cannot open camera.")

    print(f"\nSaving to: {output_dir}")
    print("Press S to start recording.")
    print("Press Q to quit.")

    collecting = False
    saved = 0
    last_save = 0

    try:
        while saved < args.count:
            success, frame = camera.read()

            if not success:
                print("Camera read failed.")
                break

            display = frame.copy()

            cv2.putText(
                display,
                f"{args.split}/{args.label}: {saved}/{args.count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Liveness Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                collecting = True
                print("Recording started.")

            if collecting and time.monotonic() - last_save >= args.interval:
                filename = output_dir / f"frame_{saved:04d}.jpg"

                if filename.exists():
                    raise FileExistsError(
                        f"{filename} already exists. Use a new session name."
                    )

                if not cv2.imwrite(str(filename), frame):
                    raise RuntimeError(f"Could not save {filename}")

                saved += 1
                last_save = time.monotonic()
                print(f"Saved {saved}/{args.count}")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\nFinished: {saved} images saved.")


if __name__ == "__main__":
    main()