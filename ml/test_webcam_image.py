import cv2
from pathlib import Path

from ml.pipeline import load_classifier, predict_attack

image_path = Path("reports/debug/webcam_face_crop.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(image_path)

for kind in ("liveness", "deepfake"):
    model = load_classifier(kind)

    if model is None:
        print(f"{kind}: MODEL NOT FOUND")
        continue

    score = predict_attack(model, image)

    print(f"{kind} fake score: {score:.4f}")