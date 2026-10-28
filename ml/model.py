"""Optional frame classifier; only works after training on user-supplied labeled data."""
from pathlib import Path
import numpy as np

def load_model(path='models/anti_spoof.keras'):
    if not Path(path).exists():
        return None
    from tensorflow import keras
    return keras.models.load_model(path)

def spoof_score(model, bgr):
    import cv2
    img = cv2.resize(bgr, (160,160))[:, :, ::-1].astype(np.float32)
    img = np.expand_dims(img, 0)
    return float(model.predict(img, verbose=0).reshape(-1)[0])
