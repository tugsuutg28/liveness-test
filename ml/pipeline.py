"""Shared preprocessing and model loading. 1 = attack/fake, 0 = genuine."""
from pathlib import Path
import numpy as np
SIZE = (160, 160)
MODEL_NAMES = ('liveness', 'deepfake')
def preprocess_bgr(frame):
    import cv2
    return cv2.cvtColor(cv2.resize(frame, SIZE), cv2.COLOR_BGR2RGB).astype('float32')[None, ...]
def load_classifier(kind, root=None):
    if kind not in MODEL_NAMES: raise ValueError(kind)
    path = (Path(root) if root else Path(__file__).resolve().parents[1])/'models'/f'{kind}.keras'
    if not path.exists(): return None
    import tensorflow as tf
    return tf.keras.models.load_model(path)
def predict_attack(model, frame):
    return float(np.asarray(model.predict(preprocess_bgr(frame), verbose=0)).reshape(-1)[0])
