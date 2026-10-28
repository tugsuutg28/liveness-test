# Face Liveness + Deepfake + Virtual Camera Security Lab (v2)

**Experimental university project — NOT certified biometric verification.** This version builds on the earlier strict prototype and adds **two separate trainable MobileNetV2 classifiers**, temporal motion warnings, local macOS camera inventory, evaluation scripts, and a live Streamlit dashboard.

## What is actually included

- Real-time MediaPipe face landmarks and randomized blink/turn/mouth challenges.
- Known virtual-camera name heuristics; manual browser label is **not** trustworthy proof of camera origin.
- Trainable `liveness` CNN (genuine vs photo/replay attacks) and `deepfake` CNN (genuine vs manipulated faces).
- Temporal static-feed warning. **It does not establish whether a feed is prerecorded** and should never be called a validated replay detector.
- Fail-closed decisions: missing models, unverified camera origin, or unverified replay protection cannot produce an authenticated PASS.
- Held-out evaluation with confusion matrices, precision, recall, and AUC. No pretrained weights or training datasets are bundled.

## Install (Apple Silicon Mac, Python 3.11)

Open the extracted folder in VS Code Terminal:

```bash
cd "/Users/tugs-ochir/Desktop/machine learning 1/face-liveness-security"
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

For **training**, install a compatible TensorFlow build (the following is an example; macOS wheel compatibility depends on Python and OS versions):

```bash
python -m pip install -r requirements-ml.txt
```

If TensorFlow cannot be installed, the challenge-based demo still runs without models. Browser webcam permissions must be granted. Stop the server with **Ctrl+C**.

## Prepare datasets

See `datasets/README.md`. For each of `liveness` and `deepfake`, create:

```
datasets/<kind>/train/real/   datasets/<kind>/train/fake/
datasets/<kind>/val/real/     datasets/<kind>/val/fake/
datasets/<kind>/test/real/    datasets/<kind>/test/fake/
```

**IMPORTANT:** Separate by person, original source video, and capture session **before extracting frames**. Source-video overlap causes severe leakage. Consent/licensing is required. Avoid using confidential internship data.

Example frame extraction (after choosing the split and class for the entire video):

```bash
python scripts/extract_frames.py --video /path/to/consented_video.mp4 --out datasets/deepfake/train/fake --prefix video001 --every 30
```

## Train and evaluate

```bash
python -m ml.train_classifier --kind liveness --epochs 10
python -m ml.train_classifier --kind deepfake --epochs 10
python -m ml.evaluate --kind liveness
python -m ml.evaluate --kind deepfake
```

Models are saved to `models/liveness.keras` and `models/deepfake.keras`; evaluation JSON appears in `reports/`. The classifiers use **MobileNetV2 ImageNet transfer learning**, frozen backbone, 160x160 RGB images, sigmoid output where **0 = genuine, 1 = attack**. The training-time image normalization is part of the model graph; inference passes raw RGB pixels. Threshold 0.5 is a baseline, **not a calibrated security threshold**. Independent test metrics should be reported separately for photo, replay, and face-swap attacks.

## Tests

```bash
python -m unittest discover -s tests -v
python -m scripts.camera_check
```

## Security limitations

1. Browser webcam frames do **not** include trusted hardware provenance; `system_profiler` only provides an informational device list and cannot identify which camera produced a WebRTC frame.
2. OBS or another virtual camera may be disguised; device-name blocking is bypassable.
3. A responsive real-time deepfake can satisfy randomized gestures.
4. The temporal monitor only flags unusually static scenes; it **cannot detect arbitrary replay videos**.
5. CNN accuracy on one dataset does not guarantee performance on unseen attacks or lighting conditions.
6. No trained models, evaluation metrics, or real-world security guarantees are claimed in this ZIP.
7. Webcam frames are processed in memory. The app does not save video by default. Dataset extraction is an explicit offline action.

## Suggested assignment demonstration

1. Show live landmarks/challenges and strict `INCONCLUSIVE` when models are missing.
2. Show informational virtual camera checks and the limitations of device-name heuristics.
3. Train both CNNs on consented, independently split datasets.
4. Show held-out confusion matrices and attack-specific error rates.
5. Compare real camera, photo, screen replay, OBS injection, and face swap without claiming perfect detection.

## Relationship to earlier CIFAKE work

The previous MobileNetV2 CIFAKE approach informs the architecture. **CIFAKE weights and performance do not transfer automatically to face-swap detection**. You must supply face-specific labeled data and retrain. No old notebook or weights have been silently imported.
