"""Local Streamlit demo. Run: streamlit run app.py"""

import json
from pathlib import Path
from collections import deque
import cv2
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

from core.challenges import ChallengeSession
from core.landmarks import FaceSignals
from core.camera import classify_camera
from ml.pipeline import load_classifier, predict_attack
from core.fusion import fuse
from core.replay import TemporalMonitor
from core.device_inspection import inspect_macos


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Face Liveness Security Lab",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 Face Liveness & Deepfake Security Lab")

st.caption(
    "Research prototype — NOT certified identity verification "
    "or guaranteed deepfake detection."
)

st.sidebar.success(
    "Fail-closed security policy enabled (cannot be disabled)."
)


# --------------------------------------------------
# CAMERA INFORMATION
# --------------------------------------------------

camera_name = st.sidebar.text_input(
    "Selected camera device label (manual / untrusted)",
    "",
)

cam_status, cam_explain = classify_camera(camera_name)

st.sidebar.write(f"Camera check: **{cam_status}**")
st.sidebar.caption(cam_explain)

st.sidebar.warning(
    "Browser video frames do not expose trusted hardware provenance. "
    "A typed camera name is untrusted. Unknown camera origin and "
    "untested replay protection prevent PASS."
)

st.markdown(
    "**How to test:** Allow camera access, face the webcam, "
    "and follow the instruction overlaid on the video. "
    "Refresh/restart the stream for a new randomized sequence."
)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

liveness_model = load_classifier("liveness")
deepfake_model = load_classifier("deepfake")

if liveness_model is None or deepfake_model is None:
    st.warning(
        "One or more trained models are missing. "
        "The app can demonstrate challenges, "
        "but cannot produce an authenticated PASS."
    )
else:
    st.success(
        "Both experimental classifiers loaded. "
        "Scores require independent validation and calibration."
    )

st.sidebar.subheader("Model readiness")

st.sidebar.write(
    "Liveness CNN:",
    "LOADED" if liveness_model is not None else "NOT TRAINED",
)

st.sidebar.write(
    "Deepfake CNN:",
    "LOADED" if deepfake_model is not None else "NOT TRAINED",
)

with st.expander("macOS camera inventory (informational only)"):
    st.write(
        inspect_macos()
        or "No camera names available from system_profiler."
    )


# --------------------------------------------------
# FACE CROPPING
# --------------------------------------------------

def crop_face(frame, bbox, padding=0.20):
    """
    Crop a square region around the detected face.
    Returns a BGR image or None.
    """
    if bbox is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    face_width = x2 - x1
    face_height = y2 - y1

    if face_width <= 0 or face_height <= 0:
        return None

    # Calculate square crop size
    side = max(
        1,
        int(round(max(face_width, face_height) * (1.0 + padding)))
    )

    if side > w or side > h:
        return None

    # Calculate face center
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Calculate crop coordinates
    x_start = int(round(cx - side / 2.0))
    y_start = int(round(cy - side / 2.0))

    # Keep the square inside the camera frame
    x_start = max(0, min(x_start, w - side))
    y_start = max(0, min(y_start, h - side))

    x_end = x_start + side
    y_end = y_start + side

    # Extract face
    face_crop = frame[y_start:y_end, x_start:x_end]

    if face_crop.shape[:2] != (side, side):
        return None

    return face_crop


# --------------------------------------------------
# WEBCAM PROCESSOR
# --------------------------------------------------

class Processor(VideoProcessorBase):

    def __init__(self):

        self.detector = FaceSignals()

        self.session = ChallengeSession.new()

        self.frames = 0

        self.last_liveness = None
        self.liveness_history = deque(maxlen=10)
        self.last_deepfake = None

        self.temporal = TemporalMonitor()

        self.replay_status = "INSUFFICIENT_DATA"

        self.last_face = "NO_FACE"

        self.last_result = (
            "INCONCLUSIVE",
            "Waiting for camera frames.",
        )

    def recv(self, frame):

        # Convert WebRTC frame to OpenCV BGR.
        img = frame.to_ndarray(format="bgr24")

        # Detect face and facial landmarks.
        signals, face, metrics = self.detector.inspect(img)

        self.last_face = face

        # Update challenge session.
        if face == "ONE_FACE":
            self.session.update(signals)

        self.frames += 1

        # Temporal replay monitoring.
        self.replay_status, motion = self.temporal.update(img)

        # ------------------------------------------
        # CNN PREDICTIONS
        # ------------------------------------------

        # Run CNN inference every 20 frames.
        if self.frames % 20 == 0:

            if (
                face == "ONE_FACE"
                and metrics is not None
                and "bbox" in metrics
            ):

                try:

                    # Extract face coordinates.
                    bbox = metrics["bbox"]

                    # Crop face instead of entire frame.
                    face_crop = crop_face(
                        img,
                        bbox,
                        padding=0.60,
                    )

                    if face_crop is None:

                        self.last_liveness = None
                        self.last_deepfake = None

                    else:

                        # ---------------------------------------
                        # DEBUG: Save webcam face crop
                        # ---------------------------------------
                        import time
                        from pathlib import Path

                        if not hasattr(self, "last_crop_save_time"):
                            self.last_crop_save_time = 0

                        if time.time() - self.last_crop_save_time >= 10:
                            debug_dir = Path("reports/debug")
                            debug_dir.mkdir(parents=True, exist_ok=True)

                            cv2.imwrite(
                                str(debug_dir / "webcam_face_crop.jpg"),
                                face_crop,
                            )

                            self.last_crop_save_time = time.time()

                        # ---------------------------------------
                        # Liveness prediction
                        # ---------------------------------------
                        if liveness_model is not None:
                            self.last_liveness = predict_attack(
                                liveness_model,
                                face_crop,
                            )
                            self.liveness_history.append(
                                self.last_liveness
                            )

                        # ---------------------------------------
                        # Deepfake prediction
                        # ---------------------------------------
                        if deepfake_model is not None:
                            self.last_deepfake = predict_attack(
                                deepfake_model,
                                face_crop,
                            )

                except Exception:
                    self.last_liveness = None
                    self.last_deepfake = None

            else:

                # Never reuse scores when a valid
                # face is unavailable.
                self.last_liveness = None
                self.last_deepfake = None

        # Clear stale scores immediately when the
        # face disappears, not just every 20 frames.
        if face != "ONE_FACE":
            self.last_liveness = None
            self.last_deepfake = None

        # ------------------------------------------
        # FINAL SECURITY DECISION
        # ------------------------------------------

        self.last_result = fuse(
            face,
            self.session.status,
            cam_status,
            self.last_liveness,
            self.last_deepfake,
            self.replay_status,
        )

        # ------------------------------------------
        # DISPLAY INFORMATION
        # ------------------------------------------

        challenge = self.session.current or "Completed"

                # Calculate the average of the last 10 predictions
        average_liveness = (
            sum(self.liveness_history) / len(self.liveness_history)
            if self.liveness_history
            else None
        )

        rows = [
            f"Face: {face}",

            (
                f"Challenge: {challenge} "
                f"({self.session.index}/{len(self.session.steps)})"
            ),

            f"Challenge status: {self.session.status}",

            f"Decision: {self.last_result[0]}",

            (
                f"Raw fake score: {self.last_liveness:.3f}"
                if self.last_liveness is not None
                else "Raw fake score: N/A"
            ),

            (
                f"Average fake score: {average_liveness:.3f}"
                if average_liveness is not None
                else "Average fake score: N/A"
            ),

            (
                f"Deepfake score: {self.last_deepfake:.3f}"
                if self.last_deepfake is not None
                else "Deepfake CNN: NOT TRAINED"
            ),

            f"Temporal: {self.replay_status}",
        ]

        for i, line in enumerate(rows):

            cv2.putText(
                img,
                line,
                (12, 30 + 29 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 230, 230),
                2,
            )

        return __import__("av").VideoFrame.from_ndarray(
            img,
            format="bgr24",
        )

    def __del__(self):
        try:
            self.detector.close()
        except Exception:
            pass


# --------------------------------------------------
# START WEBCAM
# --------------------------------------------------

ctx = webrtc_streamer(
    key="liveness",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Processor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)


# --------------------------------------------------
# SECURITY INFORMATION
# --------------------------------------------------

st.error(
    "**Strict policy:** Known virtual camera or model-flagged "
    "attack → FAIL. Missing models or unverified camera origin "
    "→ INCONCLUSIVE (access denied). Temporal static-feed "
    "warning → REVIEW. Browser camera origin cannot be "
    "authenticated; this demo never grants a verified PASS."
)

st.subheader("Evaluation checklist")

st.write(
    "Test: real webcam, printed photo, replayed video, "
    "OBS virtual camera, and face-swap deepfake. "
    "Use subject/video-disjoint data for evaluation."
)

st.subheader("Privacy")

st.write(
    "The demo processes frames in memory. "
    "It does not automatically save webcam video or "
    "biometric templates. Model training requires "
    "appropriately consented data."
)


# --------------------------------------------------
# TRAINING DASHBOARD
# --------------------------------------------------

st.subheader("Training and experiment dashboard")

st.code(
    "python -m ml.train_classifier --kind liveness\n"
    "python -m ml.train_classifier --kind deepfake\n"
    "python -m ml.evaluate --kind liveness\n"
    "python -m ml.evaluate --kind deepfake",
    language="bash",
)

for kind in ("liveness", "deepfake"):

    report = Path("reports") / f"{kind}_evaluation.json"

    if report.exists():

        with st.expander(
            f"{kind.title()} held-out evaluation"
        ):
            st.json(
                json.loads(report.read_text())
            )

    else:

        st.caption(
            f"{kind.title()} evaluation: "
            "no independently generated report yet."
        )