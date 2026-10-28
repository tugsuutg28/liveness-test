"""MediaPipe Face Mesh heuristics. Thresholds need camera-specific calibration."""
import math
import cv2
import mediapipe as mp

MESH = mp.solutions.face_mesh

def distance(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)

def eye_ratio(lm, ids):
    p = [lm[i] for i in ids]
    return (distance(p[1], p[5]) + distance(p[2], p[4])) / (2*max(distance(p[0], p[3]), 1e-6))

class FaceSignals:
    def __init__(self):
        self.mesh = MESH.FaceMesh(static_image_mode=False, max_num_faces=2,
                                  refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    def inspect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return {}, 'NO_FACE', None
        if len(result.multi_face_landmarks) != 1:
            return {}, 'MULTIPLE_FACES', None
        lm = result.multi_face_landmarks[0].landmark
        # Calculate face bounding box from MediaPipe landmarks
        height, width = frame.shape[:2]

        x_values = [p.x for p in lm]
        y_values = [p.y for p in lm]

        x1 = max(0, int(min(x_values) * width))
        y1 = max(0, int(min(y_values) * height))
        x2 = min(width, int(max(x_values) * width))
        y2 = min(height, int(max(y_values) * height))

        face_bbox = (x1, y1, x2, y2)
        left = eye_ratio(lm, [33,160,158,133,153,144])
        right = eye_ratio(lm, [362,385,387,263,373,380])
        mouth = distance(lm[13], lm[14]) / max(distance(lm[61], lm[291]), 1e-6)
        # Approximate horizontal nose displacement relative to cheek midpoint.
        mid = (lm[234].x + lm[454].x)/2
        width = max(abs(lm[454].x-lm[234].x), 1e-6)
        yaw_proxy = (lm[1].x-mid)/width
        signals = {
            'blink': (left+right)/2 < 0.19,
            'open_mouth': mouth > 0.20,
            'turn_left': yaw_proxy < -0.12,
            'turn_right': yaw_proxy > 0.12,
        }
        return signals, 'ONE_FACE', {
            'ear': round((left + right) / 2, 3),
            'mouth': round(mouth, 3),
            'yaw': round(yaw_proxy, 3),
            'bbox': face_bbox,
        }

    def close(self):
        self.mesh.close()
