"""Camera name heuristics; cannot attest hardware origin of browser media streams."""
import re
SUSPICIOUS = re.compile(r'obs|virtual|manycam|snap camera|droidcam|epoccam|camo studio|ndi|mmhmm', re.I)

def classify_camera(label: str):
    if not label or not label.strip():
        return 'UNKNOWN', 'Browser did not provide a camera label; hardware provenance cannot be verified.'
    if SUSPICIOUS.search(label):
        return 'BLOCK', 'Known or suspected virtual camera name.'
    return 'UNVERIFIED', 'No known virtual-camera keyword, but physical origin is not proven.'
