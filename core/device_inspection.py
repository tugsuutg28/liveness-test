"""macOS local camera listing; informational only, not trusted attestation."""
import platform, subprocess
from core.camera import classify_camera

def inspect_macos():
    if platform.system()!='Darwin':return []
    try:
        proc=subprocess.run(['system_profiler','SPCameraDataType'],capture_output=True,text=True,timeout=15)
        if proc.returncode:return []
        lines=[line.strip().removesuffix(':') for line in proc.stdout.splitlines()]
        names=[line for line in lines if line and not line.startswith(('Camera:','Model ID:','Unique ID:','Firmware','Manufacturer','Connection','Serial','Built-in')) and line.endswith(('Camera','Webcam','iPhone'))]
        return [{'name':name,'status':classify_camera(name)[0]} for name in names]
    except (OSError,subprocess.TimeoutExpired):return []
