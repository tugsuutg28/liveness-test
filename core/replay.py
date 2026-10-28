"""Conservative temporal replay warning; not a validated replay detector."""
from collections import deque
import numpy as np
class TemporalMonitor:
    def __init__(self,window=45):
        self.signatures=deque(maxlen=window)
    def update(self,frame):
        import cv2
        gray=cv2.cvtColor(cv2.resize(frame,(32,24)),cv2.COLOR_BGR2GRAY)
        self.signatures.append(gray.astype(np.float32))
        if len(self.signatures)<self.signatures.maxlen:return 'INSUFFICIENT_DATA',None
        changes=[float(np.mean(np.abs(b-a))) for a,b in zip(list(self.signatures)[:-1],list(self.signatures)[1:])]
        mean=float(np.mean(changes))
        # Static frames are suspicious but do not prove replay; normal users can sit still.
        return ('STATIC_WARNING' if mean < 0.6 else 'UNVERIFIED'),mean
