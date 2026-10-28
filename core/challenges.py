"""Randomized gesture challenge state machine (not a cryptographic proof of liveness)."""
import random
import time
from dataclasses import dataclass, field

CHALLENGES = ('blink', 'turn_left', 'turn_right', 'open_mouth')

@dataclass
class ChallengeSession:
    steps: list[str] = field(default_factory=list)
    index: int = 0
    started: float = field(default_factory=time.monotonic)
    timeout: float = 25.0
    consecutive: int = 0
    last_blink: bool = False
    finished: bool = False
    failed: bool = False

    @classmethod
    def new(cls, count=3, rng=None):
        rng = rng or random.SystemRandom()
        return cls(steps=rng.sample(list(CHALLENGES), count))

    def update(self, signals: dict, now=None):
        now = time.monotonic() if now is None else now
        if self.finished or self.failed:
            return self.status
        if now - self.started > self.timeout:
            self.failed = True
            return self.status
        step = self.steps[self.index]
        active = bool(signals.get(step, False))
        # Blink must have a closed-eye onset; other gestures must persist briefly.
        if step == 'blink':
            accepted = active and not self.last_blink
            self.last_blink = active
        else:
            self.consecutive = self.consecutive + 1 if active else 0
            accepted = self.consecutive >= 5
        if accepted:
            self.index += 1
            self.consecutive = 0
            self.last_blink = active
            if self.index >= len(self.steps):
                self.finished = True
        return self.status

    @property
    def status(self):
        return 'FAIL' if self.failed else 'PASS' if self.finished else 'PENDING'

    @property
    def current(self):
        return self.steps[self.index] if self.index < len(self.steps) else None
