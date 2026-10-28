"""Fail-closed experimental policy. A model score alone does not prove camera authenticity."""
def decide(face_status, challenge_status, camera_status, spoof_score=None,
           strict=True, threshold=0.5, replay_clearance=None):
    """replay_clearance must be True only after an independently validated replay check.

    None means no replay detector is available, False means replay was detected.
    Never claim a real-user PASS without trusted camera origin and replay clearance.
    """
    if camera_status == 'BLOCK':
        return 'FAIL', 'Known/suspected virtual camera blocked.'
    if face_status == 'MULTIPLE_FACES':
        return 'FAIL', 'Multiple faces detected.'
    if challenge_status == 'FAIL':
        return 'FAIL', 'Liveness challenge failed or timed out.'
    if spoof_score is not None and spoof_score >= threshold:
        return 'FAIL', 'Experimental classifier flagged possible spoof/deepfake.'
    if replay_clearance is False:
        return 'FAIL', 'Replay attack flagged.'
    if face_status != 'ONE_FACE' or challenge_status != 'PASS':
        return 'INCONCLUSIVE', 'Face or challenge checks incomplete.'
    if spoof_score is None:
        return 'INCONCLUSIVE', 'No trained anti-spoof/deepfake classifier available.'
    if replay_clearance is not True:
        return 'INCONCLUSIVE', 'Replay detection unavailable; cannot approve.'
    if camera_status != 'TRUSTED':
        return 'INCONCLUSIVE', 'Physical camera origin is not verified.'
    return 'PASS_WITH_LIMITATIONS', 'Experimental checks passed; no absolute authenticity guarantee.'
