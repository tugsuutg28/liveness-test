"""Fail-closed research fusion; cannot establish trusted webcam provenance."""
def fuse(face,challenge,camera,liveness=None,deepfake=None,replay=None,threshold=.5):
    if camera=='BLOCK':return 'FAIL','Known virtual camera label.'
    if face=='MULTIPLE_FACES' or challenge=='FAIL':return 'FAIL','Face/challenge failed.'
    if liveness is not None and liveness>=threshold:return 'FAIL','Liveness classifier flagged spoof.'
    if deepfake is not None and deepfake>=threshold:return 'FAIL','Deepfake classifier flagged manipulation.'
    if replay=='STATIC_WARNING':return 'REVIEW','Unusually static feed; not proof of replay.'
    if face!='ONE_FACE' or challenge!='PASS':return 'INCONCLUSIVE','Challenge incomplete.'
    if liveness is None or deepfake is None:return 'INCONCLUSIVE','One or both trained classifiers missing.'
    return 'INCONCLUSIVE','Camera origin and replay authenticity cannot be securely attested in browser.'
