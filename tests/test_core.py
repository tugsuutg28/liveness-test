import unittest
from core.camera import classify_camera
from core.decision import decide
from core.challenges import ChallengeSession

class TestSecurity(unittest.TestCase):
    def test_virtual_camera(self):
        self.assertEqual(classify_camera('OBS Virtual Camera')[0], 'BLOCK')
    def test_unknown_camera(self):
        self.assertEqual(classify_camera('')[0], 'UNKNOWN')
    def test_no_model_never_passes(self):
        self.assertEqual(decide('ONE_FACE','PASS','UNVERIFIED',None)[0], 'INCONCLUSIVE')
    def test_strict_block(self):
        self.assertEqual(decide('ONE_FACE','PASS','BLOCK',0.01)[0], 'FAIL')
    def test_replay_flag_rejected(self):
        self.assertEqual(decide('ONE_FACE','PASS','TRUSTED',0.01,replay_clearance=False)[0], 'FAIL')
    def test_unchecked_replay_denied(self):
        self.assertEqual(decide('ONE_FACE','PASS','UNVERIFIED',0.01)[0], 'INCONCLUSIVE')
    def test_all_checks_required(self):
        self.assertEqual(decide('ONE_FACE','PASS','TRUSTED',0.01,replay_clearance=True)[0], 'PASS_WITH_LIMITATIONS')
    def test_challenge(self):
        s=ChallengeSession(steps=['blink'], started=0, timeout=50)
        self.assertEqual(s.update({'blink':True}, now=1), 'PASS')
    def test_timeout(self):
        s=ChallengeSession(steps=['blink'], started=0, timeout=5)
        self.assertEqual(s.update({'blink':False}, now=6), 'FAIL')

if __name__ == '__main__':
    unittest.main()
