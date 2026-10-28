import unittest
from core.fusion import fuse
from core.camera import classify_camera
class TestFusion(unittest.TestCase):
    def test_camera_block(self):
        self.assertEqual(fuse('ONE_FACE','PASS','BLOCK',.01,.01)[0],'FAIL')
    def test_deepfake_block(self):
        self.assertEqual(fuse('ONE_FACE','PASS','UNVERIFIED',.01,.9)[0],'FAIL')
    def test_photo_block(self):
        self.assertEqual(fuse('ONE_FACE','PASS','UNVERIFIED',.9,.01)[0],'FAIL')
    def test_no_models(self):
        self.assertEqual(fuse('ONE_FACE','PASS','UNVERIFIED')[0],'INCONCLUSIVE')
    def test_no_false_pass(self):
        self.assertEqual(fuse('ONE_FACE','PASS','UNVERIFIED',.01,.01)[0],'INCONCLUSIVE')
    def test_replay_warning(self):
        self.assertEqual(fuse('ONE_FACE','PASS','UNVERIFIED',.01,.01,'STATIC_WARNING')[0],'REVIEW')
if __name__=='__main__':unittest.main()
