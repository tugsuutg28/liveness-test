"""Hardware-free collector tests. All session files live in temporary directories."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from ml.collect_liveness_v2 import collect, parse_args


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, *extra):
        return parse_args(['--split', 'train', '--label', 'real', '--session', 'session1',
                           '--lighting', 'daylight', '--background', 'office',
                           '--device', 'laptop', '--output-root', str(self.root),
                           '--count', '2', *extra])

    def camera(self, keys):
        cv = Mock()
        cv.FONT_HERSHEY_SIMPLEX = 0
        cv.VideoCapture.return_value.isOpened.return_value = True
        cv.VideoCapture.return_value.read.return_value = (True, Mock())
        cv.waitKey.side_effect = [ord(key) if key else -1 for key in keys]
        encoded = Mock()
        encoded.tobytes.return_value = b'encoded-test-frame'
        cv.imencode.return_value = (True, encoded)
        return cv

    def test_splits_and_labels(self):
        for split in ('train', 'val', 'test'):
            for label in ('real', 'fake'):
                with self.subTest(split=split, label=label):
                    args = self.args('--split', split, '--label', label,
                                     '--attack-type', 'screen' if label == 'fake' else 'none')
                    cv = self.camera(['s', '', '', '', ''])
                    result = collect(args, cv=cv, clock=Mock(side_effect=[0, 0, 1, 2, 3, 4]))
                    self.assertEqual(result, self.root / split / label / 'session1')
                    self.assertEqual(len(list(result.glob('*.jpg'))), 2)
                    meta = json.loads((result / 'metadata.json').read_text())
                    self.assertEqual(meta['status'], 'complete')
                    self.assertEqual(meta['saved_count'], 2)
                    for field in ('lighting', 'background', 'device', 'attack_type', 'split', 'label'):
                        self.assertEqual(meta[field], getattr(args, field))
                    self.assertIn('finished_at', meta)
                    cv.VideoCapture.return_value.release.assert_called_once()

    def test_countdown_and_repeated_start(self):
        cv = self.camera(['s', 's', '', 'q'])
        result = collect(self.args(), cv=cv, clock=Mock(side_effect=[0, 0, 1, 2, 3]))
        cv.imencode.assert_not_called()
        self.assertEqual(json.loads((result / 'metadata.json').read_text())['status'], 'cancelled')
        displayed = [call.args[1] for call in cv.putText.call_args_list]
        self.assertIn('Starting in 2', displayed[1])
        self.assertIn('Starting in 1', displayed[2])

    def test_existing_session_rejected_before_camera(self):
        args = self.args()
        session = self.root / 'train' / 'real' / args.session
        session.mkdir(parents=True)
        sentinel = session / 'frame_0000.jpg'
        sentinel.write_bytes(b'original')
        cv = Mock()
        with self.assertRaises(FileExistsError):
            collect(args, cv=cv)
        cv.VideoCapture.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b'original')

    def test_invalid_arguments(self):
        cases = [('--split', 'validation'), ('--label', 'spoof'), ('--session', '../escape'),
                 ('--session', '/absolute'), ('--count', '0'), ('--interval', 'nan'),
                 ('--interval', '-1'), ('--lighting', ' '), ('--label', 'fake'),
                 ('--label', 'fake', '--attack-type', ' '), ('--attack-type', 'screen')]
        for extra in cases:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*extra)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failures_record_metadata_and_release_camera(self):
        for failure in ('open', 'read', 'encode', 'write', 'interrupt'):
            with self.subTest(failure=failure):
                args = self.args('--session', failure)
                cv = self.camera(['s', ''])
                if failure == 'open':
                    cv.VideoCapture.return_value.isOpened.return_value = False
                elif failure == 'read':
                    cv.VideoCapture.return_value.read.return_value = (False, None)
                elif failure == 'encode':
                    cv.imencode.return_value = (False, None)
                elif failure == 'write':
                    cv.imencode.return_value[1].tobytes.side_effect = OSError('disk full')
                else:
                    cv.VideoCapture.return_value.read.side_effect = KeyboardInterrupt
                with self.assertRaises((RuntimeError, OSError, KeyboardInterrupt)):
                    collect(args, cv=cv, clock=Mock(side_effect=[0, 0, 3]))
                meta = json.loads((self.root / 'train' / 'real' / failure / 'metadata.json').read_text())
                self.assertEqual(meta['status'], 'error')
                self.assertEqual(meta['saved_count'], 0)
                cv.VideoCapture.return_value.release.assert_called_once()
                cv.destroyAllWindows.assert_called_once()

    def test_interval_and_partial_session(self):
        cv = self.camera(['s', '', '', '', 'q'])
        result = collect(self.args('--interval', '2'), cv=cv,
                         clock=Mock(side_effect=[0, 0, 3, 3.5, 4, 5]))
        self.assertEqual(len(list(result.glob('*.jpg'))), 1)
        self.assertEqual(json.loads((result / 'metadata.json').read_text())['saved_count'], 1)


if __name__ == '__main__':
    unittest.main()
