"""HTTP regression checks for local inference controls; subprocess/signals are mocked."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

import viewer
from message_signatures import store


class ControlEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), viewer.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_root, self.old_db = viewer.ROOT, store.DB
        viewer.ROOT = self.root
        store.DB = self.root / 'db.sqlite'
        (self.root / '.venv-signatures' / 'bin').mkdir(parents=True)
        (self.root / '.venv-signatures' / 'bin' / 'python').write_text('dummy executable')
        (self.root / 'message_signatures').mkdir()
        (self.root / 'message_signatures' / 'run.py').write_text('# dummy')
        self.conn = store.connect()
        store.meta(self.conn, 'source_sha256', 'snapshot-ok')
        store.meta(self.conn, 'active_run', {
            'id': 'legacy-run', 'backend': 'torch', 'device': 'cpu',
            'latest_execution_epoch_id': 'epoch-mlX',
            'execution_epochs': [{
                'id': 'epoch-mlX', 'backend': 'mlx', 'device': 'mps',
                'outer_batch_size': 64, 'pair_batch_size': 16,
            }],
        })
        store.meta(self.conn, 'progress', {'state': 'paused', 'run_id': 'legacy-run'})
        self.conn.close()
        self.snapshot = patch.object(viewer, 'matches_snapshot', return_value=True)
        self.snapshot.start()
        self.proc = patch.object(viewer.subprocess, 'Popen')
        self.mock_popen = self.proc.start()
        self.mock_popen.return_value.pid = 8765
        self.kill = patch.object(viewer.os, 'kill')
        self.mock_kill = self.kill.start()

    def tearDown(self):
        self.kill.stop()
        self.proc.stop()
        self.snapshot.stop()
        viewer.ROOT, store.DB = self.old_root, self.old_db
        self.tmp.cleanup()

    def post(self, payload, *, origin='http://127.0.0.1:8765'):
        body = json.dumps(payload).encode()
        req = Request(f'http://127.0.0.1:{self.server.server_port}/api/signatures/control',
                      data=body, method='POST', headers={
                          'Host': '127.0.0.1:8765', 'Origin': origin,
                          'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=3) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_cross_origin_is_rejected(self):
        status, body = self.post({'action': 'resume'}, origin='https://attacker.invalid')
        self.assertEqual(status, 403)
        self.assertIn('same-origin', body['error'])
        self.mock_popen.assert_not_called()

    def test_invalid_action_is_rejected(self):
        status, body = self.post({'action': 'restart'})
        self.assertEqual(status, 400)
        self.assertIn('Unknown', body['error'])
        self.mock_popen.assert_not_called()

    def test_source_mismatch_blocks_resume(self):
        self.snapshot.stop()
        self.snapshot = patch.object(viewer, 'matches_snapshot', return_value=False)
        self.snapshot.start()
        status, body = self.post({'action': 'resume'})
        self.assertEqual(status, 409)
        self.assertIn('source index', body['error'])
        self.mock_popen.assert_not_called()

    def test_resume_uses_fixed_command_and_duplicate_resume_does_not_spawn(self):
        status, body = self.post({'action': 'resume'})
        self.assertEqual((status, body['state']), (202, 'starting'))
        args, kwargs = self.mock_popen.call_args
        self.assertEqual(args[0], [str(self.root / '.venv-signatures/bin/python'),
                                   str(self.root / 'message_signatures/run.py'),
                                   '--engine', 'nli', '--device', 'mps', '--batch', '64',
                                   '--pair-batch', '16', '--backend', 'mlx',
                                   '--continue-run', 'legacy-run'])
        self.assertEqual(kwargs['cwd'], str(self.root))
        self.assertEqual(kwargs['stdin'], viewer.subprocess.DEVNULL)
        self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
        self.assertTrue(kwargs['start_new_session'])
        status2, body2 = self.post({'action': 'resume'})
        self.assertEqual((status2, body2['state']), (200, 'already-running'))
        self.mock_popen.assert_called_once()

    def test_pause_signals_only_the_recorded_active_pid(self):
        conn = store.connect()
        try:
            store.meta(conn, 'progress', {'state': 'running', 'pid': 12345})
        finally:
            conn.close()
        status, body = self.post({'action': 'pause'})
        self.assertEqual((status, body['state']), (200, 'stopping'))
        # The endpoint probes liveness with signal 0, then sends SIGTERM.
        self.assertEqual(self.mock_kill.call_args_list, [
            unittest.mock.call(12345, 0),
            unittest.mock.call(12345, viewer.signal.SIGTERM),
        ])
        self.assertEqual({call.args[0] for call in self.mock_kill.call_args_list}, {12345})
        conn = store.connect()
        try:
            self.assertEqual(store.meta(conn, 'progress')['state'], 'stopping')
        finally:
            conn.close()

    def test_pause_when_inactive_does_not_signal(self):
        status, body = self.post({'action': 'pause'})
        self.assertEqual((status, body['state']), (200, 'already-stopped'))
        self.mock_kill.assert_not_called()


if __name__ == '__main__':
    unittest.main()
