"""Integration checks use an ephemeral server and temporary review ledger."""
import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import viewer


class LocalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_root = viewer.ROOT
        cls.tmp = tempfile.TemporaryDirectory()
        cls.snapshot = json.loads((cls.original_root / 'analysis/results.json').read_text())['source']['sha256']
        viewer.ROOT = Path(cls.tmp.name)
        (viewer.ROOT / 'fieldnotes').mkdir()
        (viewer.ROOT / 'fieldnotes/bundle.json').write_text(json.dumps({
            'manifest': {'source': {'sha256': cls.snapshot}},
            'observations': [{'id': 'test-only', 'revision': '1', 'snapshotId': 'test-snapshot'}]}))
        cls.server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), viewer.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        viewer.ROOT = cls.original_root
        cls.tmp.cleanup()

    def request(self, method, path, data=None, origin='http://127.0.0.1:8765'):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1])
        conn.request(method, path, json.dumps(data) if data is not None else None,
                     {'Host': '127.0.0.1:8765', 'Origin': origin, 'Content-Type': 'application/json'})
        response = conn.getresponse()
        result = (response.status, json.loads(response.read()))
        conn.close()
        return result

    def test_review_append_and_guards(self):
        payload = {'observationId': 'test-only', 'revision': '1', 'decision': 'needs-changes',
                   'reviewer': 'test fixture', 'note': 'Not a real human review; temporary test ledger.'}
        self.assertEqual(self.request('POST', '/api/fieldnotes/reviews', payload, 'https://other.example')[0], 403)
        self.assertEqual(self.request('POST', '/api/fieldnotes/reviews', dict(payload, revision='0'))[0], 400)
        self.assertEqual(self.request('POST', '/api/fieldnotes/reviews', dict(payload, note=''))[0], 400)
        first = self.request('POST', '/api/fieldnotes/reviews', payload)
        self.assertEqual(first[0], 201)
        second = self.request('POST', '/api/fieldnotes/reviews', dict(payload, decision='reject'))
        self.assertEqual(second[0], 201)
        records = self.request('GET', '/api/fieldnotes/reviews')[1]['reviews']
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0], first[1]['review'])
        self.assertFalse(records[1]['publicationApproval'])
        self.assertEqual(records[0]['snapshotId'], 'test-snapshot')
        self.assertEqual(self.request('GET', '/fieldnotes/../../README.md')[0], 404)

    def test_pinned_source_and_adjacent_context(self):
        code, data = self.request('GET', f'/api/fieldnotes/event?id=ev-{self.snapshot[:12]}-d315-e564')
        self.assertEqual(code, 200)
        self.assertEqual(data['index'], 564)
        self.assertEqual([e['index'] for e in data['context']], [562, 563, 564, 565, 566])
        self.assertEqual(data['event']['speakerName'], 'DeepSeek-V3.2')
        self.assertEqual(self.request('GET', '/api/fieldnotes/event?id=ev-wrong-d315-e564')[0], 400)
        self.assertEqual(self.request('GET', '/api/events?day=315&event=564&snapshot=wrong')[0], 409)
        code, data = self.request('GET', f'/api/events?day=315&event=564&snapshot={self.snapshot}')
        self.assertEqual(code, 200)
        self.assertEqual(data['total'], 1)
        self.assertEqual(data['events'][0]['speakerName'], 'DeepSeek-V3.2')


if __name__ == '__main__':
    unittest.main()
