"""HTTP regressions for the local viewer and shared site assets.

All requests run against an ephemeral server with a temporary ROOT and a tiny
synthetic transcript. No model or production source data is loaded.
"""
import gzip
import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import viewer


class ViewerRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_root = viewer.ROOT
        cls.original_transcript = viewer.TRANSCRIPT
        cls.original_index = viewer.INDEX
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        viewer.ROOT = root
        viewer.TRANSCRIPT = root / 'village-transcript.json'
        viewer.INDEX = root / '.viewer-index.json'
        viewer.SOURCE_HASH_CACHE.clear()

        (root / 'home.html').write_text('<!doctype html><main>HOME_SENTINEL</main>')
        (root / 'index.html').write_text('<!doctype html><main>VILLAGE_SENTINEL</main>')
        (root / 'analysis.html').write_text('<!doctype html><main>ANALYSIS_SENTINEL</main>')
        (root / 'shared').mkdir()
        (root / 'shared/site.css').write_text('body { color: forestgreen; }\n')
        cls.write_transcript([{'day': 1, 'date': '2026-09-01', 'events': [
            {'type': 'ORIGINAL', 'content': 'before edit'}]}])
        cls.server = viewer.ThreadingHTTPServer(('127.0.0.1', 0), viewer.Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def write_transcript(cls, days):
        viewer.TRANSCRIPT.write_text(json.dumps({'days': days}), encoding='utf-8')
        stamp = time.time_ns() + 1_000_000_000
        os.utime(viewer.TRANSCRIPT, ns=(stamp, stamp))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        viewer.ROOT = cls.original_root
        viewer.TRANSCRIPT = cls.original_transcript
        viewer.INDEX = cls.original_index
        viewer.SOURCE_HASH_CACHE.clear()
        cls.tmp.cleanup()

    def request(self, path, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=5)
        conn.request('GET', path, headers=headers or {})
        response = conn.getresponse()
        result = (response.status, dict(response.getheaders()), response.read())
        conn.close()
        return result

    def test_landing_and_village_routes(self):
        for path in ('/', '/index.html'):
            with self.subTest(path=path):
                status, headers, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn('HOME_SENTINEL', body.decode())
                self.assertEqual(headers.get('Content-Type'), 'text/html; charset=utf-8')
        for path in ('/village', '/village/'):
            with self.subTest(path=path):
                status, _, body = self.request(path)
                self.assertEqual(status, 200)
                self.assertIn('VILLAGE_SENTINEL', body.decode())
        status, _, body = self.request('/analysis.html')
        self.assertEqual(status, 200)
        self.assertIn('ANALYSIS_SENTINEL', body.decode())

    def test_legacy_deep_link_preserves_query(self):
        query = '?day=315&event=40&snapshot=abc123&speaker=Claude+3.7'
        status, headers, _ = self.request('/' + query)
        self.assertEqual(status, 302)
        self.assertEqual(headers.get('Location'), '/village/' + query)

    def test_shared_css_etag_returns_not_modified(self):
        status, headers, body = self.request('/shared/site.css')
        self.assertEqual(status, 200)
        self.assertIn('text/css', headers.get('Content-Type', ''))
        self.assertIn(b'forestgreen', body)
        etag = headers.get('ETag')
        self.assertTrue(etag)
        cached_status, cached_headers, cached_body = self.request(
            '/shared/site.css', {'If-None-Match': etag})
        self.assertEqual(cached_status, 304)
        self.assertEqual(cached_body, b'')

    def test_unknown_shared_asset_is_not_served(self):
        self.assertEqual(self.request('/shared/private.txt')[0], 404)

    def test_search_output_cannot_be_served_as_an_html_page(self):
        home = viewer.ROOT / 'home.html'
        original = home.read_bytes()
        try:
            home.write_text('transcript.json:12:<script>alert("xss")</script>')
            status, headers, body = self.request('/')
            self.assertEqual(status, 503)
            self.assertIn('text/plain', headers['Content-Type'])
            self.assertNotIn(b'<script>', body)
            home.write_bytes(b'<!doctype html>' + b' ' * (128 * 1024))
            self.assertEqual(self.request('/')[0], 503)
        finally:
            home.write_bytes(original)

    def write_large_index_source(self):
        # Substantive day metadata exceeds the server's small-response threshold.
        self.write_transcript([{'day': i, 'date': f'2026-09-{i:02d}', 'events': []}
                               for i in range(1, 41)])

    def test_days_json_is_gzipped_when_requested(self):
        self.write_large_index_source()
        status, headers, body = self.request('/api/days', {'Accept-Encoding': 'gzip'})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get('Content-Encoding'), 'gzip')
        decoded = json.loads(gzip.decompress(body))
        self.assertEqual(decoded['days'][0]['date'], '2026-09-01')
        self.assertEqual(decoded['days'][0]['count'], 0)

    def test_gzip_quality_zero_and_fractional_values(self):
        self.write_large_index_source()
        status, headers, body = self.request('/api/days', {'Accept-Encoding': 'gzip;q=0'})
        self.assertEqual(status, 200)
        self.assertNotEqual(headers.get('Content-Encoding'), 'gzip')
        self.assertEqual(json.loads(body)['days'][0]['count'], 0)

        status, headers, body = self.request('/api/days', {'Accept-Encoding': 'gzip;q=0.5'})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get('Content-Encoding'), 'gzip')
        self.assertEqual(json.loads(gzip.decompress(body))['days'][0]['count'], 0)

    def test_day_index_and_event_reads_refresh_after_source_change(self):
        status, headers, body = self.request('/api/events?day=1')
        self.assertEqual(status, 200)
        original = json.loads(body)
        self.assertEqual(original['events'][0]['type'], 'ORIGINAL')
        first_index = viewer.INDEX.read_text(encoding='utf-8')
        old_stat = viewer.TRANSCRIPT.stat()

        # Same byte length, later mtime: cache freshness must not rely on size alone.
        self.write_transcript([{'day': 1, 'date': '2026-09-01', 'events': [
            {'type': 'REVISED!', 'content': 'after edits'}]}])
        new_stat = viewer.TRANSCRIPT.stat()
        self.assertEqual(old_stat.st_size, new_stat.st_size)
        self.assertGreater(new_stat.st_mtime_ns, old_stat.st_mtime_ns)
        status, _, body = self.request('/api/events?day=1')
        self.assertEqual(status, 200)
        revised = json.loads(body)
        self.assertEqual(revised['events'][0]['type'], 'REVISED!')
        self.assertNotEqual(viewer.INDEX.read_text(encoding='utf-8'), first_index)


if __name__ == '__main__':
    unittest.main()
