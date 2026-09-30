"""Source-independent regression checks for the local signature store."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from message_signatures import store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = store.DB
        store.DB = Path(self.tmp.name) / 'test.sqlite'
        self.conn = store.connect()
        text = 'same exact utterance'
        digest = hashlib.sha256(text.encode()).hexdigest()
        self.digest = digest
        self.conn.executemany(
            'INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
            [('ev-a', 1, 2, '2026-01-01', '2026-01-01T00:00:00Z', 'Ada', 'AGENT_TALK', text, digest),
             ('ev-b', 1, 3, '2026-01-01', '2026-01-01T00:00:01Z', 'Bo', 'USER_TALK', text, digest),
             ('ev-c', 2, 1, '2026-01-02', '2026-01-02T00:00:00Z', 'Ada', 'AGENT_TALK', 'other', hashlib.sha256(b'other').hexdigest())])
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        store.DB = self.old_db
        self.tmp.cleanup()

    def activate(self, run_id):
        store.meta(self.conn, 'active_run', {'id': run_id})

    def add_prediction(self, run_id, text_hash, label='A'):
        result = {'scores': {label: 1.0}, 'primary_candidate': label}
        self.conn.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                          (run_id, text_hash, label, 1.0, 1.0, 0, 0, json.dumps(result)))
        self.conn.commit()

    def test_duplicate_text_reuses_prediction_but_counts_both_source_rows(self):
        self.activate('r1')
        self.add_prediction('r1', self.digest, 'A')
        self.add_prediction('r1', self.digest, 'B')
        self.assertEqual(self.conn.execute('SELECT count(*) FROM predictions').fetchone()[0], 1)
        self.assertEqual(self.conn.execute('SELECT label FROM predictions').fetchone()[0], 'A')
        rows = store.query({'limit': '10'})
        self.assertEqual([r['id'] for r in rows['messages']], ['ev-a', 'ev-b', 'ev-c'])
        self.assertEqual([r['label'] for r in rows['messages']], ['A', 'A', None])
        s = store.status(self.conn)
        self.assertEqual((s['total'], s['classified'], s['remaining']), (3, 2, 1))

    def test_filters_and_pending(self):
        self.activate('r1')
        self.add_prediction('r1', self.digest, 'A')
        self.assertEqual([r['id'] for r in store.query({'day': '1'})['messages']], ['ev-a', 'ev-b'])
        self.assertEqual([r['id'] for r in store.query({'speaker': 'Bo'})['messages']], ['ev-b'])
        self.assertEqual([r['id'] for r in store.query({'label': 'A'})['messages']], ['ev-a', 'ev-b'])
        self.assertEqual([r['id'] for r in store.query({'pending': '1'})['messages']], ['ev-c'])
        self.assertEqual([r['id'] for r in store.query({'q': 'OTHER'})['messages']], ['ev-c'])

    def test_predictions_are_isolated_by_active_run(self):
        self.add_prediction('r1', self.digest)
        self.activate('r2')
        self.assertEqual(store.status(self.conn)['classified'], 0)
        self.assertEqual(len(store.query({'pending': '1'})['messages']), 3)
        self.add_prediction('r2', self.digest, 'B')
        self.assertEqual(store.status(self.conn)['classified'], 2)
        self.activate('r1')
        self.assertEqual(store.status(self.conn)['classified'], 2)
        labels = {r['label'] for r in store.query({})['messages'] if r['label']}
        self.assertEqual(labels, {'A'})

    def test_limit_bounds_and_malformed_numbers(self):
        self.activate('r1')
        self.assertEqual(store.query({'limit': '1000'})['limit'], 100)
        self.assertEqual(store.query({'limit': '-4'})['limit'], 1)
        self.assertEqual(store.query({'offset': '-9'})['offset'], 0)
        with self.assertRaises(ValueError):
            store.query({'limit': 'not-a-number'})
        with self.assertRaises(ValueError):
            store.query({'offset': 'not-a-number'})


if __name__ == '__main__':
    unittest.main()
