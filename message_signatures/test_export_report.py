"""Offline tests for compact run-report generation and fixed-ID watching."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from message_signatures import export_report


class ExportReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'messages.sqlite'
        self.output = self.root / 'full-run-report.json'
        self.conn = sqlite3.connect(self.db)
        self.conn.executescript('''
          CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE messages(id TEXT PRIMARY KEY, day INTEGER, event_index INTEGER,
            date TEXT, timestamp TEXT, speaker TEXT, source_type TEXT, text TEXT,
            text_hash TEXT NOT NULL, UNIQUE(day,event_index));
          CREATE TABLE predictions(run_id TEXT, text_hash TEXT, label TEXT,
            top_score REAL, margin REAL, needs_review INTEGER, truncated INTEGER,
            result TEXT, PRIMARY KEY(run_id,text_hash));
        ''')
        self.run_id = 'run-one'
        self._meta('source_sha256', 'source-hash')
        self._meta('active_run', {
            'id': self.run_id, 'engine': 'nli', 'model': 'local-test-model',
            'device': 'cpu', 'score_semantics': 'independent_entailment',
            'model_sha256': 'weights-hash', 'taxonomy_sha256': 'taxonomy-hash',
            'runtime_sha256': 'runtime-hash', 'runner_sha256': 'runner-hash',
            'backend': 'mlx', 'backend_source_sha256': 'mlx-source-hash',
            'backend_version': '0.30.0', 'execution_settings': {'outer_batch_size': 128},
            'execution_epochs': [{'id': 'epoch-one', 'backend': 'torch'},
                                 {'id': 'epoch-two', 'backend': 'mlx',
                                  'backend_source_sha256': 'mlx-source-hash'}],
            'latest_execution_epoch_id': 'epoch-two',
            'probabilities_calibrated': False, 'started_at': '2026-01-01T00:00:00Z',
        })
        self._meta('progress', {'state': 'running', 'run_id': self.run_id,
                                'completed_this_session': 36, 'unique_texts_per_second': 1.2})
        self._insert_predictions()
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _meta(self, key, value):
        self.conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key, json.dumps(value)))

    def _insert_predictions(self):
        for index in range(36):
            label = f'act-{index % 12:02d}'
            text = f'Exact source message number {index}.'
            text_hash = f'hash-{index:03d}'
            record = {
                'primary_candidate': label, 'primary_act': label,
                'score_semantics': 'independent_entailment',
                'scores': {label: 0.8, 'other': 0.2},
                'template': f'{label}(message: str) -> result: str',
                'candidate_templates': [{'act': label, 'score': 0.8,
                                         'signature': f'{label}(message: str)'}],
            }
            self.conn.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                              (f'ev-{index:03d}', 1 + index // 18, index % 18,
                               '2026-01-01', f'2026-01-01T00:{index:02d}:00Z',
                               f'Speaker {index % 3}', 'AGENT_TALK', text, text_hash))
            self.conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                              (self.run_id, text_hash, label, 0.8, 0.6, 0, 0,
                               json.dumps(record)))
        # Duplicate source rows share the same unique-text prediction.
        for extra, source_index in enumerate((0, 1)):
            base = self.conn.execute('SELECT * FROM messages WHERE id=?',
                                     (f'ev-{source_index:03d}',)).fetchone()
            self.conn.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                              (f'ev-dup-{extra}', 3, extra, '2026-01-02', None,
                               'Duplicate speaker', 'USER_TALK', base[7], base[8]))

    def _set_run(self, run_id, state):
        self._meta('active_run', {'id': run_id, 'model': f'model-{run_id}', 'engine': 'nli'})
        self._meta('progress', {'state': state, 'run_id': run_id,
                                'completed_this_session': 36})
        self.conn.commit()

    def test_compact_snapshot_has_exact_messages_and_correct_denominators(self):
        report = export_report.build_report(self.db, generated_at='fixed-time')
        self.assertEqual(report['generatedAt'], 'fixed-time')
        self.assertEqual(report['state'], 'running')
        self.assertEqual(report['source']['chatMessageRecords'], 38)
        self.assertEqual(report['source']['uniqueExactTexts'], 36)
        self.assertEqual(report['coverage']['sourceRecords']['classified'], 38)
        self.assertEqual(report['coverage']['uniqueTexts']['classified'], 36)
        self.assertEqual(report['coverage']['uniqueTexts']['remaining'], 0)
        items = report['sample']['items']
        self.assertLessEqual(len(items), 24)
        self.assertTrue(all(item['message'].startswith('Exact source message') for item in items))
        self.assertTrue(all(item['scores'] and item['template'] for item in items))
        self.assertEqual(len({item['textHash'] for item in items}), len(items))
        self.assertIn('not validated against human annotations', report['notice'])
        self.assertEqual(report['run']['backend'], 'mlx')
        self.assertEqual(report['run']['execution_epochs'][-1]['backend_source_sha256'], 'mlx-source-hash')
        self.assertEqual(report['run']['latest_execution_epoch_id'], 'epoch-two')
        self.assertNotIn('messages', report)

    def test_partial_state_is_not_promoted_to_complete_and_is_written(self):
        self._set_run(self.run_id, 'partial')
        report = export_report.export(self.db, self.output)
        self.assertEqual(report['state'], 'partial')
        self.assertEqual(json.loads(self.output.read_text())['state'], 'partial')

    def test_watch_finishes_same_run_without_real_sleep(self):
        sleeps = []

        def finish(seconds):
            sleeps.append(seconds)
            self._meta('progress', {'state': 'complete', 'run_id': self.run_id,
                                    'completed_this_session': 36})
            self.conn.commit()

        report = export_report.watch(self.db, self.output, self.run_id,
                                     poll_seconds=30, sleep_fn=finish)
        self.assertEqual(sleeps, [30])
        self.assertEqual(report['state'], 'complete')
        self.assertEqual(report['runId'], self.run_id)
        self.assertEqual(json.loads(self.output.read_text())['state'], 'complete')

    def test_watch_stops_if_active_run_is_replaced_and_keeps_original_provenance(self):
        def replace_run(_seconds):
            self._set_run('run-two', 'running')

        report = export_report.watch(self.db, self.output, self.run_id,
                                     sleep_fn=replace_run)
        self.assertEqual(report['state'], 'run_replaced')
        self.assertTrue(report['runReplaced'])
        self.assertEqual(report['runId'], self.run_id)
        self.assertEqual(report['activeRunIdAtExport'], 'run-two')
        self.assertEqual(report['run']['id'], self.run_id)
        self.assertEqual(report['progress']['run_id'], self.run_id)
        self.assertEqual(json.loads(self.output.read_text())['state'], 'run_replaced')

    def test_dead_recorded_runner_is_reported_as_interrupted(self):
        self._meta('progress', {'state': 'running', 'run_id': self.run_id, 'pid': 424242})
        self.conn.commit()
        with patch.object(export_report.os, 'kill', side_effect=ProcessLookupError) as probe:
            report = export_report.build_report(self.db)
        self.assertEqual(report['state'], 'interrupted')
        self.assertIn('no longer exists', report['stateExplanation'])
        probe.assert_called_once_with(424242, 0)

    def test_permission_denied_liveness_probe_preserves_running_state(self):
        self._meta('progress', {'state': 'running', 'run_id': self.run_id, 'pid': 424242})
        self.conn.commit()
        with patch.object(export_report.os, 'kill', side_effect=PermissionError):
            report = export_report.build_report(self.db)
        self.assertEqual(report['state'], 'running')
        self.assertIsNone(report['stateExplanation'])

    def test_complete_progress_with_remaining_rows_is_incomplete(self):
        self.conn.execute('DELETE FROM predictions WHERE run_id=? AND text_hash=?',
                          (self.run_id, 'hash-035'))
        self._meta('progress', {'state': 'complete', 'run_id': self.run_id,
                                'completed_this_session': 36})
        self.conn.commit()
        report = export_report.build_report(self.db)
        self.assertEqual(report['state'], 'incomplete')
        self.assertGreater(report['coverage']['sourceRecords']['remaining'], 0)
        self.assertIn('remain unclassified', report['stateExplanation'])

    def test_watch_terminal_failure_returns_explicit_failed_report(self):
        self._set_run(self.run_id, 'failed')
        report = export_report.watch(self.db, self.output, self.run_id,
                                     sleep_fn=lambda _: self.fail('terminal state must not sleep'))
        self.assertEqual(report['state'], 'failed')
        self.assertEqual(json.loads(self.output.read_text())['state'], 'failed')


if __name__ == '__main__':
    unittest.main()
