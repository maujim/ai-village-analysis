"""Temporary-database tests for offline CUDA transfer bundles and safe imports."""
import copy
import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from message_signatures import cloud_transfer as transfer


class CloudTransferTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db = self.root / 'messages.sqlite'
        self.bundle = self.root / 'bundle'
        self.results = self.root / 'results'
        self.results.mkdir()
        self.lock_path = self.root / 'run.lock'
        self.run_id = 'test-run-001'
        self.source_sha = 'source-fingerprint'
        self.model_sha = 'a' * 64
        self.taxonomy, self.taxonomy_sha = transfer._load_taxonomy()
        self.assets = {
            'config.json': 'cfg-sha',
            'tokenizer.json': 'tokenizer-sha',
            'spm.model': 'spm-sha',
        }
        self.conn = sqlite3.connect(self.db)
        self.conn.executescript('''
          CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE messages(id TEXT PRIMARY KEY, day INTEGER NOT NULL,
            event_index INTEGER NOT NULL, date TEXT, timestamp TEXT, speaker TEXT,
            source_type TEXT, text TEXT, text_hash TEXT NOT NULL,
            UNIQUE(day,event_index));
          CREATE TABLE predictions(run_id TEXT NOT NULL, text_hash TEXT NOT NULL,
            label TEXT NOT NULL, top_score REAL, margin REAL, needs_review INTEGER,
            truncated INTEGER, result TEXT NOT NULL, PRIMARY KEY(run_id,text_hash));
        ''')
        active = {
            'id': self.run_id, 'engine': 'nli', 'model': 'deberta-base-zeroshot',
            'model_sha256': self.model_sha, 'model_assets_sha256': self.assets,
            'taxonomy_sha256': self.taxonomy_sha, 'source_sha256': self.source_sha,
            'question': 'Classify the communicative act.',
            'score_semantics': 'independent_entailment', 'method': 'primary-act-template-v1',
            'context_used': False, 'extraction_performed': False,
            'probabilities_calibrated': False,
            'review_thresholds': {'top_score_below': 0.55, 'margin_below': 0.15},
            'runtime_settings': {'max_length': 512, 'device': 'mps', 'dtype': 'torch.float16'},
            'execution_epochs': [{'id': 'local-epoch', 'backend': 'mlx'}],
            'latest_execution_epoch_id': 'local-epoch',
        }
        self._meta('source_sha256', self.source_sha)
        self._meta('active_run', active)
        self._meta('progress', {'state': 'paused', 'run_id': self.run_id})
        self.texts = []
        for index in range(70):
            text = f'Unique source message {index}.'
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            self.texts.append((text_hash, text))
            self.conn.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                              (f'event-{index}', 1 + index // 35, index % 35,
                               '2026-09-01', None, f'Speaker {index % 3}',
                               'AGENT_TALK', text, text_hash))
        # One exact-text duplicate creates an extra source record but not a new
        # prediction hash, and the first text already has a local result.
        first_hash, first_text = self.texts[0]
        self.conn.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                          ('duplicate-0', 3, 0, '2026-09-02', None, 'Speaker X',
                           'USER_TALK', first_text, first_hash))
        self.conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                          (self.run_id, first_hash, 'question', .7, .2, 0, 0,
                           json.dumps({'primary_candidate': 'question', 'scores': {}})))
        self.conn.commit()
        self.pilot_hashes = [item[0] for item in self.texts[:48]]
        self.manifest = transfer.export_bundle(
            self.db, self.bundle, self.run_id, pilot_hashes=self.pilot_hashes,
            shard_size=32, pilot_extra_count=20)
        self._write_cloud_results()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def _meta(self, key, value):
        self.conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)',
                          (key, json.dumps(value, sort_keys=True)))

    def _execution(self):
        return {
            'executionId': 'modal-execution-1', 'backend': 'cuda',
            'device': 'cuda:0', 'dtype': 'float16',
            'modelSha256': self.model_sha, 'modelAssetsSha256': self.assets,
            'sourceSha256': self.source_sha, 'taxonomySha256': self.taxonomy_sha,
            'runtimeSha256': 'b' * 64, 'codeSha256': 'c' * 64,
            'dependencies': {'torch': '2.8.0', 'transformers': '5.6.0'},
            'runtimeVersion': 'modal-runtime-test', 'maxLength': 512,
            'scoreSemantics': 'independent_entailment', 'pairBatchSize': 32,
        }

    def _write_cloud_results(self):
        shards = []
        first_act = self.manifest['actSpecs'][0]['id']
        second_act = self.manifest['actSpecs'][1]['id']
        for source_shard in self.manifest['pendingShards']:
            input_path = self.bundle / source_shard['file']
            output_name = 'result-' + source_shard['file']
            output_path = self.results / output_name
            rows = []
            for source_row in transfer._iter_jsonl_gzip(input_path):
                scores = {act['id']: 0.1 for act in self.manifest['actSpecs']}
                scores[first_act] = .8
                scores[second_act] = .3
                rows.append({'text_hash': source_row['text_hash'], 'scores': scores,
                             'truncation': {'truncated': False, 'perHypothesis': {}}})
            count, output_sha = transfer._write_jsonl_gzip(output_path, rows)
            shards.append({'id': source_shard['id'], 'inputFile': source_shard['file'],
                           'inputSha256': source_shard['sha256'], 'count': count,
                           'outputFile': output_name, 'outputSha256': output_sha})
        self.result_manifest = {
            'schemaVersion': 1, 'exportId': self.manifest['exportId'],
            'manifestSha256': self.manifest['manifestSha256'],
            'runId': self.run_id, 'sourceSha256': self.source_sha,
            'modelSha256': self.model_sha, 'taxonomySha256': self.taxonomy_sha,
            'scoreSemantics': 'independent_entailment', 'maxLength': 512,
            'cloudExecution': self._execution(), 'shards': shards,
        }
        (self.results / 'results-manifest.json').write_text(
            json.dumps(self.result_manifest, indent=2) + '\n')
        self.approval_path = self.root / 'parity-approval.json'
        self._write_approval()

    def _write_approval(self):
        approval = {
            'approved': True, 'approvedBy': 'root-review', 'approvedAt': '2026-09-30T12:00:00Z',
            'exportId': self.manifest['exportId'], 'runId': self.run_id,
            'manifestSha256': self.manifest['manifestSha256'],
            'sourceSha256': self.source_sha, 'modelSha256': self.model_sha,
            'taxonomySha256': self.taxonomy_sha, 'maxLength': 512,
            'scoreSemantics': 'independent_entailment',
            'executionId': self.result_manifest['cloudExecution']['executionId'],
            'resultsManifestSha256': transfer._result_manifest_hash(self.result_manifest),
            'pilotComparison': {'sampleCount': 68, 'topLabelAgreement': 64},
        }
        self.approval_path.write_text(json.dumps(approval))

    def import_first(self):
        return transfer.import_shard(
            self.bundle, self.results, self.manifest['pendingShards'][0]['id'],
            self.approval_path, self.db, self.lock_path)

    def test_export_is_consistent_sharded_and_pilot_only(self):
        self.assertEqual(self.manifest['pendingUniqueCount'], 69)
        self.assertEqual([s['count'] for s in self.manifest['pendingShards']], [32, 32, 5])
        self.assertEqual([s['count'] for s in self.manifest['pilotShards']], [48, 68])
        self.assertTrue(all(s['importable'] for s in self.manifest['pendingShards']))
        self.assertTrue(all(not s['importable'] and s['kind'] == 'parity_only'
                            for s in self.manifest['pilotShards']))
        on_disk = json.loads((self.bundle / 'manifest.json').read_text())
        transfer._verify_manifest_hash(on_disk)
        self.assertEqual(on_disk['modelAssetsSha256'], self.assets)
        self.assertEqual(on_disk['scoreSemantics'], 'independent_entailment')
        self.assertEqual(len(on_disk['actSpecs']), 12)

    def test_import_creates_ui_compatible_rows_and_cuda_epoch(self):
        outcome = self.import_first()
        self.assertEqual(outcome['imported'], 32)
        self.assertEqual(outcome['state'], 'partial')
        first_shard = self.manifest['pendingShards'][0]
        first_text_hash = next(transfer._iter_jsonl_gzip(self.bundle / first_shard['file']))['text_hash']
        conn = sqlite3.connect(self.db)
        try:
            active = json.loads(conn.execute("SELECT value FROM metadata WHERE key='active_run'").fetchone()[0])
            progress = json.loads(conn.execute("SELECT value FROM metadata WHERE key='progress'").fetchone()[0])
            row = conn.execute('SELECT label,top_score,margin,needs_review,truncated,result FROM predictions WHERE run_id=? AND text_hash=?',
                               (self.run_id, first_text_hash))
            found = row.fetchone()
        finally:
            conn.close()
        self.assertEqual(progress['backend'], 'cuda')
        self.assertEqual(active['execution_epochs'][-1]['backend'], 'cuda')
        self.assertEqual(active['execution_epochs'][-1]['cloud_execution']['device'], 'cuda:0')
        self.assertIsNotNone(found)
        record = json.loads(found[5])
        self.assertEqual(record['score_semantics'], 'independent_entailment')
        self.assertEqual(record['primary_candidate'], self.manifest['actSpecs'][0]['id'])
        self.assertIn('candidate_templates', record)
        self.assertEqual(record['execution_epoch_id'], outcome['executionEpochId'])

    def test_import_reuses_one_epoch_for_multiple_shards(self):
        first = self.import_first()
        second = transfer.import_shard(self.bundle, self.results,
                                       self.manifest['pendingShards'][1]['id'],
                                       self.approval_path, self.db, self.lock_path)
        self.assertEqual(first['executionEpochId'], second['executionEpochId'])
        conn = sqlite3.connect(self.db)
        try:
            active = json.loads(conn.execute("SELECT value FROM metadata WHERE key='active_run'").fetchone()[0])
        finally:
            conn.close()
        self.assertEqual(len(active['execution_epochs']), 2)
        self.assertEqual(active['execution_epochs'][-1]['imported_prediction_count'], 64)

    def test_missing_or_mismatched_parity_approval_rejects_before_writes(self):
        bad = json.loads(self.approval_path.read_text())
        bad['approved'] = False
        self.approval_path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'approval is required'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_scores_must_have_all_acts_and_be_finite_bounded_numbers(self):
        output = self.results / self.result_manifest['shards'][0]['outputFile']
        rows = list(transfer._iter_jsonl_gzip(output))
        rows[0]['scores'].pop(self.manifest['actSpecs'][0]['id'])
        transfer._write_jsonl_gzip(output, rows)
        self._refresh_result_hash()
        with self.assertRaisesRegex(ValueError, 'exactly all twelve acts'):
            self.import_first()
        rows[0]['scores'] = {act['id']: .1 for act in self.manifest['actSpecs']}
        rows[0]['scores'][self.manifest['actSpecs'][0]['id']] = float('nan')
        transfer._write_jsonl_gzip(output, rows)
        self._refresh_result_hash()
        with self.assertRaisesRegex(ValueError, 'finite values'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def _refresh_result_hash(self):
        output_shard = self.result_manifest['shards'][0]
        output_shard['outputSha256'] = transfer._file_sha256(self.results / output_shard['outputFile'])
        (self.results / 'results-manifest.json').write_text(json.dumps(self.result_manifest))
        self._write_approval()

    def _prediction_count(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute('SELECT count(*) FROM predictions').fetchone()[0]
        finally:
            conn.close()

    def test_missing_duplicate_extra_result_hash_rejected(self):
        output = self.results / self.result_manifest['shards'][0]['outputFile']
        rows = list(transfer._iter_jsonl_gzip(output))
        rows.pop()
        transfer._write_jsonl_gzip(output, rows)
        self._refresh_result_hash()
        with self.assertRaisesRegex(ValueError, 'row count'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

        source_rows = list(transfer._iter_jsonl_gzip(
            self.bundle / self.manifest['pendingShards'][0]['file']))
        rows = [dict(row) for row in source_rows]
        scores = {act['id']: .1 for act in self.manifest['actSpecs']}
        rows[0]['scores'] = scores
        rows[0]['truncation'] = {'truncated': False}
        rows[1] = dict(rows[0])
        transfer._write_jsonl_gzip(output, rows)
        self._refresh_result_hash()
        with self.assertRaisesRegex(ValueError, 'duplicate text_hash'):
            self.import_first()
        rows[1]['text_hash'] = 'f' * 64
        transfer._write_jsonl_gzip(output, rows)
        self._refresh_result_hash()
        with self.assertRaisesRegex(ValueError, 'unexpected text_hash'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_source_or_active_run_change_rejected(self):
        self._meta('source_sha256', 'changed-source')
        self.conn.commit()
        with self.assertRaisesRegex(ValueError, 'source snapshot changed'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_running_state_requires_local_pause(self):
        self._meta('progress', {'state': 'running', 'run_id': self.run_id, 'pid': 123})
        self.conn.commit()
        with self.assertRaisesRegex(RuntimeError, 'must be paused'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_held_runner_lock_blocks_import(self):
        with patch.object(transfer.fcntl, 'flock', side_effect=BlockingIOError):
            with self.assertRaisesRegex(RuntimeError, 'lock is held'):
                self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_changed_cloud_model_or_non_cuda_metadata_rejected(self):
        self.result_manifest['cloudExecution']['modelSha256'] = 'd' * 64
        (self.results / 'results-manifest.json').write_text(json.dumps(self.result_manifest))
        self._write_approval()
        with self.assertRaisesRegex(ValueError, 'model fingerprint'):
            self.import_first()
        self.result_manifest['cloudExecution'] = self._execution()
        self.result_manifest['cloudExecution']['device'] = 'cpu'
        (self.results / 'results-manifest.json').write_text(json.dumps(self.result_manifest))
        self._write_approval()
        with self.assertRaisesRegex(ValueError, 'CUDA backend/device'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)

    def test_import_refuses_to_replace_predictions_from_another_local_pass(self):
        self.import_first()
        with self.assertRaisesRegex(ValueError, 'already classified locally'):
            self.import_first()

    def test_database_failure_rolls_back_the_entire_shard(self):
        source_rows = list(transfer._iter_jsonl_gzip(
            self.bundle / self.manifest['pendingShards'][0]['file']))
        fail_hash = source_rows[1]['text_hash']
        self.conn.execute(f'''CREATE TRIGGER fail_cloud_insert BEFORE INSERT ON predictions
            WHEN NEW.text_hash='{fail_hash}' BEGIN SELECT RAISE(ABORT,'injected test failure'); END''')
        self.conn.commit()
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'injected test failure'):
            self.import_first()
        self.assertEqual(self._prediction_count(), 1)


if __name__ == '__main__':
    unittest.main()
