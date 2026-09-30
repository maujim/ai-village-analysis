"""Provenance and explicit-continuation tests; all data stays in temp databases."""
import json
import tempfile
import unittest
from pathlib import Path

from message_signatures import store
from message_signatures import run


class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_db = store.DB
        store.DB = Path(self.tmp.name) / 'run.sqlite'
        self.conn = store.connect()
        store.meta(self.conn, 'source_sha256', 'source-a')
        self.conn.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                          ('ev-1', 1, 0, '2026-01-01', None, 'Ada', 'AGENT_TALK',
                           'source message', 'text-hash-1'))
        self.conn.commit()
        self.old_config = self.config()
        self.old_config['id'] = 'legacy-run'
        self.old_config['started_at'] = '2026-01-01T00:00:00+00:00'
        self.old_config['execution_settings'] = {'outer_batch_size': 8, 'pair_batch_size': 32}
        self.old_config['runtime_settings']['batch_size'] = 32
        self.old_config['runtime_settings']['pair_batch_size'] = 32
        store.meta(self.conn, 'active_run', self.old_config)
        store.meta(self.conn, 'progress', {'state': 'paused', 'run_id': 'legacy-run'})
        result = {'primary_candidate': 'check_report', 'scores': {'check_report': 0.8}, 'template': 'Check()'}
        self.conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                          ('legacy-run', 'text-hash-1', 'check_report', .8, .5, 0, 0, json.dumps(result)))
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        store.DB = self.old_db
        self.tmp.cleanup()

    def config(self):
        return {
            'engine': 'nli', 'model': 'model-base', 'model_sha256': 'weights-a',
            'backend': 'torch', 'backend_source_sha256': None, 'backend_version': None,
            'model_assets_sha256': {'tokenizer.json': 'tok-a', 'config.json': 'cfg-a'},
            'taxonomy_sha256': 'taxonomy-a', 'question': 'classify act',
            'runtime_sha256': 'runtime-old', 'runner_sha256': 'runner-old',
            'dependencies': {'torch': '2.0'}, 'parameter_count': 123,
            'runtime_settings': {
                'model_type': 'deberta', 'dtype': 'torch.float16', 'device': 'mps',
                'max_length': 512, 'pair_batch_size': 32, 'batch_size': 32,
                'score_semantics': 'independent_entailment',
                'label_mapping': {'0': 'entailment', '1': 'not_entailment'},
                'entailment_label': 'entailment', 'local_files_only': True,
                'trust_remote_code': False,
            },
            'score_semantics': 'independent_entailment', 'method': 'primary-act-template-v1',
            'context_used': False, 'probabilities_calibrated': False,
            'extraction_performed': False,
            'review_thresholds': {'top_score_below': .55, 'margin_below': .15},
            'source_sha256': 'source-a', 'device': 'mps',
        }

    def test_explicit_continuation_preserves_original_identity_and_adds_epoch(self):
        current = self.config()
        current.update(runtime_sha256='runtime-new', runner_sha256='runner-new',
                       dependencies={'torch': '2.1'})
        current['runtime_settings'].update(batch_size=64, pair_batch_size=64)
        continued, epoch = run.prepare_run(self.conn, current, continue_run='legacy-run',
                                           outer_batch=128, pair_batch=64,
                                           started_at='2026-02-01T00:00:00+00:00')
        self.assertEqual(continued['id'], 'legacy-run')
        self.assertEqual(continued['runner_sha256'], 'runner-old')
        self.assertEqual(continued['runtime_sha256'], 'runtime-old')
        self.assertEqual(len(continued['execution_epochs']), 2)
        legacy, new = continued['execution_epochs']
        self.assertEqual(legacy['provenance_status'], 'legacy predictions have no per-row execution epoch ID')
        self.assertEqual(legacy['predictions_missing_epoch_at_start'], 1)
        self.assertEqual(new['runner_sha256'], 'runner-new')
        self.assertEqual(new['outer_batch_size'], 128)
        self.assertEqual(new['pair_batch_size'], 64)
        self.assertEqual(new['predictions_before_epoch'], 1)
        self.assertEqual(new['resume_of_epoch_id'], legacy['id'])
        self.assertEqual(continued['latest_execution_epoch_id'], epoch['id'])
        saved = json.loads(self.conn.execute('SELECT result FROM predictions').fetchone()[0])
        self.assertNotIn('execution_epoch_id', saved)

    def test_semantic_changes_rejected_but_code_and_batch_changes_allowed(self):
        current = self.config()
        current.update(runtime_sha256='runtime-new', runner_sha256='runner-new')
        current['runtime_settings']['pair_batch_size'] = 128
        run.validate_continuation(self.old_config, current)
        current['backend'] = 'mlx'
        current['backend_source_sha256'] = 'new-backend-source'
        current['backend_version'] = '0.30.0'
        run.validate_continuation(self.old_config, current)
        for key, value in [('model_sha256', 'other-weights'),
                           ('taxonomy_sha256', 'other-taxonomy'),
                           ('source_sha256', 'other-source'), ('device', 'cpu')]:
            changed = self.config()
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'semantic settings changed'):
                run.validate_continuation(self.old_config, changed)
        changed = self.config()
        changed['runtime_settings']['max_length'] = 384
        with self.assertRaisesRegex(ValueError, 'max_length'):
            run.validate_continuation(self.old_config, changed)
        changed = self.config()
        changed['model_assets_sha256']['tokenizer.json'] = 'other-tokenizer'
        with self.assertRaisesRegex(ValueError, 'model_assets_sha256'):
            run.validate_continuation(self.old_config, changed)

    def test_json_roundtrip_label_mapping_keys_compare_equivalently(self):
        current = self.config()
        current['runtime_settings']['label_mapping'] = {0: 'entailment', 1: 'not_entailment'}
        # Persisted configs are JSON objects, so keys become strings on reload.
        self.old_config['runtime_settings']['label_mapping'] = {'0': 'entailment', '1': 'not_entailment'}
        run.validate_continuation(self.old_config, current)

    def test_epoch_records_backend_execution_source_provenance(self):
        current = self.config()
        current.update(backend='mlx', backend_source_sha256='mlx-code-hash', backend_version='0.30.0')
        _, epoch = run.prepare_run(self.conn, current, continue_run='legacy-run',
                                   outer_batch=128, pair_batch=32,
                                   started_at='2026-04-01T00:00:00+00:00')
        self.assertEqual(epoch['backend'], 'mlx')
        self.assertEqual(epoch['backend_source_sha256'], 'mlx-code-hash')
        self.assertEqual(epoch['backend_version'], '0.30.0')

    def test_new_logical_runs_have_unique_ids_and_do_not_reuse_old_rows(self):
        first, _ = run.prepare_run(self.conn, self.config(), continue_run=None,
                                  outer_batch=64, pair_batch=32,
                                  started_at='2026-03-01T00:00:00+00:00')
        second, _ = run.prepare_run(self.conn, self.config(), continue_run=None,
                                    outer_batch=64, pair_batch=32,
                                    started_at='2026-03-02T00:00:00+00:00')
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(self.conn.execute('SELECT count(*) FROM predictions').fetchone()[0], 1)
        self.assertEqual(store.meta(self.conn, 'active_run')['id'], second['id'])

    def test_continue_requires_current_logical_run_id(self):
        with self.assertRaisesRegex(ValueError, 'not the active logical run'):
            run.prepare_run(self.conn, self.config(), continue_run='unknown-run',
                            outer_batch=8, pair_batch=32)


if __name__ == '__main__':
    unittest.main()
