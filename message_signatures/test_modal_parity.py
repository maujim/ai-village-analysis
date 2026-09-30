import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from message_signatures.modal_parity import (
    _manifest_digest, _sha, build_report, compare_rows, normalize_truncation,
)


class ModalParityTests(unittest.TestCase):
    def test_local_and_worker_truncation_shapes_normalize_including_dropped(self):
        local = {
            'message': True, 'message_tokens_dropped_max': 3,
            'per_hypothesis': {'a': {'message': True, 'message_tokens_dropped': 3,
                                     'message_tokens': 512, 'hypothesis': False}},
        }
        worker = {
            'truncated': True,
            'perHypothesis': {'a': {'message': True, 'message_tokens_dropped': 3,
                                    'message_tokens': 512, 'hypothesis': False}},
        }
        self.assertEqual(normalize_truncation(local), normalize_truncation(worker))
        worker['perHypothesis']['a']['message_tokens_dropped'] = 2
        self.assertNotEqual(normalize_truncation(local), normalize_truncation(worker))

    def test_pass_requires_all_categorical_parity_and_numeric_threshold(self):
        trunc_local = {'message': False, 'per_hypothesis': {'a': {'message': False}}}
        trunc_worker = {'truncated': False, 'perHypothesis': {'a': {'message': False}}}
        local = {'h': {'scores': {'a': .8, 'b': .1}, 'truncation': trunc_local}}
        modal = {'h': {'scores': {'a': .81, 'b': .1}, 'truncation': trunc_worker}}
        result = compare_rows(modal, local, {'top_score_below': .55, 'margin_below': .15})
        self.assertTrue(result['allTopLabelsSame'])
        self.assertTrue(result['allReviewSame'])
        self.assertTrue(result['allTruncationSame'])
        self.assertAlmostEqual(result['maxAbsScoreDelta'], .01)

        modal['h']['scores'] = {'a': .7, 'b': .71}
        result = compare_rows(modal, local, {'top_score_below': .55, 'margin_below': .15})
        self.assertFalse(result['allTopLabelsSame'])
        self.assertFalse(result['allReviewSame'])

    def test_report_loads_checked_result_shard_and_never_approves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export = root / 'export'
            results = root / 'results'
            export.mkdir(); results.mkdir()
            act_scores = {'a': .8, 'b': .1}
            local_row = {'scores': act_scores, 'runtime': {'truncation': {
                'message': False, 'per_hypothesis': {'a': {'message': False}, 'b': {'message': False}},
            }}}
            local_file = root / 'local.json'
            local_file.write_text(json.dumps({'abc': local_row}), encoding='utf-8')
            exp = {'exportId': 'e1', 'runId': 'r1', 'sourceSha256': 's', 'modelSha256': 'm',
                   'modelAssetsSha256': {}, 'taxonomySha256': 't', 'maxLength': 512,
                   'scoreSemantics': 'independent_entailment', 'reviewThresholds': {
                       'top_score_below': .55, 'margin_below': .15}, 'runtimeSettingsAtExport': {}}
            exp['manifestSha256'] = hashlib.sha256(json.dumps(
                {k: v for k, v in exp.items() if k != 'manifestSha256'}, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (export / 'manifest.json').write_text(json.dumps(exp), encoding='utf-8')
            output = results / 'parity-pilot.jsonl.gz'
            with gzip.open(output, 'wt', encoding='utf-8') as stream:
                stream.write(json.dumps({'text_hash': 'abc', 'scores': act_scores,
                                         'truncation': {'truncated': False, 'perHypothesis': {
                                             'a': {'message': False}, 'b': {'message': False}}}}) + '\n')
            shard = {'kind': 'parity_only', 'outputFile': output.name,
                     'outputSha256': _sha(output), 'count': 1}
            cloud = {'backend': 'cuda', 'sourceSha256': 's', 'modelSha256': 'm',
                     'modelAssetsSha256': {}, 'taxonomySha256': 't', 'maxLength': 512,
                     'scoreSemantics': 'independent_entailment', 'runtimeSha256': 'r' * 64,
                     'codeSha256': 'c' * 64}
            shard['cloudExecution'] = cloud
            rm = {'exportId': 'e1', 'manifestSha256': exp['manifestSha256'], 'complete': True,
                  'mode': 'pilot', 'shards': [shard], 'cloudExecution': cloud}
            rm['resultsManifestSha256'] = _manifest_digest(rm)
            (results / 'results-manifest.json').write_text(json.dumps(rm), encoding='utf-8')
            report = build_report(export, results, local_file)
            self.assertEqual(report['recommendation'], 'pass')
            self.assertFalse(report['approvalGranted'])
            self.assertEqual(report['comparison']['messageCount'], 1)


if __name__ == '__main__':
    unittest.main()
