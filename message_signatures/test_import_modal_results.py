import unittest

from message_signatures.import_modal_results import (
    _assert_results_complete, make_approval,
)


def fixtures():
    assets = {'config.json': 'a' * 64}
    export = {
        'exportId': 'exp-2', 'manifestSha256': 'b' * 64, 'runId': 'run-1',
        'sourceSha256': 'c' * 64, 'modelSha256': 'd' * 64,
        'modelAssetsSha256': assets, 'taxonomySha256': 'e' * 64,
        'maxLength': 512, 'scoreSemantics': 'independent_entailment',
        'pendingShards': [{'id': 'pending-00', 'file': 'pending-00.jsonl.gz', 'sha256': 'a' * 64, 'count': 100},
                          {'id': 'pending-01', 'file': 'pending-01.jsonl.gz', 'sha256': 'b' * 64, 'count': 90},
                          {'id': 'pending-02', 'file': 'pending-02.jsonl.gz', 'sha256': 'c' * 64, 'count': 50}],
    }
    cloud = {
        'backend': 'cuda', 'device': 'cuda:0 (NVIDIA L4)', 'dtype': 'float16',
        'sourceSha256': export['sourceSha256'], 'modelSha256': export['modelSha256'],
        'modelAssetsSha256': assets, 'taxonomySha256': export['taxonomySha256'],
        'maxLength': 512, 'scoreSemantics': 'independent_entailment',
        'runtimeSha256': 'f' * 64, 'codeSha256': '1' * 64,
        'executionId': 'exec-3',
    }
    report = {
        'reportType': 'numeric-parity-only; not an approval artifact',
        'runId': export['runId'], 'sourceSha256': export['sourceSha256'],
        'modelSha256': export['modelSha256'], 'modelAssetsSha256': assets,
        'taxonomySha256': export['taxonomySha256'], 'cloudExecution': cloud,
        'thresholds': {'maxAbsScoreDelta': .02, 'requireAllTopLabelsSame': True,
                       'requireAllReviewSame': True, 'requireAllTruncationSame': True},
        'comparison': {'messageCount': 68, 'scoreCount': 816,
                       'maxAbsScoreDelta': .0034, 'meanAbsScoreDelta': .0002,
                       'topLabelAgreement': 68, 'reviewAgreement': 68,
                       'truncationAgreement': 68, 'allTopLabelsSame': True,
                       'allReviewSame': True, 'allTruncationSame': True},
    }
    results = {
        'complete': False, 'mode': 'pending', 'selectedShardCount': 3,
        'completedShardCount': 2, 'errors': [], 'partialShards': [],
        'shards': [{'id': 'pending-00', 'count': 100, 'kind': 'pending',
                    'inputFile': 'pending-00.jsonl.gz', 'inputSha256': 'a' * 64},
                   {'id': 'pending-01', 'count': 90, 'kind': 'pending',
                    'inputFile': 'pending-01.jsonl.gz', 'inputSha256': 'b' * 64}],
        'cloudExecution': dict(cloud),
    }
    return report, export, results


class ImportModalResultsTests(unittest.TestCase):
    def test_approves_only_exact_pilot_and_current_l4_execution(self):
        report, export, results = fixtures()
        approval = make_approval(report, export, results, '2' * 64,
                                 'root reviewed parity report', '3' * 64,
                                 '2026-09-30T12:00:00+00:00')
        self.assertEqual(approval['executionId'], 'exec-3')
        self.assertEqual(approval['resultsManifestSha256'], '2' * 64)
        self.assertEqual(approval['pilotComparison']['sampleCount'], 68)

    def test_accepts_only_completed_subset_for_bounded_wave(self):
        _, export, results = fixtures()
        _assert_results_complete(results, export)
        results['shards'][0]['count'] = 99
        with self.assertRaisesRegex(ValueError, 'do not match'):
            _assert_results_complete(results, export)

    def test_rejects_bad_parity_and_execution_fingerprint(self):
        report, export, results = fixtures()
        report['comparison']['truncationAgreement'] = 67
        with self.assertRaisesRegex(ValueError, 'agreement'):
            make_approval(report, export, results, '2' * 64, 'root', '3' * 64)
        report, export, results = fixtures()
        results['cloudExecution']['codeSha256'] = '4' * 64
        with self.assertRaisesRegex(ValueError, 'codeSha256'):
            make_approval(report, export, results, '2' * 64, 'root', '3' * 64)

    def test_rejects_incomplete_without_finished_shards(self):
        _, export, results = fixtures()
        results['shards'] = []
        results['completedShardCount'] = 0
        with self.assertRaisesRegex(ValueError, 'empty'):
            _assert_results_complete(results, export)


if __name__ == '__main__':
    unittest.main()
