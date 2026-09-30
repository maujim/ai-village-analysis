"""Root-reviewed, exact-manifest approval and sequential import for Modal results.

No model/cloud calls are made here. Import uses cloud_transfer.import_shard,
which validates source rows and atomically commits one complete shard at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from message_signatures import cloud_transfer


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _parity_passed(report: dict) -> None:
    comparison = report.get('comparison')
    thresholds = report.get('thresholds')
    if report.get('reportType') != 'numeric-parity-only; not an approval artifact':
        raise ValueError('reviewed input is not the expected numeric parity report')
    if not isinstance(comparison, dict) or not isinstance(thresholds, dict):
        raise ValueError('parity report is missing its comparison or thresholds')
    count = int(comparison.get('messageCount', 0))
    if count < 48 or int(comparison.get('topLabelAgreement', -1)) != count or \
            int(comparison.get('reviewAgreement', -1)) != count or \
            int(comparison.get('truncationAgreement', -1)) != count:
        raise ValueError('parity report does not show complete top/review/truncation agreement')
    max_delta = float(comparison.get('maxAbsScoreDelta', float('inf')))
    if not max_delta <= .02:
        raise ValueError('parity report exceeds the approved maximum score delta')
    if (thresholds.get('maxAbsScoreDelta') != .02 or
            thresholds.get('requireAllTopLabelsSame') is not True or
            thresholds.get('requireAllReviewSame') is not True or
            thresholds.get('requireAllTruncationSame') is not True):
        raise ValueError('parity report criteria differ from the required thresholds')
    cloud = report.get('cloudExecution') or {}
    if cloud.get('backend') != 'cuda' or 'l4' not in str(cloud.get('device', '')).lower():
        raise ValueError('reviewed parity report is not from an NVIDIA L4 CUDA execution')
    if cloud.get('dtype') not in ('float16', 'torch.float16'):
        raise ValueError('L4 parity report is not FP16')
    if cloud.get('scoreSemantics') != 'independent_entailment' or cloud.get('maxLength') != 512:
        raise ValueError('parity report scoring semantics/max length do not match the frozen method')
    for key in ('runtimeSha256', 'codeSha256'):
        value = cloud.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'parity report is missing {key}')


def _assert_same_report_and_export(report: dict, export: dict) -> None:
    for key in ('runId', 'sourceSha256', 'modelSha256', 'modelAssetsSha256', 'taxonomySha256'):
        if report.get(key) != export.get(key):
            raise ValueError(f'parity report {key} differs from this export')
    if export.get('maxLength') != 512 or export.get('scoreSemantics') != 'independent_entailment':
        raise ValueError('export scoring configuration does not match the reviewed parity method')
    parity_cloud = report['cloudExecution']
    for key, export_key in (
        ('sourceSha256', 'sourceSha256'), ('modelSha256', 'modelSha256'),
        ('modelAssetsSha256', 'modelAssetsSha256'), ('taxonomySha256', 'taxonomySha256'),
        ('maxLength', 'maxLength'), ('scoreSemantics', 'scoreSemantics'),
    ):
        if parity_cloud.get(key) != export.get(export_key):
            raise ValueError(f'L4 parity execution {key} does not match this export')


def _assert_results_complete(results: dict, export: dict) -> None:
    if results.get('mode') != 'pending':
        raise ValueError('results manifest must be a pending-mode run')
    if int(results.get('selectedShardCount', 0)) < 1:
        raise ValueError('results manifest selected no pending shards')
    if bool(results.get('complete')) and (results.get('errors') or results.get('partialShards')):
        raise ValueError('results manifest claims completion while listing failures or partial shards')
    expected = {str(item['id']): item for item in export.get('pendingShards', [])}
    observed = {}
    for item in results.get('shards', []):
        shard_id = str(item.get('id'))
        if shard_id in observed or item.get('kind') not in ('pending', 'importable'):
            raise ValueError('results contain duplicate or non-pending shard output')
        observed[shard_id] = item
    if not observed or any(
            shard_id not in expected or int(item.get('count', -1)) != int(expected[shard_id]['count']) or
            item.get('inputFile') != expected[shard_id]['file'] or
            item.get('inputSha256') != expected[shard_id]['sha256']
            for shard_id, item in observed.items()):
        raise ValueError('completed outputs are empty or do not match the exported pending shards')
    if results.get('completedShardCount') != len(observed):
        raise ValueError('completed shard count does not match the result entries')
    selected = int(results['selectedShardCount'])
    if len(observed) > selected:
        raise ValueError('completed shard count exceeds selected shard count')


def make_approval(report: dict, export: dict, results: dict,
                  results_manifest_sha256: str, approved_by: str,
                  parity_report_sha256: str, approved_at: str | None = None) -> dict:
    """Create explicit approval bound to reviewed pilot and exact result execution."""
    if not approved_by.strip():
        raise ValueError('approved_by must identify the human reviewer')
    _parity_passed(report)
    _assert_same_report_and_export(report, export)
    _assert_results_complete(results, export)
    execution = results.get('cloudExecution') or {}
    pilot_execution = report['cloudExecution']
    for key in ('backend', 'dtype', 'runtimeSha256', 'codeSha256', 'modelSha256',
                'modelAssetsSha256', 'sourceSha256', 'taxonomySha256', 'maxLength',
                'scoreSemantics'):
        if execution.get(key) != pilot_execution.get(key):
            raise ValueError(f'full L4 execution {key} differs from reviewed pilot')
    if 'l4' not in str(execution.get('device', '')).lower():
        raise ValueError('full result execution is not on NVIDIA L4')
    return {
        'schemaVersion': 1, 'approved': True, 'approvedBy': approved_by.strip(),
        'approvedAt': approved_at or datetime.now(timezone.utc).isoformat(),
        'approvalBasis': 'explicit CLI approval of a passing, reviewed L4 numeric parity report',
        'parityReportSha256': parity_report_sha256,
        'pilotComparison': {
            'sampleCount': int(report['comparison']['messageCount']),
            'scoreCount': int(report['comparison']['scoreCount']),
            'maxAbsScoreDelta': float(report['comparison']['maxAbsScoreDelta']),
            'meanAbsScoreDelta': float(report['comparison']['meanAbsScoreDelta']),
            'topLabelAgreement': int(report['comparison']['topLabelAgreement']),
            'reviewAgreement': int(report['comparison']['reviewAgreement']),
            'truncationAgreement': int(report['comparison']['truncationAgreement']),
            'device': pilot_execution['device'], 'runtimeSha256': pilot_execution['runtimeSha256'],
            'codeSha256': pilot_execution['codeSha256'],
        },
        'exportId': export['exportId'], 'manifestSha256': export['manifestSha256'],
        'runId': export['runId'], 'sourceSha256': export['sourceSha256'],
        'modelSha256': export['modelSha256'], 'taxonomySha256': export['taxonomySha256'],
        'maxLength': int(export['maxLength']), 'scoreSemantics': export['scoreSemantics'],
        'executionId': execution['executionId'],
        'resultsManifestSha256': results_manifest_sha256,
    }


def load_inputs(bundle_dir: Path, results_dir: Path, parity_report_path: Path,
                approved_by: str) -> tuple[dict, list[dict], dict, str]:
    export = cloud_transfer._validate_export_bundle(bundle_dir.resolve())
    manifest_path = results_dir / 'results-manifest.json'
    if not manifest_path.is_file():
        raise ValueError('results-manifest.json is missing')
    results = json.loads(manifest_path.read_text(encoding='utf-8'))
    results_sha = cloud_transfer._result_manifest_hash(results)
    cloud_transfer._validate_result_manifest(results, export, results_dir.resolve())
    _assert_results_complete(results, export)
    for shard in results.get('shards', []):
        output_path = cloud_transfer._safe_child(results_dir.resolve(), shard.get('outputFile'))
        if _sha(output_path) != shard.get('outputSha256'):
            raise ValueError(f"completed output digest mismatch for {shard.get('id')}")
        rows = list(cloud_transfer._iter_jsonl_gzip(
            cloud_transfer._safe_child(bundle_dir.resolve(), shard['inputFile'])))
        if len(rows) != int(shard['count']):
            raise ValueError(f"export input count mismatch for {shard.get('id')}")
    parity_raw = parity_report_path.read_bytes()
    report = json.loads(parity_raw)
    approval = make_approval(report, export, results, results_sha, approved_by,
                              hashlib.sha256(parity_raw).hexdigest())
    shards = sorted(results['shards'], key=lambda item: str(item['id']))
    return export, shards, approval, results_sha


def import_completed(bundle_dir: Path, results_dir: Path, approval_path: Path,
                     db_path: Path | None = None, preserve_existing: bool = False) -> list[dict]:
    # Recheck immutable artifacts and exact approval binding before every shard.
    export = cloud_transfer._validate_export_bundle(bundle_dir.resolve())
    result_path = results_dir / 'results-manifest.json'
    results = json.loads(result_path.read_text(encoding='utf-8'))
    result_sha = cloud_transfer._result_manifest_hash(results)
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    cloud_transfer._validate_result_manifest(results, export, results_dir.resolve())
    cloud_transfer._validate_approval(approval, results, result_sha, export)
    _assert_results_complete(results, export)
    imported = []
    for shard in sorted(results['shards'], key=lambda item: str(item['id'])):
        result = cloud_transfer.import_shard(bundle_dir, results_dir, str(shard['id']),
                                             approval_path, db_path or cloud_transfer.DEFAULT_DB,
                                             preserve_existing=preserve_existing)
        imported.append(result)
        print(json.dumps({'event': 'shard_imported', **result}), flush=True)
    return imported


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--reviewed-parity-report', type=Path, required=True,
                        help='human-reviewed passing L4 parity report')
    parser.add_argument('--approval-output', type=Path, required=True)
    parser.add_argument('--approved-by', required=True,
                        help='reviewer identity; use the explicit value agreed by root')
    parser.add_argument('--db', type=Path, default=cloud_transfer.DEFAULT_DB)
    parser.add_argument('--import-completed', action='store_true',
                        help='after writing approval, import all complete pending shards')
    parser.add_argument('--preserve-existing', action='store_true',
                        help='retain predictions completed locally after export; insert only missing hashes')
    args = parser.parse_args(argv)
    export, shards, approval, results_sha = load_inputs(
        args.bundle, args.results, args.reviewed_parity_report, args.approved_by)
    args.approval_output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.approval_output.with_suffix(args.approval_output.suffix + '.tmp')
    temp.write_text(json.dumps(approval, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(args.approval_output)
    print(json.dumps({'event': 'approval_written', 'approval': str(args.approval_output),
                      'exportId': export['exportId'], 'resultsManifestSha256': results_sha,
                      'eligibleShards': len(shards), 'eligibleMessages': sum(int(s['count']) for s in shards),
                      'importStarted': bool(args.import_completed)}, indent=2), flush=True)
    if args.import_completed:
        done = import_completed(args.bundle, args.results, args.approval_output, args.db, args.preserve_existing)
        print(json.dumps({'event': 'import_finished', 'shardsImported': len(done),
                          'messagesImported': sum(int(item['imported']) for item in done),
                          'remainingUnique': done[-1]['remainingUnique'] if done else None,
                          'state': done[-1]['state'] if done else None}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
