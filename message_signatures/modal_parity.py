"""Read-only numeric parity report for the frozen Modal pilot.

This is a comparison aid, not an approval artifact. It never loads a model,
contacts Modal, or writes to the signatures database.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _manifest_digest(manifest: dict) -> str:
    payload = {k: v for k, v in manifest.items() if k != 'resultsManifestSha256'}
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _rows(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: invalid JSON') from exc
                if not isinstance(row, dict):
                    raise ValueError(f'{path}:{line_no}: expected JSON object')
                yield row


def normalize_truncation(value: dict) -> dict:
    """Reduce the local and worker schemas to comparable message/hypothesis flags."""
    if not isinstance(value, dict):
        raise ValueError('missing truncation object')
    per = value.get('perHypothesis', value.get('per_hypothesis', {}))
    if not isinstance(per, dict):
        raise ValueError('truncation per-hypothesis field must be an object')
    normalized = {}
    for act, detail in per.items():
        if not isinstance(detail, dict):
            raise ValueError(f'invalid truncation details for {act}')
        normalized[str(act)] = {
            'message': bool(detail.get('message', False)),
            'hypothesis': bool(detail.get('hypothesis', False)),
            'message_tokens_dropped': int(detail.get('message_tokens_dropped', 0)),
            'hypothesis_tokens_dropped': int(detail.get('hypothesis_tokens_dropped', 0)),
        }
    message = value.get('truncated', value.get('message', False))
    dropped_max = max((item['message_tokens_dropped'] for item in normalized.values()), default=0)
    return {'message': bool(message),
            'message_tokens_dropped_max': int(value.get('message_tokens_dropped_max', dropped_max)),
            'per_hypothesis': normalized}


def _scores_and_truncation(row: dict) -> tuple[dict[str, float], dict]:
    scores = row.get('scores')
    if scores is None:
        runtime = row.get('runtime', {})
        scores = row.get('probabilities', runtime.get('probabilities'))
    if not isinstance(scores, dict) or not scores:
        raise ValueError('row has no scores object')
    clean = {}
    for key, value in scores.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f'non-finite score for {key}')
        clean[str(key)] = number
    runtime = row.get('runtime', {})
    truncation = row.get('truncation', runtime.get('truncation'))
    return clean, normalize_truncation(truncation)


def _top(scores: dict[str, float], act_order: list[str] | None = None) -> str:
    order = {key: i for i, key in enumerate(act_order or [])}
    return min(scores, key=lambda k: (-scores[k], order.get(k, len(order)), k))


def _review(scores: dict[str, float], truncation: dict, thresholds: dict,
            blank: bool = False) -> bool:
    vals = sorted(scores.values(), reverse=True)
    margin = vals[0] - (vals[1] if len(vals) > 1 else 0.0)
    return (blank or vals[0] < float(thresholds['top_score_below']) or
            margin < float(thresholds['margin_below']) or truncation['message'])


def compare_rows(modal: dict[str, dict], local: dict[str, dict],
                 thresholds: dict, blank_by_hash: dict[str, bool] | None = None,
                 act_order: list[str] | None = None) -> dict:
    if set(modal) != set(local):
        missing = sorted(set(local) - set(modal))
        extra = sorted(set(modal) - set(local))
        raise ValueError(f'pilot hash mismatch; missing={len(missing)}, extra={len(extra)}')
    per_message = []
    deltas = []
    top_same = review_same = trunc_same = True
    for text_hash in sorted(local):
        ms, mt = _scores_and_truncation(modal[text_hash])
        ls, lt = _scores_and_truncation(local[text_hash])
        if set(ms) != set(ls):
            raise ValueError(f'act score keys differ for {text_hash}')
        absd = {k: abs(ms[k] - ls[k]) for k in ls}
        maximum = max(absd.values())
        deltas.extend(absd.values())
        modal_top, local_top = _top(ms, act_order), _top(ls, act_order)
        same_top = modal_top == local_top
        same_trunc = mt == lt
        blank = bool((blank_by_hash or {}).get(text_hash, False))
        same_review = (_review(ms, mt, thresholds, blank) == _review(ls, lt, thresholds, blank))
        top_same &= same_top
        trunc_same &= same_trunc
        review_same &= same_review
        per_message.append({'textHash': text_hash, 'maxAbsScoreDelta': maximum,
                            'topModal': modal_top, 'topLocal': local_top,
                            'topSame': same_top, 'reviewSame': same_review,
                            'truncationSame': same_trunc})
    return {
        'messageCount': len(per_message),
        'scoreCount': len(deltas),
        'maxAbsScoreDelta': max(deltas, default=0.0),
        'meanAbsScoreDelta': sum(deltas) / len(deltas) if deltas else 0.0,
        'topLabelAgreement': sum(x['topSame'] for x in per_message),
        'reviewAgreement': sum(x['reviewSame'] for x in per_message),
        'truncationAgreement': sum(x['truncationSame'] for x in per_message),
        'allTopLabelsSame': top_same, 'allReviewSame': review_same,
        'allTruncationSame': trunc_same, 'perMessage': per_message,
    }


def _load_modal_results(results_dir: Path, manifest: dict) -> tuple[dict[str, dict], dict]:
    result_manifest_path = results_dir / 'results-manifest.json'
    rm = json.loads(result_manifest_path.read_text(encoding='utf-8'))
    expected = rm.get('resultsManifestSha256')
    if not expected or expected != _manifest_digest(rm):
        raise ValueError('results-manifest digest is missing or invalid')
    if rm.get('exportId') != manifest.get('exportId') or rm.get('manifestSha256') != manifest.get('manifestSha256'):
        raise ValueError('Modal results are pinned to a different export')
    if not rm.get('complete'):
        raise ValueError('Modal results manifest is incomplete')
    if rm.get('mode') != 'pilot':
        raise ValueError('expected parity-only pilot results')
    cloud = rm.get('cloudExecution')
    if not isinstance(cloud, dict):
        raise ValueError('results manifest has no common cloud execution metadata')
    expected_cloud = {
        'backend': 'cuda', 'sourceSha256': manifest.get('sourceSha256'),
        'modelSha256': manifest.get('modelSha256'),
        'modelAssetsSha256': manifest.get('modelAssetsSha256'),
        'taxonomySha256': manifest.get('taxonomySha256'),
        'maxLength': manifest.get('maxLength'),
        'scoreSemantics': manifest.get('scoreSemantics'),
    }
    for key, expected_value in expected_cloud.items():
        if cloud.get(key) != expected_value:
            raise ValueError(f'cloud execution {key} does not match the frozen export')
    for key in ('runtimeSha256', 'codeSha256'):
        digest = cloud.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f'cloud execution has no valid {key}')
    found: dict[str, dict] = {}
    for shard in rm.get('shards', []):
        if shard.get('kind') not in ('parity_only', 'pilot'):
            continue
        shard_cloud = shard.get('cloudExecution', cloud)
        if shard_cloud != cloud:
            raise ValueError('pilot shards have inconsistent execution metadata')
        path = results_dir / shard['outputFile']
        if _sha(path) != shard.get('outputSha256'):
            raise ValueError(f"output shard digest mismatch: {shard['outputFile']}")
        rows = list(_rows(path))
        if len(rows) != int(shard['count']):
            raise ValueError(f"output shard count mismatch: {shard['outputFile']}")
        for row in rows:
            key = row.get('text_hash')
            if not isinstance(key, str) or key in found:
                raise ValueError('missing or duplicate text_hash in Modal output')
            found[key] = row
    if not found:
        raise ValueError('no pilot result rows found')
    return found, rm


def build_report(export_dir: Path, results_dir: Path,
                 local_baseline_path: Path, torch_baseline_path: Path | None = None) -> dict:
    export_manifest = json.loads((export_dir / 'manifest.json').read_text(encoding='utf-8'))
    modal, rm = _load_modal_results(results_dir, export_manifest)
    local = json.loads(local_baseline_path.read_text(encoding='utf-8'))
    thresholds = export_manifest.get('reviewThresholds', {'top_score_below': .55, 'margin_below': .15})
    blank_by_hash: dict[str, bool] = {}
    result_manifest = json.loads((results_dir / 'results-manifest.json').read_text(encoding='utf-8'))
    for shard in result_manifest.get('shards', []):
        if shard.get('kind') not in ('parity_only', 'pilot'):
            continue
        input_file = shard.get('inputFile')
        if not isinstance(input_file, str):
            continue
        input_path = export_dir / input_file
        if not input_path.exists():
            continue
        for row in _rows(input_path):
            if isinstance(row.get('text_hash'), str):
                blank_by_hash[row['text_hash']] = not str(row.get('text', '')).strip()
    act_order = [str(a['id']) for a in export_manifest.get('actSpecs', [])]
    main = compare_rows(modal, local, thresholds, blank_by_hash, act_order)

    # Independently verify the 48-message local baseline against the original
    # Torch/MPS benchmark that predates the CUDA pilot.
    torch_check = None
    if torch_baseline_path:
        torch_baseline = json.loads(torch_baseline_path.read_text(encoding='utf-8'))
        hashes = torch_baseline.get('hashes', [])
        results = torch_baseline.get('results', [])
        if len(hashes) != len(results):
            raise ValueError('Torch baseline hash/result cardinality mismatch')
        torch_rows = {h: row for h, row in zip(hashes, results)}
        pilot48 = {h: local[h] for h in hashes if h in local}
        if len(pilot48) != len(hashes):
            raise ValueError('Torch baseline hashes are not all present in local pilot baseline')
        torch_check = compare_rows(pilot48, torch_rows, thresholds, act_order=act_order)

    cloud = rm.get('cloudExecution') or {}
    run_config = export_manifest.get('activeRunConfigAtExport', {})
    report = {
        'schemaVersion': 1,
        'reportType': 'numeric-parity-only; not an approval artifact',
        'exportId': export_manifest.get('exportId'), 'runId': export_manifest.get('runId'),
        'sourceSha256': export_manifest.get('sourceSha256'),
        'modelSha256': export_manifest.get('modelSha256'),
        'modelAssetsSha256': export_manifest.get('modelAssetsSha256'),
        'taxonomySha256': export_manifest.get('taxonomySha256'),
        'localRuntimeSha256': run_config.get('runtime_sha256'),
        'localRunnerSha256': run_config.get('runner_sha256'),
        'localBackendSourceSha256': (run_config.get('execution_epochs') or [{}])[-1].get('backend_source_sha256'),
        'cloudExecution': cloud,
        'resultsManifestSha256': rm.get('resultsManifestSha256'),
        'comparison': main,
        'torchMpsBaselineCrosscheck': torch_check,
        'thresholds': {'maxAbsScoreDelta': .02,
                       'requireAllTopLabelsSame': True,
                       'requireAllReviewSame': True,
                       'requireAllTruncationSame': True},
    }
    report['recommendation'] = ('pass' if main['maxAbsScoreDelta'] <= .02 and
                                main['allTopLabelsSame'] and main['allReviewSame'] and
                                main['allTruncationSame'] else 'fail')
    report['approvalGranted'] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--local-baseline', type=Path, required=True)
    parser.add_argument('--torch-baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.export_dir, args.results_dir, args.local_baseline, args.torch_baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(args.output)
    print(json.dumps({'output': str(args.output), 'recommendation': report['recommendation'],
                      'messageCount': report['comparison']['messageCount'],
                      'maxAbsScoreDelta': report['comparison']['maxAbsScoreDelta'],
                      'approvalGranted': False}, indent=2))


if __name__ == '__main__':
    main()
