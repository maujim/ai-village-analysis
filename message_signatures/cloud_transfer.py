"""Local, auditable handoff of NLI work to an external CUDA executor.

This module performs no network calls, cloud authentication, or model loading.
Exports contain text and must be handled as sensitive source data. Imports only
accept a root-reviewed numeric-parity approval and never replace predictions.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from message_signatures import store

HERE = Path(__file__).resolve().parent
DEFAULT_DB = store.DB
DEFAULT_SHARD_SIZE = 256
ALLOWED_IDLE_STATES = {'paused', 'interrupted', 'partial', 'stopped'}
SCORE_SEMANTICS = 'independent_entailment'
MAX_LENGTH = 512
METHOD = 'primary-act-template-v1'
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
MAX_SHARD_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_meta(conn: sqlite3.Connection, key: str) -> Any:
    row = conn.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _write_meta_uncommitted(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute('INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)',
                 (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))


def _open_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    conn = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    return conn


def _load_taxonomy(taxonomy_path: str | Path | None = None) -> tuple[dict, str]:
    path = Path(taxonomy_path) if taxonomy_path else HERE / 'taxonomy.json'
    raw = path.read_bytes()
    taxonomy = json.loads(raw)
    acts = taxonomy.get('acts')
    if not isinstance(acts, list) or len(acts) != 12:
        raise ValueError('taxonomy must contain exactly twelve acts')
    ids = [a.get('id') for a in acts]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != 12:
        raise ValueError('taxonomy act IDs must be twelve unique strings')
    return taxonomy, hashlib.sha256(raw).hexdigest()


def _manifest_payload(manifest: dict) -> dict:
    return {key: value for key, value in manifest.items() if key != 'manifestSha256'}


def _verify_manifest_hash(manifest: dict) -> None:
    expected = manifest.get('manifestSha256')
    actual = _object_sha256(_manifest_payload(manifest))
    if not isinstance(expected, str) or expected != actual:
        raise ValueError('manifest SHA-256 mismatch')


def _write_jsonl_gzip(path: Path, rows: Iterable[dict]) -> tuple[int, str]:
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as zipped:
            count = 0
            for row in rows:
                zipped.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                zipped.write(b'\n')
                count += 1
    return count, _file_sha256(path)


def _iter_jsonl_gzip(path: Path):
    total_bytes = 0
    with gzip.open(path, 'rb') as stream:
        line_number = 0
        while True:
            raw_line = stream.readline(MAX_JSONL_LINE_BYTES + 1)
            if not raw_line:
                break
            line_number += 1
            total_bytes += len(raw_line)
            if len(raw_line) > MAX_JSONL_LINE_BYTES or total_bytes > MAX_SHARD_UNCOMPRESSED_BYTES:
                raise ValueError(f'{path.name}: JSONL shard exceeds safety size limits')
            try:
                line = raw_line.decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError(f'{path.name}:{line_number}: invalid UTF-8') from exc
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path.name}:{line_number}: invalid JSON') from exc
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}:{line_number}: expected an object')
            yield row


def _safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError('result file path is missing')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('result manifest contains a path outside its bundle')
    return path


def _asset_subset(assets: dict) -> dict:
    return {key: value for key, value in assets.items()
            if key.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'spm'))}


def _pilot_hashes_from_reference(path: Path | None = None) -> list[str]:
    candidate = path or HERE / 'further-tuning.json'
    if candidate.is_file():
        data = json.loads(candidate.read_text(encoding='utf-8'))
        values = data.get('fixedSample', {}).get('messageTextSha256', [])
        if isinstance(values, list) and len(values) >= 48 and len(set(values)) == len(values):
            return values[:48]
    return []


def export_bundle(db_path: str | Path = DEFAULT_DB,
                  output_dir: str | Path | None = None,
                  run_id: str | None = None,
                  taxonomy_path: str | Path | None = None,
                  shard_size: int = DEFAULT_SHARD_SIZE,
                  pilot_hashes: list[str] | None = None,
                  pilot_extra_count: int = 20) -> dict:
    """Write a consistent run snapshot and pending-text shards to a new directory."""
    if shard_size < 1 or pilot_extra_count < 0:
        raise ValueError('shard_size must be positive and pilot_extra_count nonnegative')
    taxonomy, taxonomy_sha = _load_taxonomy(taxonomy_path)
    conn = _open_readonly(db_path)
    temporary = None
    try:
        conn.execute('BEGIN')
        active = _read_meta(conn, 'active_run') or {}
        selected_run_id = run_id or str(active.get('id') or '')
        if not selected_run_id or selected_run_id != str(active.get('id') or ''):
            raise ValueError('requested run ID is not the active logical run')
        source_sha = _read_meta(conn, 'source_sha256')
        if not source_sha or source_sha != active.get('source_sha256'):
            raise ValueError('active run and indexed source snapshot do not match')
        if active.get('engine') != 'nli' or active.get('score_semantics') != SCORE_SEMANTICS:
            raise ValueError('cloud transfer supports only the current NLI independent-entailment run')
        if active.get('taxonomy_sha256') != taxonomy_sha:
            raise ValueError('active run taxonomy fingerprint does not match local taxonomy.json')
        settings = active.get('runtime_settings') or {}
        if settings.get('max_length') != MAX_LENGTH:
            raise ValueError('cloud transfer requires max_length=512 to preserve truncation semantics')
        if not active.get('model_sha256') or not active.get('model_assets_sha256'):
            raise ValueError('active run is missing model/tokenizer provenance')
        thresholds = active.get('review_thresholds') or {}
        for threshold_name in ('top_score_below', 'margin_below'):
            threshold_value = thresholds.get(threshold_name)
            if (isinstance(threshold_value, bool) or
                    not isinstance(threshold_value, (int, float)) or
                    not math.isfinite(threshold_value) or not 0 <= threshold_value <= 1):
                raise ValueError(f'invalid or missing review threshold: {threshold_name}')

        export_id = hashlib.sha256(_canonical_bytes({
            'runId': selected_run_id, 'sourceSha256': source_sha,
            'modelSha256': active['model_sha256'], 'taxonomySha256': taxonomy_sha,
            'createdAt': _utc_now(),
        })).hexdigest()[:24]
        output = Path(output_dir) if output_dir else HERE / 'cloud-exports' / export_id
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f'export directory already exists: {output}')
        temporary = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))

        shard_meta = []
        cursor = conn.execute('''
            SELECT m.text_hash, min(m.text) AS text, min(m.day) AS first_day,
                   min(m.event_index) AS first_index
            FROM messages m
            WHERE NOT EXISTS (SELECT 1 FROM predictions p
                              WHERE p.run_id=? AND p.text_hash=m.text_hash)
            GROUP BY m.text_hash
            ORDER BY first_day, first_index
        ''', (selected_run_id,))
        buffer = []
        shard_number = 0

        def save_shard(rows: list[dict], number: int) -> None:
            shard_id = f'pending-{number:05d}'
            filename = shard_id + '.jsonl.gz'
            count, digest = _write_jsonl_gzip(temporary / filename, rows)
            shard_meta.append({'id': shard_id, 'file': filename, 'count': count,
                               'sha256': digest, 'kind': 'pending', 'importable': True})

        while True:
            batch = cursor.fetchmany(shard_size)
            if not batch:
                break
            for record in batch:
                text = record['text'] or ''
                actual_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                if actual_hash != record['text_hash']:
                    raise ValueError('source index contains a text hash mismatch')
                buffer.append({'text_hash': actual_hash, 'text': text})
            save_shard(buffer, shard_number)
            shard_number += 1
            buffer = []

        # Parity-only inputs are intentionally disjoint from pending shards and
        # can never be imported by this module.
        fixed = pilot_hashes if pilot_hashes is not None else _pilot_hashes_from_reference()
        if not fixed:
            fixed = [row[0] for row in conn.execute(
                'SELECT text_hash FROM messages GROUP BY text_hash ORDER BY text_hash LIMIT 48')]
        fixed = list(dict.fromkeys(fixed))
        if len(fixed) < 48:
            raise ValueError('pilot export needs at least 48 unique fixed text hashes')
        fixed = fixed[:48]
        pilot_by_hash = {}
        query = conn.execute('SELECT text_hash,min(text) AS text FROM messages GROUP BY text_hash')
        wanted = set(fixed)
        for row in query:
            if row['text_hash'] in wanted:
                actual_hash = hashlib.sha256((row['text'] or '').encode('utf-8')).hexdigest()
                if actual_hash != row['text_hash']:
                    raise ValueError('source index contains a pilot text hash mismatch')
                pilot_by_hash[row['text_hash']] = row['text'] or ''
        if set(pilot_by_hash) != wanted:
            raise ValueError('one or more requested parity pilot hashes are absent from this source snapshot')
        extras = []
        if pilot_extra_count:
            for row in conn.execute('''SELECT m.text_hash,min(m.text) AS text,
                    min(m.day) AS first_day,min(m.event_index) AS first_index
                FROM messages m GROUP BY m.text_hash
                ORDER BY first_day,first_index'''):
                if row['text_hash'] not in wanted:
                    extras.append({'text_hash': row['text_hash'], 'text': row['text'] or ''})
                    if len(extras) >= pilot_extra_count:
                        break
        pilot48 = [{'text_hash': value, 'text': pilot_by_hash[value]} for value in fixed]
        pilot68 = pilot48 + extras
        pilot_files = []
        pilot_sets = [(48, pilot48)]
        if len(pilot68) == 68:
            pilot_sets.append((68, pilot68))
        for size, rows in pilot_sets:
            filename = f'pilot-{size}.jsonl.gz'
            count, digest = _write_jsonl_gzip(temporary / filename, rows)
            pilot_files.append({'id': f'pilot-{size}', 'file': filename,
                                'count': count, 'sha256': digest,
                                'kind': 'parity_only', 'importable': False})

        configs = [
            {'id': a['id'], 'definition': a['definition'],
             'hypothesis': a['hypothesis'], 'signature': a['signature']}
            for a in taxonomy['acts']
        ]
        manifest = {
            'schemaVersion': 1, 'exportId': export_id,
            'createdAt': _utc_now(), 'runId': selected_run_id,
            'sourceSha256': source_sha,
            'engine': 'nli', 'model': active.get('model'),
            'modelSha256': active['model_sha256'],
            'modelAssetsSha256': active['model_assets_sha256'],
            'tokenizerAssetsSha256': _asset_subset(active['model_assets_sha256']),
            'taxonomySha256': taxonomy_sha,
            'question': active.get('question'),
            'actSpecs': configs,
            'scoreSemantics': SCORE_SEMANTICS,
            'probabilitiesCalibrated': False,
            'maxLength': MAX_LENGTH,
            'truncationPolicy': {
                'strategy': 'only_first',
                'limit': MAX_LENGTH,
                'messageTruncationReportedPerHypothesis': True,
            },
            'reviewThresholds': active.get('review_thresholds'),
            'method': active.get('method', METHOD),
            'contextUsed': bool(active.get('context_used', False)),
            'extractionPerformed': bool(active.get('extraction_performed', False)),
            'runtimeSettingsAtExport': settings,
            'activeRunConfigAtExport': active,
            'pendingShards': shard_meta,
            'pilotShards': pilot_files,
            'pendingUniqueCount': sum(item['count'] for item in shard_meta),
        }
        manifest['manifestSha256'] = _object_sha256(manifest)
        (temporary / 'manifest.json').write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, output)
        temporary = None
        return manifest
    finally:
        if temporary is not None:
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)
        conn.close()


def _validate_export_bundle(bundle_dir: Path) -> dict:
    manifest_path = bundle_dir / 'manifest.json'
    if not manifest_path.is_file() or manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('export manifest is missing or too large')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    _verify_manifest_hash(manifest)
    for required in ('exportId', 'runId', 'sourceSha256', 'modelSha256',
                     'taxonomySha256', 'actSpecs', 'pendingShards'):
        if not manifest.get(required):
            raise ValueError(f'export manifest missing {required}')
    if manifest.get('scoreSemantics') != SCORE_SEMANTICS or manifest.get('maxLength') != MAX_LENGTH:
        raise ValueError('export manifest uses unsupported scoring or sequence-length semantics')
    if len(manifest['actSpecs']) != 12:
        raise ValueError('export manifest must pin exactly twelve acts')
    ids = [item.get('id') for item in manifest['actSpecs']]
    if len(set(ids)) != 12 or any(not isinstance(value, str) for value in ids):
        raise ValueError('export manifest contains invalid act IDs')
    return manifest


def _validate_cloud_execution(execution: dict, manifest: dict) -> None:
    if not isinstance(execution, dict):
        raise ValueError('result manifest is missing cloudExecution metadata')
    required = ('executionId', 'backend', 'device', 'dtype', 'modelSha256',
                'modelAssetsSha256', 'sourceSha256', 'taxonomySha256',
                'runtimeSha256', 'codeSha256', 'dependencies', 'maxLength')
    missing = [key for key in required if key not in execution]
    if missing:
        raise ValueError('cloudExecution missing required metadata: ' + ', '.join(missing))
    if execution.get('backend') != 'cuda' or not str(execution.get('device', '')).startswith('cuda'):
        raise ValueError('cloud execution must identify an actual CUDA backend/device')
    if execution.get('dtype') not in ('float16', 'torch.float16'):
        raise ValueError('cloud execution dtype must be float16')
    if execution.get('modelSha256') != manifest['modelSha256']:
        raise ValueError('cloud result model fingerprint does not match the export')
    if execution.get('modelAssetsSha256') != manifest['modelAssetsSha256']:
        raise ValueError('cloud result tokenizer/model assets do not match the export')
    if execution.get('sourceSha256') != manifest['sourceSha256'] or execution.get('taxonomySha256') != manifest['taxonomySha256']:
        raise ValueError('cloud execution source or taxonomy fingerprint does not match the export')
    if execution.get('maxLength') != MAX_LENGTH:
        raise ValueError('cloud execution max_length must remain 512')
    if execution.get('scoreSemantics', SCORE_SEMANTICS) != SCORE_SEMANTICS:
        raise ValueError('cloud execution score semantics do not match the export')
    for key in ('runtimeSha256', 'codeSha256'):
        value = execution.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f'cloudExecution {key} must be a SHA-256 hex digest')
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f'cloudExecution {key} must be a SHA-256 hex digest') from exc
    if not isinstance(execution.get('dependencies'), dict):
        raise ValueError('cloudExecution dependencies must be an object')


def _validate_result_manifest(result_manifest: dict, export_manifest: dict,
                              results_dir: Path) -> tuple[dict, dict]:
    if result_manifest.get('exportId') != export_manifest['exportId']:
        raise ValueError('cloud results belong to a different export')
    for result_key, export_key in (
        ('runId', 'runId'), ('sourceSha256', 'sourceSha256'),
        ('modelSha256', 'modelSha256'), ('taxonomySha256', 'taxonomySha256'),
        ('scoreSemantics', 'scoreSemantics'), ('maxLength', 'maxLength'),
        ('manifestSha256', 'manifestSha256'),
    ):
        if result_manifest.get(result_key) != export_manifest.get(export_key):
            raise ValueError(f'cloud result {result_key} does not match export')
    _validate_cloud_execution(result_manifest.get('cloudExecution'), export_manifest)
    results = result_manifest.get('shards')
    if not isinstance(results, list):
        raise ValueError('results manifest must contain shard outputs')
    by_id = {item.get('id'): item for item in results}
    if len(by_id) != len(results):
        raise ValueError('results manifest contains duplicate shard IDs')
    return result_manifest['cloudExecution'], by_id


def _validate_approval(approval: dict, result_manifest: dict,
                       result_manifest_sha: str, export_manifest: dict) -> None:
    if approval.get('approved') is not True:
        raise ValueError('root parity approval is required before importing CUDA predictions')
    if not approval.get('approvedBy') or not approval.get('approvedAt'):
        raise ValueError('parity approval must record who approved it and when')
    expected = {
        'exportId': export_manifest['exportId'],
        'manifestSha256': export_manifest['manifestSha256'],
        'runId': export_manifest['runId'],
        'sourceSha256': export_manifest['sourceSha256'],
        'modelSha256': export_manifest['modelSha256'],
        'taxonomySha256': export_manifest['taxonomySha256'],
        'maxLength': MAX_LENGTH,
        'scoreSemantics': SCORE_SEMANTICS,
        'executionId': result_manifest['cloudExecution']['executionId'],
        'resultsManifestSha256': result_manifest_sha,
    }
    for key, value in expected.items():
        if approval.get(key) != value:
            raise ValueError(f'parity approval {key} does not match imported results')
    pilot = approval.get('pilotComparison')
    if not isinstance(pilot, dict) or pilot.get('sampleCount', 0) < 48:
        raise ValueError('parity approval must record at least 48 fixed-sample comparisons')


def _result_manifest_hash(data: dict) -> str:
    return _object_sha256({key: value for key, value in data.items()
                           if key != 'resultsManifestSha256'})


def _validated_predictions(bundle_dir: Path, results_dir: Path,
                           export_manifest: dict, result_manifest: dict,
                           shard_id: str, outputs_by_id: dict,
                           execution: dict) -> tuple[list[tuple], dict]:
    input_shard = next((item for item in export_manifest['pendingShards']
                        if item.get('id') == shard_id), None)
    if input_shard is None:
        raise ValueError('only pending source shards can be imported; pilot shards are never importable')
    output_shard = outputs_by_id.get(shard_id)
    if output_shard is None:
        raise ValueError(f'results manifest has no output for {shard_id}')
    if output_shard.get('inputFile') != input_shard['file']:
        raise ValueError('result shard input filename does not match exported source shard')
    if output_shard.get('inputSha256') != input_shard['sha256']:
        raise ValueError('result shard input fingerprint does not match exported source shard')
    if output_shard.get('count') != input_shard['count']:
        raise ValueError('result shard count does not match source shard count')
    input_rows = list(_iter_jsonl_gzip(_safe_child(bundle_dir, input_shard['file'])))
    expected = {row['text_hash']: row['text'] for row in input_rows}
    if len(expected) != len(input_rows):
        raise ValueError('exported input shard contains duplicate text hashes')
    if any(hashlib.sha256(text.encode('utf-8')).hexdigest() != text_hash
           for text_hash, text in expected.items()):
        raise ValueError('exported input shard text hash does not match its text')
    actual_input_sha = _file_sha256(_safe_child(bundle_dir, input_shard['file']))
    if actual_input_sha != input_shard['sha256']:
        raise ValueError('exported source shard SHA-256 mismatch')
    output_path = _safe_child(results_dir, output_shard.get('outputFile'))
    if not output_path.is_file() or _file_sha256(output_path) != output_shard.get('outputSha256'):
        raise ValueError('cloud output shard is missing or has a SHA-256 mismatch')
    output_rows = list(_iter_jsonl_gzip(output_path))
    if len(output_rows) != input_shard['count']:
        raise ValueError('cloud output row count does not match source shard')
    act_specs = export_manifest['actSpecs']
    act_ids = [item['id'] for item in act_specs]
    signatures = {item['id']: item['signature'] for item in act_specs}
    threshold = export_manifest.get('reviewThresholds') or {}
    weak = float(threshold.get('top_score_below', 0.55))
    margin_threshold = float(threshold.get('margin_below', 0.15))
    predictions = []
    seen = set()
    for row in output_rows:
        text_hash = row.get('text_hash')
        if not isinstance(text_hash, str) or text_hash in seen:
            raise ValueError('cloud results contain a missing or duplicate text_hash')
        seen.add(text_hash)
        if text_hash not in expected:
            raise ValueError('cloud results contain an unexpected text_hash')
        scores = row.get('scores')
        if not isinstance(scores, dict) or set(scores) != set(act_ids):
            raise ValueError('each result must contain scores for exactly all twelve acts')
        clean_scores = {}
        for label in act_ids:
            score = scores[label]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError('scores must be numeric values')
            score = float(score)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('scores must be finite values in [0, 1]')
            clean_scores[label] = score
        truncation = row.get('truncation')
        if not isinstance(truncation, dict) or not isinstance(truncation.get('truncated'), bool):
            raise ValueError('each result must include a boolean truncation.truncated field')
        ranked = sorted(act_ids, key=lambda key: clean_scores[key], reverse=True)
        top = ranked[0]
        top_score = clean_scores[top]
        margin = top_score - clean_scores[ranked[1]]
        truncated = truncation['truncated']
        text = expected[text_hash]
        needs_review = (top_score < weak or margin < margin_threshold or
                        truncated or not text.strip())
        runtime = dict(row.get('runtime') or {})
        runtime.update({
            'probabilities': clean_scores,
            'score_semantics': SCORE_SEMANTICS,
            'probabilities_sum_to_one': False,
            'truncated': truncated,
            'truncation': truncation,
            'device': execution['device'], 'backend': 'cuda',
            'calibrated': False,
            'label_mapping': {'0': 'entailment', '1': 'not_entailment'},
            'entailment_label': 'entailment',
        })
        record = {
            'scores': clean_scores,
            'primary_candidate': top,
            'primary_act': 'uncertain' if needs_review else top,
            'score_semantics': SCORE_SEMANTICS,
            'candidate_templates': [
                {'act': label, 'score': clean_scores[label], 'signature': signatures[label]}
                for label in ranked if clean_scores[label] >= 0.5
            ],
            'template': signatures[top],
            'template_status': 'proposed skeleton; slots not extracted',
            'slot_values': None, 'extraction_performed': False,
            'context_used': False,
            'multi_act_status': 'not segmented; one primary candidate only',
            'needs_review': needs_review,
            'review_reasons': (['weak score or narrow margin']
                               if top_score < weak or margin < margin_threshold else [])
                              + (['message truncated'] if truncated else [])
                              + (['empty message'] if not text.strip() else []),
            'runtime': runtime,
            'run_id': export_manifest['runId'],
            'execution_epoch_id': None,
        }
        predictions.append((export_manifest['runId'], text_hash, top, top_score,
                            margin, int(needs_review), int(truncated), record))
    if seen != set(expected):
        raise ValueError('cloud output omits one or more source text hashes')
    return predictions, output_shard


def import_shard(bundle_dir: str | Path,
                 results_dir: str | Path,
                 shard_id: str,
                 parity_approval_path: str | Path,
                 db_path: str | Path = DEFAULT_DB,
                 lock_path: str | Path | None = None) -> dict:
    """Validate a whole result shard, then transactionally import it into the active run."""
    bundle = Path(bundle_dir).resolve()
    result_root = Path(results_dir).resolve()
    export_manifest = _validate_export_bundle(bundle)
    result_manifest_path = result_root / 'results-manifest.json'
    if not result_manifest_path.is_file() or result_manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('results manifest is missing or too large')
    result_manifest = json.loads(result_manifest_path.read_text(encoding='utf-8'))
    execution, outputs_by_id = _validate_result_manifest(result_manifest, export_manifest, result_root)
    result_manifest_sha = _result_manifest_hash(result_manifest)
    approval_path = Path(parity_approval_path)
    if not approval_path.is_file() or approval_path.stat().st_size > 64 * 1024:
        raise ValueError('parity approval artifact is missing or too large')
    approval = json.loads(approval_path.read_text(encoding='utf-8'))
    _validate_approval(approval, result_manifest, result_manifest_sha, export_manifest)
    rows, output_shard = _validated_predictions(bundle, result_root, export_manifest,
                                                result_manifest, shard_id,
                                                outputs_by_id, execution)

    lock_file = Path(lock_path) if lock_path else HERE / 'run.lock'
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_file.open('a+')
    lock_acquired = False
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError as exc:
            raise RuntimeError('local runner lock is held; pause the local run before importing cloud results') from exc

        # `store.connect` applies only schema setup before the transaction. All
        # writes below bypass store.meta's auto-commit helper to remain atomic.
        conn = sqlite3.connect(str(Path(db_path)), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA busy_timeout=30000')
        try:
            conn.execute('BEGIN IMMEDIATE')
            active = _read_meta(conn, 'active_run') or {}
            progress = _read_meta(conn, 'progress') or {}
            if active.get('id') != export_manifest['runId']:
                raise ValueError('active logical run changed since export')
            if progress.get('state') not in ALLOWED_IDLE_STATES:
                raise RuntimeError('local run must be paused or interrupted before importing cloud results')
            if _read_meta(conn, 'source_sha256') != export_manifest['sourceSha256']:
                raise ValueError('indexed source snapshot changed since export')
            if active.get('source_sha256') != export_manifest['sourceSha256']:
                raise ValueError('active run source fingerprint changed since export')
            if active.get('model_sha256') != export_manifest['modelSha256']:
                raise ValueError('active model fingerprint changed since export')
            if active.get('model_assets_sha256') != export_manifest['modelAssetsSha256']:
                raise ValueError('active model/tokenizer assets changed since export')
            if active.get('taxonomy_sha256') != export_manifest['taxonomySha256']:
                raise ValueError('active taxonomy fingerprint changed since export')
            if active.get('score_semantics') != SCORE_SEMANTICS or (active.get('runtime_settings') or {}).get('max_length') != MAX_LENGTH:
                raise ValueError('active run scoring semantics or max_length changed since export')
            text_hashes = [row[1] for row in rows]
            placeholders = ','.join('?' for _ in text_hashes)
            source_rows = conn.execute(
                f'SELECT text_hash,min(text) AS text FROM messages WHERE text_hash IN ({placeholders}) GROUP BY text_hash',
                text_hashes).fetchall() if text_hashes else []
            source_texts = {record['text_hash']: (record['text'] or '') for record in source_rows}
            imported_texts = {record['text_hash']: record['text'] for record in _iter_jsonl_gzip(
                _safe_child(bundle, output_shard['inputFile']))}
            if source_texts != imported_texts:
                raise ValueError('one or more exported texts do not match the indexed source snapshot')
            if len(source_texts) != len(rows):
                raise ValueError('one or more exported text hashes are absent from the source index')
            conflict = conn.execute('''SELECT count(*) FROM predictions
                WHERE run_id=? AND text_hash IN (%s)''' % placeholders,
                [export_manifest['runId']] + text_hashes).fetchone()[0] if rows else 0
            if conflict:
                raise ValueError('one or more cloud results were already classified locally; refusing to replace predictions')

            epochs = list(active.get('execution_epochs') or [])
            execution_id = execution['executionId']
            epoch_id = hashlib.sha256(_canonical_bytes({
                'runId': active['id'], 'exportId': export_manifest['exportId'],
                'executionId': execution_id,
                'codeSha256': execution['codeSha256'],
                'modelSha256': execution['modelSha256'],
            })).hexdigest()[:20]
            epoch = next((item for item in epochs if item.get('id') == epoch_id), None)
            if epoch is None:
                epoch = {
                    'id': epoch_id, 'run_id': active['id'],
                    'epoch_number': len(epochs) + 1, 'started_at': execution.get('startedAt') or _utc_now(),
                    'runner_sha256': execution['codeSha256'],
                    'runtime_sha256': execution['runtimeSha256'],
                    'dependencies': execution['dependencies'],
                    'runtime_settings': {
                        'model_type': 'deberta-v2', 'backend': 'cuda',
                        'device': execution['device'], 'dtype': execution['dtype'],
                        'max_length': MAX_LENGTH, 'score_semantics': SCORE_SEMANTICS,
                        'label_mapping': {'0': 'entailment', '1': 'not_entailment'},
                        'entailment_label': 'entailment', 'local_files_only': True,
                        'trust_remote_code': False,
                    },
                    'device': execution['device'], 'dtype': execution['dtype'],
                    'backend': 'cuda', 'backend_source_sha256': execution['codeSha256'],
                    'backend_version': execution.get('runtimeVersion'),
                    'outer_batch_size': None, 'pair_batch_size': execution.get('pairBatchSize'),
                    'source_sha256': export_manifest['sourceSha256'],
                    'model_sha256': export_manifest['modelSha256'],
                    'taxonomy_sha256': export_manifest['taxonomySha256'],
                    'cloud_execution': execution,
                    'cloud_export_id': export_manifest['exportId'],
                    'imported_shards': [], 'imported_prediction_count': 0,
                    'predictions_before_epoch': conn.execute(
                        'SELECT count(*) FROM predictions WHERE run_id=?', (active['id'],)
                    ).fetchone()[0],
                    'resume_of_epoch_id': epochs[-1].get('id') if epochs else None,
                }
                epochs.append(epoch)
            elif epoch.get('cloud_export_id') != export_manifest['exportId']:
                raise ValueError('execution epoch ID collision with another export')
            if shard_id in epoch['imported_shards']:
                raise ValueError('cloud shard has already been imported')

            for run, text_hash, top, score, margin, needs_review, truncated, record in rows:
                record['execution_epoch_id'] = epoch_id
                conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                             (run, text_hash, top, score, margin, needs_review, truncated,
                              json.dumps(record, ensure_ascii=False, sort_keys=True)))
            epoch['imported_shards'].append(shard_id)
            epoch['imported_prediction_count'] += len(rows)
            epoch['last_imported_at'] = _utc_now()
            active['execution_epochs'] = epochs
            active['latest_execution_epoch_id'] = epoch_id
            active['last_resumed_at'] = epoch['last_imported_at']
            remaining = conn.execute('''SELECT count(*) FROM (
                SELECT DISTINCT m.text_hash FROM messages m WHERE NOT EXISTS
                (SELECT 1 FROM predictions p WHERE p.run_id=? AND p.text_hash=m.text_hash))''',
                (active['id'],)).fetchone()[0]
            imported_total = conn.execute('SELECT count(*) FROM predictions WHERE run_id=?',
                                          (active['id'],)).fetchone()[0]
            new_progress = {
                'state': 'complete' if remaining == 0 else 'partial',
                'run_id': active['id'], 'execution_epoch_id': epoch_id,
                'execution_epoch_number': epoch['epoch_number'],
                'backend': 'cuda', 'device': execution['device'],
                'imported_this_shard': len(rows),
                'imported_shards_this_epoch': len(epoch['imported_shards']),
                'cumulative_unique_predictions': imported_total,
                'unique_remaining': remaining,
                'updated_at': _utc_now(),
            }
            _write_meta_uncommitted(conn, 'active_run', active)
            _write_meta_uncommitted(conn, 'progress', new_progress)
            conn.commit()
            return {'runId': active['id'], 'shardId': shard_id,
                    'imported': len(rows), 'remainingUnique': remaining,
                    'state': new_progress['state'], 'executionEpochId': epoch_id,
                    'resultsManifestSha256': result_manifest_sha,
                    'outputShardSha256': output_shard['outputSha256']}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    finally:
        if lock_acquired:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export_parser = commands.add_parser('export')
    export_parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    export_parser.add_argument('--output', type=Path, required=True)
    export_parser.add_argument('--run-id')
    export_parser.add_argument('--shard-size', type=int, default=DEFAULT_SHARD_SIZE)
    import_parser = commands.add_parser('import-shard')
    import_parser.add_argument('--bundle', type=Path, required=True)
    import_parser.add_argument('--results', type=Path, required=True)
    import_parser.add_argument('--shard-id', required=True)
    import_parser.add_argument('--parity-approval', type=Path, required=True)
    import_parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    if args.command == 'export':
        manifest = export_bundle(args.db, args.output, args.run_id, shard_size=args.shard_size)
        print(json.dumps({'exportId': manifest['exportId'], 'output': str(args.output),
                          'pendingUniqueCount': manifest['pendingUniqueCount'],
                          'pilotCounts': [item['count'] for item in manifest['pilotShards']]}))
    else:
        print(json.dumps(import_shard(args.bundle, args.results, args.shard_id,
                                      args.parity_approval, args.db), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
