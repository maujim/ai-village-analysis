"""Resumable offline scoring with explicit per-execution provenance epochs.

No generative extraction is performed. Continuing a logical run after code or
execution changes requires --continue-run and a semantic compatibility check.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from store import connect, meta, status, ROOT

QUESTION = 'What is the primary communicative function of this message? Choose the best matching act.'
RUNTIME_FILES = {'nano': 'nano_runtime.py', 'jev': 'jev_runtime.py', 'nli': 'nli_runtime.py'}
SEMANTIC_RUNTIME_KEYS = (
    'model_type', 'dtype', 'max_length', 'max_message_tokens', 'score_semantics',
    'label_mapping', 'entailment_label', 'local_files_only', 'trust_remote_code',
)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def semantic_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Fields that must match before old and new predictions share one run ID.

    Runner/runtime hashes, package versions and batch sizes are execution
    provenance. Device, dtype, model assets, tokenizer assets, length policy,
    taxonomy, question and source are compatibility requirements.
    """
    settings = config.get('runtime_settings') or {}
    projection = {
        'engine': config.get('engine'),
        'model': config.get('model'),
        'model_sha256': config.get('model_sha256'),
        'model_assets_sha256': config.get('model_assets_sha256'),
        'taxonomy_sha256': config.get('taxonomy_sha256'),
        'question': config.get('question'),
        'source_sha256': config.get('source_sha256'),
        'score_semantics': config.get('score_semantics'),
        'method': config.get('method'),
        'context_used': config.get('context_used'),
        'probabilities_calibrated': config.get('probabilities_calibrated'),
        'extraction_performed': config.get('extraction_performed'),
        'review_thresholds': config.get('review_thresholds'),
        'parameter_count': config.get('parameter_count'),
        'device': config.get('device'),
        'runtime_semantics': {key: settings.get(key) for key in SEMANTIC_RUNTIME_KEYS},
    }
    # Metadata read from SQLite has JSON string keys, while freshly loaded
    # model config dictionaries can still use integer label IDs. Compare the
    # serialized representation so equivalent mappings remain compatible.
    return json.loads(json.dumps(projection, sort_keys=True))


def validate_continuation(old: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject continuing a run when source/model/scoring semantics differ."""
    before = semantic_projection(old)
    after = semantic_projection(current)
    differences = [key for key in before if key != 'runtime_semantics' and before[key] != after[key]]
    differences.extend(
        f'runtime_settings.{key}'
        for key in SEMANTIC_RUNTIME_KEYS
        if before['runtime_semantics'].get(key) != after['runtime_semantics'].get(key)
    )
    if differences:
        raise ValueError('Cannot continue this run; semantic settings changed: ' + ', '.join(differences))


def _prediction_count(conn, run_id: str) -> int:
    return conn.execute('SELECT count(*) FROM predictions WHERE run_id=?', (run_id,)).fetchone()[0]


def _untagged_prediction_count(conn, run_id: str) -> int:
    # Existing result JSON predates epoch tracking; avoid loading the corpus-sized
    # result column into Python just to identify those old rows.
    return conn.execute('''SELECT count(*) FROM predictions
        WHERE run_id=? AND result NOT LIKE '%"execution_epoch_id"%' ''', (run_id,)).fetchone()[0]


def _epoch_record(config: dict[str, Any], epoch_number: int, *, run_id: str,
                  started_at: str, outer_batch: int, pair_batch: int | None,
                  predictions_before: int, unique_remaining_before: int,
                  resume_of: str | None, untagged_before: int = 0) -> dict[str, Any]:
    epoch_core = {
        'run_id': run_id,
        'epoch_number': epoch_number,
        'started_at': started_at,
        'runner_sha256': config.get('runner_sha256'),
        'runtime_sha256': config.get('runtime_sha256'),
        'dependencies': config.get('dependencies'),
        'runtime_settings': config.get('runtime_settings'),
        'device': config.get('device'),
        'dtype': (config.get('runtime_settings') or {}).get('dtype'),
        'backend': config.get('backend', 'torch'),
        'backend_source_sha256': config.get('backend_source_sha256'),
        'backend_version': config.get('backend_version'),
        'outer_batch_size': outer_batch,
        'pair_batch_size': pair_batch,
        'source_sha256': config.get('source_sha256'),
        'model_sha256': config.get('model_sha256'),
        'taxonomy_sha256': config.get('taxonomy_sha256'),
        'predictions_before_epoch': predictions_before,
        'unique_remaining_before_epoch': unique_remaining_before,
        'resume_of_epoch_id': resume_of,
    }
    epoch_id = hashlib.sha256(json.dumps(epoch_core, sort_keys=True).encode()).hexdigest()[:20]
    return {'id': epoch_id, **epoch_core, 'predictions_missing_epoch_at_start': untagged_before}


def prepare_run(conn, config: dict[str, Any], *, continue_run: str | None,
                outer_batch: int, pair_batch: int | None,
                started_at: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a new logical run or append a provenance epoch to an explicit run."""
    started_at = started_at or timestamp()
    current = dict(config)
    current['execution_settings'] = {
        'outer_batch_size': outer_batch,
        'pair_batch_size': pair_batch,
        'backend': current.get('backend', 'torch'),
    }
    if continue_run:
        old = meta(conn, 'active_run') or {}
        if str(old.get('id') or '') != continue_run:
            raise ValueError('The requested --continue-run is not the active logical run')
        validate_continuation(old, current)
        active = dict(old)  # Preserve original model/code hashes and original start time.
        epochs = list(active.get('execution_epochs') or [])
        predictions_before = _prediction_count(conn, continue_run)
        unique_total = conn.execute('SELECT count(DISTINCT text_hash) FROM messages').fetchone()[0]
        unique_remaining = max(0, unique_total - predictions_before)
        untagged = _untagged_prediction_count(conn, continue_run)
        if not epochs:
            # Make the historical provenance explicit without claiming the new
            # runner knows which old rows came from it.
            old_settings = active.get('runtime_settings') or {}
            legacy_core = {
                'run_id': continue_run,
                'epoch_number': 1,
                'started_at': active.get('started_at'),
                'runner_sha256': active.get('runner_sha256'),
                'runtime_sha256': active.get('runtime_sha256'),
                'dependencies': active.get('dependencies'),
                'runtime_settings': old_settings,
                'device': active.get('device'),
                'dtype': old_settings.get('dtype'),
                'backend': active.get('backend', 'torch'),
                'backend_source_sha256': active.get('backend_source_sha256'),
                'backend_version': active.get('backend_version'),
                'outer_batch_size': None,
                'pair_batch_size': old_settings.get('pair_batch_size', old_settings.get('batch_size')),
                'source_sha256': active.get('source_sha256'),
                'model_sha256': active.get('model_sha256'),
                'taxonomy_sha256': active.get('taxonomy_sha256'),
                'predictions_before_epoch': 0,
                'unique_remaining_before_epoch': unique_total,
                'resume_of_epoch_id': None,
            }
            legacy_id = 'legacy-' + continue_run
            epochs.append({
                'id': legacy_id, **legacy_core,
                'predictions_missing_epoch_at_start': untagged,
                'provenance_status': 'legacy predictions have no per-row execution epoch ID',
            })
        prior_epoch = epochs[-1].get('id')
        epoch = _epoch_record(current, len(epochs) + 1, run_id=continue_run,
                              started_at=started_at, outer_batch=outer_batch,
                              pair_batch=pair_batch, predictions_before=predictions_before,
                              unique_remaining_before=unique_remaining,
                              resume_of=prior_epoch, untagged_before=untagged)
        epochs.append(epoch)
        active['execution_epochs'] = epochs
        active['latest_execution_epoch_id'] = epoch['id']
        active['last_resumed_at'] = started_at
        meta(conn, 'active_run', active)
        return active, epoch

    # A new run ID binds every original provenance field plus the batching
    # request. Execution epochs record changes without mutating this identity.
    current['started_at'] = started_at
    current['id'] = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:20]
    unique_total = conn.execute('SELECT count(DISTINCT text_hash) FROM messages').fetchone()[0]
    epoch = _epoch_record(current, 1, run_id=current['id'], started_at=started_at,
                          outer_batch=outer_batch, pair_batch=pair_batch,
                          predictions_before=0, unique_remaining_before=unique_total,
                          resume_of=None)
    current['execution_epochs'] = [epoch]
    current['latest_execution_epoch_id'] = epoch['id']
    meta(conn, 'active_run', current)
    return current, epoch


def current_config(args, runtime, model: str, model_dir: Path, source_sha256: str) -> dict[str, Any]:
    runtime_settings = runtime.metadata()
    device = str(runtime.device)
    dependencies = {
        key: importlib.metadata.version(key)
        for key in ('torch', 'transformers', 'tokenizers', 'safetensors', 'dspy')
    }
    backend = args.backend if args.engine == 'nli' else 'torch'
    backend_file = HERE / 'mlx_deberta.py'
    backend_sha = digest_file(backend_file) if backend == 'mlx' else None
    backend_version = importlib.metadata.version('mlx') if backend == 'mlx' else None
    parameter_count = getattr(runtime, 'parameter_count', None)
    if parameter_count is None:
        parameter_count = sum(p.numel() for p in runtime.model.parameters())
    return {
        'engine': args.engine,
        'backend': backend,
        'backend_source_sha256': backend_sha,
        'backend_version': backend_version,
        'model': model,
        'model_sha256': digest_file(model_dir / 'model.safetensors'),
        'taxonomy_sha256': digest_file(HERE / 'taxonomy.json'),
        'question': QUESTION,
        'runtime_sha256': digest_file(HERE / RUNTIME_FILES[args.engine]),
        'runner_sha256': digest_file(Path(__file__)),
        'model_assets_sha256': {
            p.name: digest_file(p) for p in sorted(model_dir.glob('*'))
            if p.is_file() and p.suffix in ('.json', '.model', '.jinja')
        },
        'dependencies': dependencies | ({'mlx': backend_version} if backend_version else {}),
        'parameter_count': parameter_count,
        'runtime_settings': runtime_settings,
        'score_semantics': 'independent_entailment' if args.engine == 'nli' else 'normalized_forced_choice',
        'method': 'primary-act-template-v1',
        'context_used': False,
        'probabilities_calibrated': False,
        'extraction_performed': False,
        'review_thresholds': {'top_score_below': 0.55, 'margin_below': 0.15},
        'source_sha256': source_sha256,
        'device': device,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', choices=['nano', 'jev', 'nli'], default='nli')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int, default=0, help='New unique texts; zero means all pending texts')
    parser.add_argument('--batch', type=int, default=128, help='Messages per outer runner batch')
    parser.add_argument('--pair-batch', type=int, default=32,
                        help='Message/hypothesis pairs per NLI model forward batch')
    parser.add_argument('--backend', choices=['torch', 'mlx'], default='torch',
                        help='NLI execution backend; backend changes require an explicit continuation epoch')
    parser.add_argument('--day', type=int)
    parser.add_argument('--continue-run', help='Explicitly continue this active logical run after semantic validation')
    args = parser.parse_args(argv)
    if args.engine != 'nli' and args.backend != 'torch':
        parser.error('--backend mlx is currently supported only with --engine nli')
    if args.batch <= 0 or args.pair_batch <= 0 or args.limit < 0:
        parser.error('--batch and --pair-batch must be positive and --limit must be nonnegative')

    import fcntl
    lock = (HERE / 'run.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('another message-signature runner holds the process lock')
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()

    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    conn = connect()
    original_progress = meta(conn, 'progress')
    previous_active = meta(conn, 'active_run') or {}
    if args.continue_run and str(previous_active.get('id') or '') != args.continue_run:
        conn.close()
        raise ValueError('--continue-run must name the currently active logical run')
    meta(conn, 'progress', {
        'state': 'loading', 'pid': os.getpid(), 'engine': args.engine,
        'run_id': args.continue_run, 'updated_at': timestamp(),
        'requested_outer_batch_size': args.batch,
        'requested_pair_batch_size': args.pair_batch if args.engine == 'nli' else None,
    })

    try:
        import torch
        torch.set_num_threads(4)
        taxonomy = json.loads((HERE / 'taxonomy.json').read_text())
        acts = taxonomy['acts']
        options = {a['id']: a['definition'] for a in acts}
        templates = {a['id']: a['signature'] for a in acts}
        pair_batch = args.pair_batch if args.engine == 'nli' else None
        if args.engine == 'nano':
            from nano_runtime import NanoRuntime
            runtime = NanoRuntime(device=args.device, max_length=512, batch_size=32)
            model = 'sdmlai/nano-jev'
            model_dir = ROOT / 'models/nano-jev'
        elif args.engine == 'nli':
            from nli_runtime import NLIRuntime
            runtime = NLIRuntime(device=args.device, batch_size=args.pair_batch,
                                 pair_batch_size=args.pair_batch, backend=args.backend)
            model = 'MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c'
            model_dir = ROOT / 'models/deberta-base-zeroshot'
        else:
            from jev_runtime import JevRuntime
            runtime = JevRuntime(device=args.device, batch_size=args.batch)
            model = 'Qwen/Qwen3-0.6B'
            model_dir = ROOT / 'models/qwen3-0.6b'
        source_sha = meta(conn, 'source_sha256')
        if not source_sha:
            raise RuntimeError('Run message_signatures/store.py first')
        config = current_config(args, runtime, model, model_dir, source_sha)
        config, epoch = prepare_run(conn, config, continue_run=args.continue_run,
                                    outer_batch=args.batch, pair_batch=pair_batch)
    except Exception:
        # A failed load or rejected continuation must not replace the progress
        # state for the pre-existing logical run.
        meta(conn, 'progress', original_progress or {})
        conn.close()
        raise

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    start = time.monotonic()
    completed = 0
    error = None

    def progress(state: str) -> dict[str, Any]:
        elapsed = time.monotonic() - start
        cumulative = _prediction_count(conn, config['id'])
        value = {
            'state': state, 'pid': os.getpid(), 'run_id': config['id'],
            'execution_epoch_id': epoch['id'], 'execution_epoch_number': epoch['epoch_number'],
            'completed_this_session': completed,
            'completed_this_epoch': completed,
            'cumulative_unique_predictions': cumulative,
            'predictions_before_epoch': epoch['predictions_before_epoch'],
            'unique_texts_per_second': round(completed / max(elapsed, .001), 3),
            'elapsed_seconds': round(elapsed, 2), 'updated_at': timestamp(),
            'day_scope': args.day, 'limit': args.limit, 'error': error,
            'outer_batch_size': args.batch, 'pair_batch_size': pair_batch,
        }
        meta(conn, 'progress', value)
        return value

    progress('running')
    try:
        where = ' AND m.day=?' if args.day is not None else ''
        values = [config['id']] + ([args.day] if args.day is not None else [])
        pending = conn.execute('''SELECT m.text_hash,m.text FROM messages m WHERE NOT EXISTS
          (SELECT 1 FROM predictions p WHERE p.run_id=? AND p.text_hash=m.text_hash)'''
          + where + ' GROUP BY m.text_hash ORDER BY min(m.day),min(m.event_index)', values).fetchall()
        if args.limit:
            pending = pending[:args.limit]
        for batch_start in range(0, len(pending), args.batch):
            if stopping:
                break
            rows = pending[batch_start:batch_start + args.batch]
            output = runtime.predict([row['text'] for row in rows], options, QUESTION)
            if isinstance(output, dict):
                output = output.get('results', output.get('predictions'))
            if output is None or len(output) != len(rows):
                raise ValueError('Runtime result cardinality mismatch')
            for row, result in zip(rows, output):
                scores = result.get('probabilities') or result.get('scores')
                if not isinstance(scores, dict) or set(scores) != set(options):
                    raise ValueError('Model did not score the exact taxonomy')
                if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores.values()):
                    raise ValueError('Invalid non-finite or out-of-range score')
                if args.engine != 'nli' and abs(sum(scores.values()) - 1) > .001:
                    raise ValueError('Forced-choice scores must sum to one')
                ranked = sorted(scores, key=scores.get, reverse=True)
                top = ranked[0]
                score = scores[top]
                margin = score - scores[ranked[1]]
                truncated = bool(result.get('truncated') or result.get('message_truncated'))
                needs_review = score < .55 or margin < .15 or truncated or not row['text'].strip()
                record = {
                    'scores': scores, 'primary_candidate': top,
                    'primary_act': 'uncertain' if needs_review else top,
                    'score_semantics': config['score_semantics'],
                    'candidate_templates': [
                        {'act': label, 'score': scores[label], 'signature': templates[label]}
                        for label in ranked if scores[label] >= .5
                    ],
                    'template': templates[top],
                    'template_status': 'proposed skeleton; slots not extracted',
                    'slot_values': None, 'extraction_performed': False,
                    'context_used': False,
                    'multi_act_status': 'not segmented; one primary candidate only',
                    'needs_review': needs_review,
                    'review_reasons': (['weak score or narrow margin']
                                       if score < .55 or margin < .15 else [])
                                      + (['message truncated'] if truncated else [])
                                      + (['empty message'] if not row['text'].strip() else []),
                    'runtime': result, 'run_id': config['id'],
                    'execution_epoch_id': epoch['id'],
                }
                conn.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?)',
                             (config['id'], row['text_hash'], top, score, margin,
                              int(needs_review), int(truncated), json.dumps(record)))
            conn.commit()
            completed += len(rows)
            update = progress('running')
            if completed % max(args.batch * 10, 1) == 0:
                print(json.dumps(update), flush=True)
        summary = status(conn)
        final = 'interrupted' if stopping else ('complete' if summary['remaining'] == 0 else 'partial')
        print(json.dumps(progress(final)), flush=True)
        print(json.dumps({'classified_source_records': summary['classified'],
                          'unique_predictions': summary['unique_classified'],
                          'total_source_records': summary['total'],
                          'total_unique_texts': summary['unique_total'],
                          'state': final, 'run_id': config['id'],
                          'execution_epoch_id': epoch['id']}, ensure_ascii=False), flush=True)
    except Exception as exc:
        error = str(exc)
        progress('failed')
        raise
    finally:
        conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
