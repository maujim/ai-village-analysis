"""Export a compact, read-only snapshot of a message-signature run.

No model is loaded and this module never starts, pauses, or signals a process.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / 'messages.sqlite'
DEFAULT_OUTPUT = HERE / 'full-run-report.json'
SAMPLE_PER_LABEL = 2
MAX_SAMPLE = 24
FINAL_STATES = {'complete', 'failed', 'interrupted', 'incomplete', 'partial', 'paused', 'stopped', 'no_run'}

NOTICE = (
    'Exploratory classifier output only. The act categories and hypotheses are '
    'author-defined; scores and review thresholds are not validated against '
    'human annotations. A label is not task extraction, source verification, '
    'or evidence that a reported action succeeded.'
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def open_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    conn = sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def meta(conn: sqlite3.Connection, key: str):
    row = conn.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
    return json.loads(row['value']) if row else None


def _sample_rows(conn: sqlite3.Connection, run_id: str) -> list[dict]:
    labels = [r[0] for r in conn.execute(
        'SELECT DISTINCT label FROM predictions WHERE run_id=? ORDER BY label', (run_id,)
    )]
    selected = []
    for label in labels:
        rows = conn.execute('''
            SELECT m.id,m.day,m.event_index,m.date,m.timestamp,m.speaker,m.source_type,
                   m.text,m.text_hash,p.label,p.top_score,p.margin,p.needs_review,
                   p.truncated,p.result
            FROM messages m JOIN predictions p ON p.text_hash=m.text_hash
            WHERE p.run_id=? AND p.label=?
              AND NOT EXISTS (
                SELECT 1 FROM messages earlier
                WHERE earlier.text_hash=m.text_hash
                  AND (earlier.day<m.day OR
                       (earlier.day=m.day AND earlier.event_index<m.event_index))
              )
            ORDER BY m.day,m.event_index
            LIMIT ?
        ''', (run_id, label, SAMPLE_PER_LABEL)).fetchall()
        for row in rows:
            result = json.loads(row['result'])
            selected.append({
                'id': row['id'], 'day': row['day'], 'eventIndex': row['event_index'],
                'date': row['date'], 'timestamp': row['timestamp'],
                'speaker': row['speaker'], 'sourceType': row['source_type'],
                'textHash': row['text_hash'], 'message': row['text'],
                'primaryCandidate': result.get('primary_candidate', row['label']),
                'primaryAct': result.get('primary_act'),
                'topScore': row['top_score'], 'margin': row['margin'],
                'needsReview': bool(row['needs_review']), 'truncated': bool(row['truncated']),
                'scoreSemantics': result.get('score_semantics'),
                'scores': result.get('scores', {}),
                'template': result.get('template'),
                'candidateTemplates': result.get('candidate_templates', []),
            })
    return selected[:MAX_SAMPLE]


def build_report(db_path: str | Path = DEFAULT_DB, run_id: str | None = None,
                 generated_at: str | None = None) -> dict:
    """Build a bounded report for the selected run without changing the DB."""
    conn = open_readonly(db_path)
    try:
        conn.execute('BEGIN')
        active = meta(conn, 'active_run') or {}
        active_id = str(active.get('id') or '')
        target_id = str(run_id if run_id is not None else active_id)
        replaced = bool(target_id and active_id != target_id)
        progress = meta(conn, 'progress') or {}
        state_explanation = None
        if replaced:
            state = 'run_replaced'
        elif progress.get('run_id') not in (None, '', target_id):
            state = 'run_replaced'
            replaced = True
        else:
            state = str(progress.get('state') or ('no_run' if not target_id else 'unknown'))
            if state in {'running', 'loading', 'starting'} and progress.get('pid') is not None:
                try:
                    os.kill(int(progress['pid']), 0)
                except ProcessLookupError:
                    state = 'interrupted'
                    state_explanation = 'The recorded runner process no longer exists; the last saved progress is an interrupted run.'
                except PermissionError:
                    # A liveness probe may be denied for a process owned by another account.
                    # Preserve the recorded state rather than inferring that it stopped.
                    pass

        total_records = conn.execute('SELECT count(*) FROM messages').fetchone()[0]
        unique_total = conn.execute('SELECT count(DISTINCT text_hash) FROM messages').fetchone()[0]
        if target_id:
            classified_records = conn.execute('''
                SELECT count(*) FROM messages m JOIN predictions p
                  ON p.text_hash=m.text_hash AND p.run_id=?
            ''', (target_id,)).fetchone()[0]
            unique_done = conn.execute('''
                SELECT count(*) FROM predictions p WHERE p.run_id=?
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.text_hash=p.text_hash)
            ''', (target_id,)).fetchone()[0]
            label_records = [dict(r) for r in conn.execute('''
                SELECT p.label,count(*) AS sourceRecords,
                       sum(p.needs_review) AS needsReview,
                       sum(p.truncated) AS truncated
                FROM messages m JOIN predictions p
                  ON p.text_hash=m.text_hash AND p.run_id=?
                GROUP BY p.label ORDER BY sourceRecords DESC,p.label
            ''', (target_id,))]
            label_unique = [dict(r) for r in conn.execute('''
                SELECT p.label,count(*) AS uniqueTexts
                FROM predictions p WHERE p.run_id=?
                  AND EXISTS (SELECT 1 FROM messages m WHERE m.text_hash=p.text_hash)
                GROUP BY p.label ORDER BY uniqueTexts DESC,p.label
            ''', (target_id,))]
            samples = _sample_rows(conn, target_id)
        else:
            classified_records = unique_done = 0
            label_records, label_unique, samples = [], [], []
        remaining_records = max(0, total_records - classified_records)
        remaining_unique = max(0, unique_total - unique_done)
        if state == 'complete' and (remaining_records > 0 or remaining_unique > 0):
            state = 'incomplete'
            state_explanation = (
                'Progress metadata said complete, but indexed source records or unique texts '
                'remain unclassified; the report does not promote this run to complete.'
            )
        config_fields = (
            'id', 'engine', 'model', 'device', 'score_semantics', 'method',
            'source_sha256', 'model_sha256', 'model_assets_sha256',
            'taxonomy_sha256', 'runtime_sha256', 'runner_sha256',
            'dependencies', 'runtime_settings', 'review_thresholds',
            'probabilities_calibrated', 'extraction_performed', 'context_used',
            'started_at',
        )
        run = ({key: active[key] for key in config_fields if key in active}
               if target_id and target_id == active_id else {})
        return {
            'schemaVersion': 1,
            'generatedAt': generated_at or utc_now(),
            'notice': NOTICE,
            'state': state,
            'stateExplanation': state_explanation,
            'runId': target_id or None,
            'activeRunIdAtExport': active_id or None,
            'runReplaced': replaced,
            'source': {
                'sha256': meta(conn, 'source_sha256') or active.get('source_sha256'),
                'chatMessageRecords': total_records,
                'uniqueExactTexts': unique_total,
            },
            'run': run,
            'coverage': {
                'sourceRecords': {
                    'classified': classified_records, 'total': total_records,
                    'remaining': remaining_records,
                    'fraction': classified_records / total_records if total_records else None,
                },
                'uniqueTexts': {
                    'classified': unique_done, 'total': unique_total,
                    'remaining': remaining_unique,
                    'fraction': unique_done / unique_total if unique_total else None,
                },
                'labelsBySourceRecord': label_records,
                'labelsByUniqueText': label_unique,
            },
            'progress': progress,
            'sample': {
                'selection': 'At most two earliest source records per predicted label, one per exact text hash; deterministic, descriptive only.',
                'maxItems': MAX_SAMPLE,
                'items': samples,
            },
        }
    finally:
        conn.close()


def write_atomic(report: dict, output_path: str | Path = DEFAULT_OUTPUT) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp',
                                         delete=False) as temp:
            temp_name = temp.name
            temp.write(encoded)
            temp.flush()
            os.fsync(temp.fileno())
        os.replace(temp_name, path)
        temp_name = None
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    finally:
        if temp_name:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return path


def export(db_path: str | Path = DEFAULT_DB, output_path: str | Path = DEFAULT_OUTPUT,
           run_id: str | None = None, generated_at: str | None = None) -> dict:
    report = build_report(db_path, run_id, generated_at)
    write_atomic(report, output_path)
    return report


def watch(db_path: str | Path = DEFAULT_DB, output_path: str | Path = DEFAULT_OUTPUT,
          run_id: str | None = None, poll_seconds: int = 30,
          sleep_fn: Callable[[float], None] = time.sleep,
          on_poll: Callable[[dict], None] | None = None) -> dict:
    """Watch one fixed run; export final or explicit run-replaced status."""
    if poll_seconds < 1:
        raise ValueError('poll_seconds must be positive')
    first = build_report(db_path, run_id)
    target = first.get('runId')
    if run_id is not None and target != run_id:
        first['state'] = 'run_replaced'
        first['runReplaced'] = True
        first['activeRunIdAtExport'] = first.get('activeRunIdAtExport')
        write_atomic(first, output_path)
        return first
    report = first
    while report['state'] not in FINAL_STATES:
        if on_poll:
            on_poll(report)
        sleep_fn(poll_seconds)
        report = build_report(db_path, target)
        if report['runReplaced']:
            report['run'] = first['run']
            report['source'] = first['source']
            report['progress'] = first['progress']
            break
    write_atomic(report, output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run-id', help='Expected run ID; by default watch the active run at launch.')
    parser.add_argument('--wait', action='store_true', help='Poll every 30 seconds until this run is terminal.')
    args = parser.parse_args(argv)
    if args.wait:
        report = watch(args.db, args.output, args.run_id)
    else:
        report = export(args.db, args.output, args.run_id)
    print(json.dumps({'output': str(args.output), 'runId': report['runId'],
                      'state': report['state'], 'coverage': report['coverage']['sourceRecords']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
