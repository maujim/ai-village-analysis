"""Local source index and resumable model annotations, without loading model weights."""
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import viewer

HERE = Path(__file__).resolve().parent
DB = HERE / 'messages.sqlite'


def connect():
    conn = sqlite3.connect(str(DB), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA journal_mode=WAL')
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS messages (
      id TEXT PRIMARY KEY, day INTEGER NOT NULL, event_index INTEGER NOT NULL,
      date TEXT, timestamp TEXT, speaker TEXT, source_type TEXT, text TEXT,
      text_hash TEXT NOT NULL, UNIQUE(day,event_index));
    CREATE INDEX IF NOT EXISTS messages_day ON messages(day,event_index);
    CREATE INDEX IF NOT EXISTS messages_hash ON messages(text_hash);
    CREATE TABLE IF NOT EXISTS predictions (
      run_id TEXT NOT NULL, text_hash TEXT NOT NULL, label TEXT NOT NULL,
      top_score REAL, margin REAL, needs_review INTEGER, truncated INTEGER,
      result TEXT NOT NULL, PRIMARY KEY(run_id,text_hash));
    CREATE INDEX IF NOT EXISTS predictions_label ON predictions(run_id,label);
    ''')
    return conn


def meta(conn, key, value=None):
    if value is not None:
        conn.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', (key,json.dumps(value)))
        conn.commit()
        return value
    row = conn.execute('SELECT value FROM metadata WHERE key=?',(key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_cloud_progress(progress, conn=None):
    """Persist separate Modal-output progress; never changes local run/prediction state.

    The caller should count only complete, verified output shards. This helper
    stores a compact status snapshot, not per-message results.
    """
    if not isinstance(progress, dict):
        raise ValueError('cloud progress must be an object')
    state = progress.get('state')
    if state not in ('running', 'complete'):
        raise ValueError("cloud progress state must be 'running' or 'complete'")
    clean = {'state': state}
    for key in ('processed_cloud_outputs', 'queued_unique_at_start', 'total_unique'):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
        clean[key] = value
    updated = progress.get('updated_at')
    if not isinstance(updated, str) or len(updated) > 64:
        raise ValueError('updated_at must be an ISO timestamp')
    try:
        datetime.fromisoformat(updated.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('updated_at must be an ISO timestamp') from exc
    clean['updated_at'] = updated
    backend = progress.get('backend', 'Modal L4')
    if not isinstance(backend, str) or not backend.strip() or len(backend) > 80:
        raise ValueError('backend must be a short nonempty label')
    clean['backend'] = backend.strip()
    eta = progress.get('eta_estimate_seconds')
    if eta is not None:
        if isinstance(eta, bool) or not isinstance(eta, int) or eta < 0:
            raise ValueError('eta_estimate_seconds must be a nonnegative integer')
        clean['eta_estimate_seconds'] = eta
    owned = conn is None
    conn = conn or connect()
    try:
        meta(conn, 'cloud_progress', clean)
    finally:
        if owned:
            conn.close()
    return clean


def import_source():
    conn = connect()
    digest = hashlib.sha256()
    with viewer.TRANSCRIPT.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):
            digest.update(part)
    sha = digest.hexdigest()
    existing = meta(conn,'source_sha256')
    if existing and existing != sha:
        raise RuntimeError('Source snapshot changed; preserve this database and start a new one.')
    meta(conn,'source_sha256',sha)
    for entry in viewer.day_index()['days']:
        day = viewer.read_day(entry)
        rows = []
        for i,event in enumerate(day.get('events',[])):
            if event.get('type') not in ('AGENT_TALK','USER_TALK'):
                continue
            text = event.get('content') or event.get('message') or ''
            rows.append((f'ev-{sha[:12]}-d{day["day"]}-e{i}', day['day'], i,
                         day.get('date'), event.get('timestamp'),
                         event.get('speakerName') or event.get('agentName') or 'Unknown',
                         event['type'],text,hashlib.sha256(text.encode()).hexdigest()))
        conn.executemany('INSERT OR IGNORE INTO messages VALUES (?,?,?,?,?,?,?,?,?)',rows)
        conn.commit()
    total = conn.execute('SELECT count(*) FROM messages').fetchone()[0]
    meta(conn,'total_messages',total)
    print(json.dumps({'source_sha256':sha,'messages':total,
                      'unique_texts':conn.execute('SELECT count(DISTINCT text_hash) FROM messages').fetchone()[0]}))
    conn.close()


def status(conn):
    run = meta(conn,'active_run') or {}
    run_id = run.get('id','')
    total = conn.execute('SELECT count(*) FROM messages').fetchone()[0]
    done = conn.execute('SELECT count(*) FROM messages m JOIN predictions p ON m.text_hash=p.text_hash AND p.run_id=?',(run_id,)).fetchone()[0]
    counts = [dict(r) for r in conn.execute('''SELECT p.label,count(*) count,
        sum(p.needs_review) needs_review,sum(p.truncated) truncated
        FROM messages m JOIN predictions p ON m.text_hash=p.text_hash AND p.run_id=?
        GROUP BY p.label ORDER BY count DESC''',(run_id,))]
    progress=meta(conn,'progress') or {}
    if progress.get('state') in ('running','loading','starting','stopping') and progress.get('pid'):
        try:
            os.kill(int(progress['pid']),0)
        except ProcessLookupError:
            progress['state']='paused' if progress['state']=='stopping' else 'interrupted'
        except PermissionError:
            pass
    unique_total=conn.execute('SELECT count(DISTINCT text_hash) FROM messages').fetchone()[0]
    unique_done=conn.execute('SELECT count(*) FROM predictions WHERE run_id=?',(run_id,)).fetchone()[0]
    rate=progress.get('unique_texts_per_second',0)
    if rate and progress.get('completed_this_session',0)>=32:
        progress['estimated_remaining_seconds']=round(max(0,unique_total-unique_done)/rate)
    return {'total':total,'classified':done,'remaining':total-done,'counts':counts,
            'unique_total':unique_total,'unique_classified':unique_done,
            'source_sha256':meta(conn,'source_sha256'),'run':run,
            'progress':progress,'cloud_progress':meta(conn,'cloud_progress') or None}


def query(params):
    conn = connect()
    run = meta(conn,'active_run') or {}
    clauses=[]; args=[run.get('id','')]
    def add(clause, value):
        clauses.append(clause); args.append(value)
    for key,column in [('day','m.day'),('label','p.label'),('speaker','m.speaker')]:
        if params.get(key): add(column+'=?',params[key])
    if params.get('q'): add('instr(lower(m.text),lower(?))>0',params['q'])
    if params.get('id'): add('m.id=?',params['id'])
    if params.get('review')=='1': clauses.append('p.needs_review=1')
    if params.get('pending')=='1': clauses.append('p.label IS NULL')
    where=' WHERE '+' AND '.join(clauses) if clauses else ''
    join=' FROM messages m LEFT JOIN predictions p ON m.text_hash=p.text_hash AND p.run_id=?'
    offset=max(0,int(params.get('offset',0))); limit=min(100,max(1,int(params.get('limit',30))))
    total=conn.execute('SELECT count(*)'+join+where,args).fetchone()[0]
    rows=conn.execute('SELECT m.*,p.label,p.top_score,p.margin,p.needs_review,p.truncated,p.result'+join+where+
                      ' ORDER BY m.day,m.event_index LIMIT ? OFFSET ?',args+[limit,offset])
    records=[]
    for row in rows:
        record=dict(row)
        record['prediction']=json.loads(record.pop('result')) if record['result'] else None
        records.append(record)
    conn.close()
    return {'total':total,'offset':offset,'limit':limit,'messages':records}


if __name__=='__main__':
    import_source()
