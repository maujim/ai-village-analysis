#!/usr/bin/env python3
"""A tiny local browser for the large AI Village transcript."""
import json
import os
import re
import threading
import hashlib
import signal
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
TRANSCRIPT = ROOT / "village-transcript.json"
INDEX = ROOT / ".viewer-index.json"
PAGE_SIZE = 100
REVIEW_LOCK = threading.Lock()
SOURCE_HASH_CACHE = {}


def matches_snapshot(expected):
    stat = TRANSCRIPT.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    if key not in SOURCE_HASH_CACHE:
        digest = hashlib.sha256()
        with TRANSCRIPT.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        SOURCE_HASH_CACHE.clear()
        SOURCE_HASH_CACHE[key] = digest.hexdigest()
    return SOURCE_HASH_CACHE[key] == expected


def fieldnotes_bundle():
    return json.loads((ROOT / 'fieldnotes/bundle.json').read_text())


def review_records():
    path = ROOT / 'fieldnotes/reviews.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def load_days():
    """Read one day at a time so the full file isn't held in memory."""
    decoder = json.JSONDecoder()
    with TRANSCRIPT.open("r", encoding="utf-8") as f:
        buffer = f.read(1024 * 1024)
        match = re.search(r'"days"\s*:\s*\[', buffer)
        if not match:
            raise ValueError('Could not find the transcript "days" array')
        buffer_start = len(buffer[:match.end()].encode("utf-8"))
        buffer = buffer[match.end():]
        while True:
            trimmed = buffer.lstrip()
            buffer_start += len(buffer[:len(buffer) - len(trimmed)].encode("utf-8"))
            buffer = trimmed
            if buffer.startswith(","):
                buffer = buffer[1:]
                buffer_start += 1
            if buffer.startswith("]"):
                break
            if not buffer:
                buffer += f.read(256 * 1024)
                if not buffer:
                    break
            try:
                day, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = f.read(256 * 1024)
                if not more:
                    raise
                buffer += more
                continue
            day_end = buffer_start + len(buffer[:end].encode("utf-8"))
            yield day, buffer_start, day_end
            buffer_start = day_end
            buffer = buffer[end:]


def day_index():
    if INDEX.exists() and INDEX.stat().st_mtime >= TRANSCRIPT.stat().st_mtime:
        cached = json.loads(INDEX.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and "days" in cached and "types" in cached:
            return cached
    entries = []
    types = set()
    for day, start, end in load_days():
        entries.append({"day": day.get("day"), "date": day.get("date"), "count": len(day.get("events", [])), "start": start, "end": end})
        types.update(event.get("type") for event in day.get("events", []) if event.get("type"))
    index = {"days": entries, "types": sorted(types)}
    INDEX.write_text(json.dumps(index), encoding="utf-8")
    return index


def read_day(entry):
    with TRANSCRIPT.open("rb") as f:
        f.seek(entry["start"])
        return json.loads(f.read(entry["end"] - entry["start"]))


class Handler(BaseHTTPRequestHandler):
    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ('/signatures', '/signatures/'):
            body = (ROOT / 'message_signatures/index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path.startswith('/api/signatures/'):
            from message_signatures import store
            try:
                conn = store.connect()
                try:
                    expected = store.meta(conn, 'source_sha256')
                finally:
                    conn.close()
                if expected and not matches_snapshot(expected):
                    self.send_json({'error':'Signature index belongs to a different transcript snapshot'},409)
                    return
                if parsed.path == '/api/signatures/taxonomy':
                    self.send_json(json.loads((ROOT / 'message_signatures/taxonomy.json').read_text()))
                elif parsed.path == '/api/signatures/status':
                    conn = store.connect()
                    try:
                        self.send_json(store.status(conn))
                    finally:
                        conn.close()
                elif parsed.path == '/api/signatures/messages':
                    self.send_json(store.query({k:v[0] for k,v in parse_qs(parsed.query).items()}))
                else:
                    self.send_json({'error':'Not found'},404)
            except (ValueError, TypeError) as exc:
                self.send_json({'error':str(exc)},400)
            return
        if parsed.path in ('/fieldnotes', '/fieldnotes/'):
            body = (ROOT / 'fieldnotes/index.html').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == '/api/fieldnotes/reviews':
            self.send_json({'reviews': review_records()})
            return
        if parsed.path == '/api/fieldnotes/event':
            query = parse_qs(parsed.query)
            event_id = query.get('id', [''])[0]
            bundle = fieldnotes_bundle()
            if not matches_snapshot(bundle['manifest']['source']['sha256']):
                self.send_json({'error': 'Source snapshot changed; rebuild the investigation before reading evidence'}, 409)
                return
            prefix = bundle['manifest']['source']['sha256'][:12]
            match = re.fullmatch(r'ev-' + prefix + r'-d(\d+)-e(\d+)', event_id)
            if not match:
                self.send_json({'error': 'Invalid snapshot event locator'}, 400)
                return
            day_id, index = map(int, match.groups())
            entry = next((d for d in day_index()['days'] if d['day'] == day_id), None)
            if entry is None:
                self.send_json({'error': 'Day not found'}, 404)
                return
            day = read_day(entry)
            if index >= len(day['events']):
                self.send_json({'error': 'Event not found'}, 404)
                return
            self.send_json({'id': event_id, 'snapshotId': bundle['manifest']['source']['sha256'],
                            'day': day_id, 'index': index, 'date': day['date'],
                            'event': day['events'][index], 'context': [
                                {'id': f'ev-{prefix}-d{day_id}-e{i}', 'index': i, 'event': day['events'][i]}
                                for i in range(max(0, index - 2), min(len(day['events']), index + 3))]})
            return
        if parsed.path.startswith('/fieldnotes/'):
            name = parsed.path[len('/fieldnotes/'):]
            allowed = {'manifest.json', 'bundle.json', 'observations-index.json', 'build-report.json', 'review.json'}
            allowed.update(o['id'] + '.json' for o in fieldnotes_bundle()['observations'])
            if name not in allowed:
                self.send_json({'error': 'Not found'}, 404)
                return
            path = ROOT / 'fieldnotes' / name
            if not path.is_file():
                self.send_json({'error': 'Not built'}, 404)
                return
            self.send_json(json.loads(path.read_text()))
            return
        if parsed.path in ("/", "/index.html", "/analysis.html"):
            body = (ROOT / ("analysis.html" if parsed.path == "/analysis.html" else "index.html")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/api/days":
            self.send_json(day_index())
            return
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            if query.get('snapshot') and not matches_snapshot(query['snapshot'][0]):
                self.send_json({'error': 'The linked source snapshot no longer matches the local transcript'}, 409)
                return
            requested = int(query.get("day", ["1"])[0])
            offset = max(0, int(query.get("offset", ["0"])[0]))
            needle = query.get("q", [""])[0].casefold().strip()
            event_type = query.get("type", [""])[0]
            speaker = query.get("speaker", [""])[0]
            event_index = query.get("event", [""])[0]
            index = day_index()
            entry = next((d for d in index["days"] if int(d.get("day", -1)) == requested), None)
            if entry is None:
                self.send_json({"error": "Day not found"}, 404)
                return
            selected = read_day(entry)
            matches = []
            for ordinal, event in enumerate(selected.get("events", [])):
                if event_index and str(ordinal) != event_index:
                    continue
                name = event.get("speakerName", event.get("agentName", ""))
                if event_type and event.get("type") != event_type:
                    continue
                if speaker and name != speaker:
                    continue
                if needle and not any(needle in value.casefold() for value in [json.dumps(event, ensure_ascii=False)] + [v for v in event.values() if isinstance(v, str)]):
                    continue
                matches.append(event)
            self.send_json({
                "day": selected.get("day"), "date": selected.get("date"),
                "total": len(matches), "offset": offset, "pageSize": PAGE_SIZE,
                "events": matches[offset:offset + PAGE_SIZE],
                "speakers": sorted({e.get("speakerName", e.get("agentName", "")) for e in selected.get("events", []) if e.get("speakerName", e.get("agentName", ""))}),
            })
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path == '/api/signatures/control':
            from message_signatures import store
            host=self.headers.get('Host','')
            origin=self.headers.get('Origin')
            if host not in ('127.0.0.1:8765','localhost:8765') or (origin and origin != 'http://'+host):
                self.send_json({'error':'Local same-origin requests only'},403)
                return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<2048: raise ValueError('Invalid request size')
                payload=json.loads(self.rfile.read(size))
                action=payload.get('action')
                if action not in ('pause','resume'): raise ValueError('Unknown control action')
                with REVIEW_LOCK:
                    conn=store.connect()
                    try:
                        expected=store.meta(conn,'source_sha256')
                        if not expected or not matches_snapshot(expected):
                            self.send_json({'error':'Rebuild the source index before starting inference'},409)
                            return
                        current=store.status(conn)['progress']
                        active=current.get('state') in ('running','loading','starting','stopping')
                        if action=='pause':
                            if not active:
                                self.send_json({'state':'already-stopped'})
                                return
                            os.kill(int(current['pid']),signal.SIGTERM)
                            current['state']='stopping'
                            store.meta(conn,'progress',current)
                            self.send_json({'state':'stopping'})
                            return
                        if active:
                            self.send_json({'state':'already-running'})
                            return
                        executable=ROOT/'.venv-signatures/bin/python'
                        if not executable.exists(): raise ValueError('Local model environment is not installed')
                        env=os.environ.copy()
                        env.update(HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
                        with (ROOT/'message_signatures/run.log').open('ab') as log:
                            process=subprocess.Popen([str(executable),str(ROOT/'message_signatures/run.py'),
                                '--engine','nli','--device','auto','--batch','8'],cwd=str(ROOT),env=env,
                                stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                        store.meta(conn,'progress',{'state':'starting','pid':process.pid,
                                   'updated_at':datetime.now(timezone.utc).isoformat()})
                        self.send_json({'state':'starting','pid':process.pid},202)
                    finally:
                        conn.close()
            except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:
                self.send_json({'error':str(exc)},400)
            except ProcessLookupError:
                self.send_json({'state':'already-stopped'})
            return
        if urlparse(self.path).path != '/api/fieldnotes/reviews':
            self.send_json({'error': 'Not found'}, 404)
            return
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        if host not in ('127.0.0.1:8765', 'localhost:8765') or (origin and origin != 'http://' + host):
            self.send_json({'error': 'Local same-origin request required'}, 403)
            return
        if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
            self.send_json({'error': 'JSON required'}, 415)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Review must be under 16 KiB')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Review must be an object')
            obs = next((o for o in fieldnotes_bundle()['observations'] if o['id'] == payload.get('observationId')), None)
            if obs is None or str(payload.get('revision')) != str(obs['revision']):
                raise ValueError('Unknown observation or stale revision')
            decision = payload.get('decision')
            if decision not in ('approve', 'reject', 'needs-changes', 'approved', 'rejected', 'needs_changes'):
                raise ValueError('Invalid review decision')
            reviewer = payload.get('reviewer', '').strip()
            note = payload.get('note', '').strip()
            if not reviewer or not note or len(reviewer) > 100 or len(note) > 8000:
                raise ValueError('A reviewer name and review note are required')
            with REVIEW_LOCK:
                history = review_records()
                record = {'id': f'review-{len(history)+1}', 'observationId': obs['id'], 'revision': obs['revision'],
                          'snapshotId': obs['snapshotId'], 'decision': decision, 'reviewer': reviewer,
                          'note': note, 'timestamp': datetime.now(timezone.utc).isoformat(),
                          'identityStatus': 'self-reported local reviewer; not authenticated', 'publicationApproval': False}
                with (ROOT / 'fieldnotes/reviews.jsonl').open('a', encoding='utf-8') as out:
                    out.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.send_json({'review': record, 'reviews': history + [record]}, 201)
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError) as error:
            self.send_json({'error': str(error)}, 400)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    print("AI Village viewer: http://127.0.0.1:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
