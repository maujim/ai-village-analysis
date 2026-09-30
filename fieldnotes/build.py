#!/usr/bin/env python3
"""Hydrate frozen annotations and build FIELDNOTES. No model or network calls."""
import copy
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import viewer
OUT = ROOT / 'fieldnotes'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def build():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    # Keep the full audit available as a separate local file; scenes need only its summary.
    compact_manifest = copy.deepcopy(manifest)
    compact_manifest['fullAuditUrl'] = '/fieldnotes/manifest.json'
    for section, bulky in {'coverage': ['days', 'sourceSpans', 'schemaShapeCounts'],
                           'identity': ['rawSpeakerLabels', 'agentAuthoredRawLabels', 'labels', 'allSpeakerLabels'],
                           'duplicates': ['examples'],
                           'artifacts': ['distinctURLCandidates', 'distinctPathCandidates']}.items():
        for key in bulky:
            value = compact_manifest.get(section, {}).pop(key, None)
            if value is not None:
                compact_manifest[section][key + 'Entries'] = len(value) if hasattr(value, '__len__') else value
    sha = manifest['source']['sha256']
    source_results = json.loads((ROOT / 'analysis/results.json').read_text())
    assert sha == source_results['source']['sha256'], 'Existing analysis belongs to a different snapshot'
    raw = json.loads((OUT / 'investigations.json').read_text())
    observations = copy.deepcopy(raw['observations'])
    presentation = json.loads((OUT / 'presentation.json').read_text())
    entries = {d['day']: d for d in viewer.day_index()['days']}
    cache = {}
    for obs in observations:
        display = presentation[obs['id']]
        obs['mechanism'] = display['mechanism']
        for event in obs['events']:
            event['label'] = display['labels'][str(event['index'])]
        aliases = {e['id']: f"ev-{sha[:12]}-d{e['day']}-e{e['index']}" for e in obs['events']}
        obs['snapshotId'] = sha
        obs['status'] = 'exploratory'
        obs['reviewStatus'] = 'awaiting-human-review'
        obs['publication'] = {'permissionStatus': 'unconfirmed', 'redactionReview': 'pending', 'distribution': 'local-only'}
        obs.setdefault('revision', '1')
        for e in obs['events']:
            if e['day'] not in cache:
                cache[e['day']] = viewer.read_day(entries[e['day']])
            day = cache[e['day']]
            source = day['events'][e['index']]
            text = source.get('content') or source.get('message') or source.get('goal') or source.get('thinking') or ''
            if e.get('quote') and e['quote'] not in text:
                raise ValueError(f"Non-verbatim evidence quote: {obs['id']} {e['id']}")
            e.update(id=aliases[e['id']], text=text, speaker=source.get('speakerName') or source.get('agentName') or 'Unknown',
                     timestamp=source.get('timestamp'), time=source.get('time'), date=day['date'],
                     sourceOrder=e['index'], sourceType=source.get('type'), rawTimestamp=source.get('timestamp'),
                     link=f"http://127.0.0.1:8765/?day={e['day']}&event={e['index']}&snapshot={sha}")
            try:
                dt = datetime.fromisoformat(source.get('timestamp', '').replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    e['timestampQuality'] = 'timezone-unknown'
                    e['utcTimestamp'] = None
                else:
                    e['timestampQuality'] = 'valid-offset'
                    e['utcTimestamp'] = dt.astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError):
                e['timestampQuality'] = 'missing-or-invalid'
                e['utcTimestamp'] = None
            e['locator'] = {'snapshotId': sha, 'day': e['day'], 'index': e['index']}
        obs['events'].sort(key=lambda e: (e['day'], e['index']))
        for i, event in enumerate(obs['events']):
            event['step'] = i
        for edge in obs.get('edges', []):
            edge['source'] = aliases.get(edge['source'], edge['source'])
            edge['target'] = aliases.get(edge['target'], edge['target'])
            edge['evidenceEventIds'] = [aliases.get(v, v) for v in edge['evidenceEventIds']]
            edge['reviewStatus'] = 'ai-reviewed' if edge.get('reviewStatus') == 'ai-reviewed' else 'unreviewed'
        for item in obs.get('roots', []) + obs.get('artifacts', []):
            item['eventIds'] = [aliases.get(v, v) for v in item.get('eventIds', [])]
        for item in obs.get('counterevidence', []):
            item['eventId'] = aliases.get(item['eventId'], item['eventId'])
        obs['annotationProvenance'] = {'annotator': 'gpt-6-luna', 'method': 'bounded source inspection and frozen annotations',
                                       'humanReviewed': False, 'editorialReview': 'primary agent source review and corrections', 'modelReannotationDeterministic': False}
        obs['scene'] = {'version': 'fixed-lanes-v1', 'clock': 'recorded event order', 'displayTimezone': 'UTC',
                        'steps': [{'step': i, 'eventIds': [e['id']]} for i, e in enumerate(obs['events'])],
                        'lanes': ['expression', 'evidence', 'action', 'context']}
    review_path = OUT / 'review.json'
    review = json.loads(review_path.read_text()) if review_path.exists() else {
        'status': 'ai-review-pending', 'humanReview': 'pending', 'note': 'No human review has been recorded.'}
    adapters = [{'id': k, 'version': 'legacy-adapter-v1', 'source': 'analysis/results.json',
                 'snapshotId': sha, 'methodVersion': source_results['methodVersion'], 'status': 'exploratory',
                 'outputCount': len(source_results[k]), 'originalPreserved': True}
                for k in ['days', 'edges', 'artifacts', 'duplicates', 'shifts']]
    bundle = {'manifest': compact_manifest, 'observations': observations, 'review': review, 'adapters': adapters}
    from validate import validate_bundle
    errors = validate_bundle(bundle, source_path=viewer.TRANSCRIPT)
    if errors:
        raise ValueError('Evidence validation failed: ' + str(errors))
    write_json(OUT / 'bundle.json', bundle)
    write_json(OUT / 'observations-index.json', [{k: o.get(k) for k in ['id', 'title', 'question', 'phenomenon', 'status', 'date', 'day', 'revision']} for o in observations])
    for obs in observations:
        write_json(OUT / (obs['id'] + '.json'), {'manifest': compact_manifest, 'observation': obs, 'review': review})
    template = (OUT / 'template.html').read_text()
    assert template.count('__FIELDNOTES_DATA__') == 1
    payload = json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(',', ':')).replace('<', '\\u003c')
    html = template.replace('__FIELDNOTES_DATA__', payload)
    (OUT / 'index.html').write_text(html)
    metrics = {'htmlBytes': len(html.encode()), 'gzipBytes': len(gzip.compress(html.encode(), mtime=0)),
               'observations': len(observations), 'selectedEvents': sum(len(o['events']) for o in observations),
               'deterministicSceneSha256': hashlib.sha256(json.dumps(observations, sort_keys=True).encode()).hexdigest()}
    write_json(OUT / 'build-report.json', metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    build()
