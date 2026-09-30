"""Export aggregate sensitivity results, without copying source conversation text."""
import argparse
import json
import math
from pathlib import Path


def summarize(raw, controlled):
    provenance = ('sourceSha256', 'modelSha256', 'taxonomySha256')
    if any(raw[k] != controlled[k] for k in provenance):
        raise ValueError('Experiments must use the same frozen source/model/taxonomy')
    def index(data):
        if not data['complete'] or data['count'] != len(data['rows']):
            raise ValueError('Incomplete experiment')
        out = {}
        for row in data['rows']:
            key = (row['base_text_hash'], row['condition'])
            if key in out or len(row['scores']) != 12:
                raise ValueError('Duplicate row or missing scores')
            if any(not math.isfinite(s) or not 0 <= s <= 1 for s in row['scores'].values()):
                raise ValueError('Invalid score')
            if row['truncation']['truncated']:
                raise ValueError('Truncated input')
            out[key] = row
        return out
    a, b = index(raw), index(controlled)
    if a.keys() != b.keys():
        raise ValueError('Target sets differ')
    targets = {k[0] for k in a}
    if len(a) != 3 * len(targets):
        raise ValueError('Expected three conditions per target')
    top = lambda row: max(row['scores'], key=row['scores'].get)
    changes = {}
    for name, rows in [('rawBaseline', a), ('matchedFraming', b)]:
        changes[name] = {
            condition: sum(top(rows[(h, condition)]) != top(rows[(h, 'isolated')]) for h in targets)
            for condition in ('real_context', 'shuffled_context')
        }
    for h in targets:
        for condition in ('real_context', 'shuffled_context'):
            if a[(h, condition)]['text_hash'] != b[(h, condition)]['text_hash']:
                raise ValueError('Context inputs differ between controls')
    return {
        'targetCount': len(targets), 'scoredInputs': len(a) + len(b),
        'seed': 20260930, 'gpu': 'L4',
        'method': '128 short exact texts; two preceding same-day messages versus approximately length-matched earlier messages from different days; second trial uses identical empty-context wrapper for the isolated baseline.',
        'labelChanges': changes,
        'wrapperOnlyLabelChanges': sum(top(a[(h, 'isolated')]) != top(b[(h, 'isolated')]) for h in targets),
        'truncatedInputs': 0,
        'interpretation': 'Naively prepending context strongly changes proposed acts, including shuffled context. Evaluate a target-aware method before using conversation windows.',
        'limitations': ['Sensitivity, not accuracy: no human ground truth.', 'One model, fixed seed, short-message sample; not a corpus-wide finding.', 'The NLI model scores the entire premise, so surrounding messages can dominate.', 'Context length matching is approximate.'],
        'experiments': [{k: d[k] for k in (*provenance, 'experimentId', 'inputManifestSha256', 'resultSha256', 'performance')} for d in (raw, controlled)],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw', type=Path)
    p.add_argument('controlled', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(summarize(json.loads(a.raw.read_text()), json.loads(a.controlled.read_text())), indent=2) + '\n')

if __name__ == '__main__':
    main()
