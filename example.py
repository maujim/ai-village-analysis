#!/usr/bin/env python3
"""
A tiny, dependency-free starting point for the AI Village dataset.

It reads `village-transcript.json` (the chat + activity timeline) and counts how
often Claude vs non-Claude agents say "genuinely" in chat, by month — a fun look
at one of Claude's well-known verbal tics. No packages, no setup:

    python example.py                         # looks for ./village-transcript.json
    python example.py path/to/village-transcript.json
"""
import json
import sys
from collections import defaultdict

WORD = "genuinely"
path = sys.argv[1] if len(sys.argv) > 1 else "village-transcript.json"
transcript = json.load(open(path))

# month ("YYYY-MM") -> {"Claude": count, "Other": count}
counts = defaultdict(lambda: {"Claude": 0, "Other": 0})
for day in transcript["days"]:
    for event in day["events"]:
        if event.get("type") != "AGENT_TALK":
            continue  # only agent chat messages
        if WORD not in (event.get("content") or "").lower():
            continue
        month = event["timestamp"][:7]
        family = "Claude" if "claude" in event["speakerName"].lower() else "Other"
        counts[month][family] += 1

print(f'Chat messages containing "{WORD}", by month:\n')
print(f'{"month":<9}{"Claude":>8}{"Other":>8}')
for month in sorted(counts):
    row = counts[month]
    print(f'{month:<9}{row["Claude"]:>8}{row["Other"]:>8}')

claude_total = sum(r["Claude"] for r in counts.values())
other_total = sum(r["Other"] for r in counts.values())
print("-" * 25)
print(f'{"total":<9}{claude_total:>8}{other_total:>8}')
if other_total:
    print(f'\nClaude says "{WORD}" {claude_total / other_total:.1f}x as often as everyone else combined.')
