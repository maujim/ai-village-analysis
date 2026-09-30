FIELDNOTES — local evidence investigations

Open http://127.0.0.1:8765/fieldnotes/ after starting python3 viewer.py
from the ai-village directory. The transcript browser remains at / and the
original descriptive analysis at /analysis.html.

Reproduce:
  python3 fieldnotes/audit.py
  python3 fieldnotes/investigate.py
  python3 fieldnotes/build.py
  python3 fieldnotes/validate.py

Audit freezes the source digest and import time. The scene build is deterministic
for the same saved manifest and annotations. Re-running the audit changes import
metadata; re-annotating is a new research revision, not a deterministic model run.
No model is run by these scripts. GPT-6-luna assisted source inspection and frozen
annotation authoring. Nano-Jev is not used in this release.

manifest.json is the complete inventory; bundle.json contains a compact audit
plus the three bounded investigations. Raw source remains unchanged. Each source
ID combines the source digest prefix, source day ID, and zero-based event index.
The server can return adjacent raw context via /api/fieldnotes/event?id=EVENT_ID.

All observations are exploratory and await human review. Local review decisions
append to reviews.jsonl; earlier decisions are never overwritten. Reviewer names
are self-reported and unauthenticated. A review decision is not publication
permission. The app and evidence exports are local research artifacts, not a
public release. Source redistribution and redaction status remain unconfirmed.

This is the first working vertical slice of the PRD. It does not implement
corpus-wide causal inference, model training, video export, a validated gold set,
or a claim that reported external actions have been independently verified.

Attribution: AI Digest / AI Village, aidigestorg/ai-village on Hugging Face.
