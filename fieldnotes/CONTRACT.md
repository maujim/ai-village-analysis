# FIELDNOTES local v1 contract

Source: immutable local village-transcript.json, SHA256 from analysis/results.json.
All work stays local. AI review is not human review. No public release status may be invented.

## Files and ownership
- audit.py, manifest.json, identities.json: provenance worker.
- investigate.py, investigations.json: investigation worker.
- template.html: interface worker (self-contained CSS/JS; root embeds __FIELDNOTES_DATA__).
- build.py, bundle.json, index.html, viewer server integration: root.
- validate.py, review.json: later validation/review workers.

## Embedded data
`{manifest, observations: [...], review, adapters}`. No external dependencies.
Every observation must use the shape below; optional fields are allowed. Root hydrates event text, speaker, timestamps, IDs and links from raw locators. Source-derived text is always rendered as text, never HTML.

```
{
 id, title, question, summary, mechanism, boundary, implication,
 phenomenon: 'idea-lineage'|'artifact-pathway'|'boundary-case',
 status: 'exploratory', reviewStatus: 'awaiting-human-review',
 snapshotId, revision: '1', date, day,
 task, cohort: [raw speaker labels], scope: {start, end, prehistory, coverage},
 claims: [{id, text, qualifier}],
 events: [{id, day, index, speaker, text, timestamp, time, sourceOrder,
   timestampQuality, track:'expression'|'evidence'|'action'|'context',
   claimId, stance:'mention'|'endorsement'|'challenge'|'qualification'|'report',
   label, observationBasis:'direct-logged-event'|'explicit-agent-report'|'analyst-inference',
   outcomeStatus:'proposed'|'result-reported'|'unknown'|'contradicted',
   rootFamily: string|null, rootReason: string|null,
   isDocumentedCheck: boolean, isEndorsement: boolean,
   quote: exact short source span, link}],
 roots: [{id,label,basis,eventIds:[ids],caveat}],
 edges: [{id,source,target,type,explanation,evidenceEventIds:[ids],
   observationBasis,relationshipStrength:'explicit-attribution'|'candidate'|'unresolved',
   alternatives:[strings],reviewStatus:'ai-reviewed'|'unreviewed', quote}],
 artifacts: [{id,label,location,status,versions:[strings],eventIds:[ids],caveat}],
 alternatives:[strings], counterevidence:[{eventId,explanation}],
 unknowns:[strings], method:{version,selection,parameters},
 publication:{permissionStatus:'unconfirmed',redactionReview:'pending'}
}
```

Stable event id formula: `ev-<sha256 first12>-d<day>-e<zero-based index>`.
Root hydrate accepts event IDs as short `d315-e123` initially and remaps throughout.
No relationship direction can go back in source order; same-time explicit attribution may remain ordered by source index, never fabricated elapsed time.
`rootFamily` groups only explicitly attributed support, never URL matching alone. Unresolved events keep null. Separate reported checks may be shown as distinct documented *reports*, not independently evidenced outcomes.
At replay step N only events up to N and edges with both ends visible exist. Future checks must not justify earlier certainty.
Metrics calculated from visible events: endorsing speakers, reported checks, known-lineage endorsement occurrences / all endorsement occurrences; expose counts not a confidence score. No roots means 'No documented root in this coverage'.

## UI
One page FIELDNOTES: finding hero, compact atlas of 3 local exploratory observations, fixed replay lanes (Expression / Evidence / Action / Context), source evidence rail, argument+limits. Play/pause, previous/next, range scrub, claim+speaker filters, collapse repeated support, checks/artifact paths. Edge selection shows basis+alternatives+source quote. Unknown roots remain visible. Stable geometry while scrubbing, mobile stacked, keyboard accessible, reduced motion. Deep links `#observation=<id>&step=<n>`. Export evidence bundle JSON and static SVG card using Blob downloads. Local review form records decisions via POST /api/fieldnotes/reviews; GET same endpoint returns append-only records. Request body {observationId, revision, decision, reviewer, note}; UI labels human review self-reported, not authenticated. Existing browser / and analysis /analysis.html linked.
App URL /fieldnotes/. Root serves generated fieldnotes/index.html.
