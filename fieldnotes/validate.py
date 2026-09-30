#!/usr/bin/env python3
"""Evidence-bundle validation shared by the FIELDNOTES builder and CLI.

The validator checks invariants that are meaningful for this transcript model;
it does not grade the truth of an agent's report or infer causal relationships.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import viewer

ID_RE = re.compile(r"^ev-([0-9a-f]{12})-d([^\s-]+)-e(\d+)$")
DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b", re.I)
TRACKS = {"expression", "evidence", "action", "context"}
STANCES = {"mention", "endorsement", "challenge", "qualification", "report"}
BASES = {"direct-logged-event", "explicit-agent-report", "analyst-inference"}
OUTCOMES = {"proposed", "result-reported", "unknown", "contradicted"}
EDGE_STRENGTHS = {"explicit-attribution", "candidate", "unresolved"}
READ_WORDS = re.compile(r"\b(?:verified[- ]read|observed[- ]read|consumed|consumption|read[- ]verified)\b", re.I)


def _order(event):
    return (event.get("day", -1), event.get("sourceOrder", event.get("index", -1)))


def _duration_claim(text):
    return bool(isinstance(text, str) and DURATION_RE.search(text))


def validate_observation(observation: dict, snapshot_id: str) -> list[str]:
    """Return actionable invariant violations for one hydrated observation."""
    errors: list[str] = []
    oid = observation.get("id", "<missing-id>")
    def err(message):
        errors.append(f"{oid}: {message}")

    for key in ("id", "title", "question", "summary", "phenomenon", "scope", "claims", "events", "roots", "edges", "artifacts"):
        if key not in observation:
            err(f"missing required field {key}")
    if observation.get("status") != "exploratory":
        err("status must remain exploratory")
    if observation.get("reviewStatus") != "awaiting-human-review":
        err("reviewStatus must remain awaiting-human-review")
    if observation.get("snapshotId") != snapshot_id:
        err("snapshotId does not match manifest source hash")
    if observation.get("phenomenon") not in {"idea-lineage", "artifact-pathway", "boundary-case"}:
        err("unknown phenomenon type")
    publication = observation.get("publication", {})
    if publication.get("permissionStatus") != "unconfirmed" or publication.get("redactionReview") != "pending":
        err("publication permission and redaction review must remain pending")

    claims = observation.get("claims") or []
    claim_ids = [c.get("id") for c in claims if isinstance(c, dict)]
    if len(claim_ids) != len(set(claim_ids)):
        err("claim IDs must be unique")
    claim_set = set(claim_ids)
    events = observation.get("events") or []
    ids = [e.get("id") for e in events if isinstance(e, dict)]
    if len(ids) != len(set(ids)):
        err("event IDs must be unique")
    event_by_id = {e.get("id"): e for e in events if isinstance(e, dict)}
    source_prefix = snapshot_id[:12]
    for e in events:
        eid = e.get("id", "<missing-event-id>")
        match = ID_RE.match(str(eid))
        if not match or match.group(1) != source_prefix or int(match.group(3)) != e.get("index") or str(e.get("day")) != match.group(2):
            err(f"event {eid} does not use the stable snapshot/day/index locator")
        if e.get("sourceOrder") != e.get("index"):
            err(f"event {eid} sourceOrder must equal its recorded source index")
        if e.get("track") not in TRACKS:
            err(f"event {eid} has invalid track")
        if e.get("stance") not in STANCES:
            err(f"event {eid} has invalid stance")
        if e.get("observationBasis") not in BASES:
            err(f"event {eid} has invalid observationBasis")
        if e.get("outcomeStatus") not in OUTCOMES:
            err(f"event {eid} has invalid outcomeStatus")
        if e.get("claimId") is not None and e.get("claimId") not in claim_set:
            err(f"event {eid} references unknown claimId")
        if bool(e.get("isEndorsement")) != (e.get("stance") == "endorsement"):
            err(f"event {eid} endorsement flag conflicts with stance")
        if e.get("stance") == "challenge" and e.get("isEndorsement"):
            err(f"challenge event {eid} cannot count as endorsement")
        quote = e.get("quote")
        text = e.get("text")
        if not quote:
            err(f"event {eid} needs a short exact source quote")
        elif not isinstance(text, str) or quote not in text:
            err(f"event {eid} quote is not an exact span of hydrated source text")
        if e.get("timestampQuality") in {"missing", "malformed", "timezone-unknown", "missing-or-invalid"} and e.get("utcTimestamp"):
            err(f"event {eid} has a normalized UTC time despite an invalid/missing timestamp quality")

    roots = observation.get("roots") or []
    root_by_id = {r.get("id"): r for r in roots if isinstance(r, dict)}
    for root in roots:
        if not root.get("basis") or not root.get("caveat"):
            err(f"root {root.get('id')} needs basis and caveat")
        if not root.get("eventIds"):
            err(f"root {root.get('id')} must cite event IDs")
        if any(eid not in event_by_id for eid in root.get("eventIds", [])):
            err(f"root {root.get('id')} cites an absent event")

    edges = observation.get("edges") or []
    for edge in edges:
        edge_id = edge.get("id", "<missing-edge-id>")
        source, target = event_by_id.get(edge.get("source")), event_by_id.get(edge.get("target"))
        if source is None or target is None:
            err(f"edge {edge_id} has a missing source or target event")
            continue
        if _order(source) > _order(target):
            err(f"edge {edge_id} points backward in source order")
        if edge.get("type") not in {"replies_to", "mentions", "attributes_to", "proposes", "endorses", "challenges", "reports_attempt", "observed_attempt", "reports_result", "evidences_result", "references_artifact", "observed_reads_version", "observed_modifies_version", "candidate_informed_by", "retracts", "contradicts", "strengthens_without_cited_new_evidence"}:
            err(f"edge {edge_id} has an unsupported relationship type")
        if edge.get("observationBasis") not in BASES or edge.get("relationshipStrength") not in EDGE_STRENGTHS:
            err(f"edge {edge_id} needs explicit observation basis and relationship strength")
        if edge.get("type") in {"observed_attempt", "observed_modifies_version", "observed_reads_version", "evidences_result"} and edge.get("observationBasis") != "direct-logged-event":
            err(f"edge {edge_id} type {edge.get('type')} requires direct logged evidence")
        if edge.get("type") == "candidate_informed_by" and edge.get("relationshipStrength") == "explicit-attribution":
            err(f"edge {edge_id} candidate_informed_by cannot be explicit attribution")
        evidence_ids = edge.get("evidenceEventIds", [])
        if not evidence_ids or any(ref not in event_by_id for ref in evidence_ids):
            err(f"edge {edge_id} must cite present evidence event IDs")
            continue
        # A later observation may not justify an earlier replay step.
        if any(_order(event_by_id[ref]) > _order(target) for ref in evidence_ids):
            err(f"edge {edge_id} uses future evidence relative to its target")
        q = edge.get("quote")
        if not q:
            err(f"edge {edge_id} needs a short exact source quote")
        elif not any(q in str(event_by_id[ref].get("text", "")) for ref in evidence_ids):
            err(f"edge {edge_id} quote is not an exact span of its cited evidence")
        target_quality = target.get("timestampQuality")
        source_time, target_time = source.get("utcTimestamp"), target.get("utcTimestamp")
        tied = source_time and target_time and source_time == target_time
        uncertain_time = not source_time or not target_time or tied or target_quality in {"missing", "malformed", "timezone-unknown", "missing-or-invalid"}
        duration_fields = [v for k, v in edge.items() if re.search(r"duration|elapsed", k, re.I) and v not in (None, "", False, 0)]
        narrative = " ".join(str(edge.get(k, "")) for k in ("explanation", "quote"))
        if uncertain_time and (duration_fields or _duration_claim(narrative)):
            err(f"edge {edge_id} claims elapsed duration despite missing or tied timestamps")
        if edge.get("relationshipStrength") == "explicit-attribution" and not edge.get("explanation"):
            err(f"edge {edge_id} needs an explanation for explicit attribution")

    # Lineage families are only allowed when grounded in a cited root and an
    # explicit attribution edge; a URL match or topical similarity is not enough.
    explicit_graph = {}
    for edge in edges:
        if edge.get("relationshipStrength") == "explicit-attribution" and edge.get("type") in {"attributes_to", "endorses", "mentions", "proposes", "replies_to"}:
            explicit_graph.setdefault(edge.get("source"), []).append(edge.get("target"))

    def has_explicit_root_path(root, target_event):
        starts = [rid for rid in root.get("eventIds", []) if rid in event_by_id and _order(event_by_id[rid]) <= _order(target_event)]
        pending, visited = list(starts), set()
        while pending:
            node = pending.pop(0)
            if node == target_event.get("id"):
                return True
            if node in visited:
                continue
            visited.add(node)
            for next_id in explicit_graph.get(node, []):
                if next_id in event_by_id and _order(event_by_id[next_id]) <= _order(target_event):
                    pending.append(next_id)
        return False

    for e in events:
        family = e.get("rootFamily")
        if family is None:
            continue
        root = root_by_id.get(family)
        if root is None:
            err(f"event {e.get('id')} names a rootFamily that does not exist")
            continue
        if not e.get("rootReason"):
            err(f"event {e.get('id')} rootFamily lacks an explicit justification")
        if e.get("id") not in root.get("eventIds", []) and not has_explicit_root_path(root, e):
            err(f"event {e.get('id')} has no time-ordered explicit attribution path from a cited root event")

    artifacts = observation.get("artifacts") or []
    for artifact in artifacts:
        aid = artifact.get("id", "<missing-artifact-id>")
        if any(eid not in event_by_id for eid in artifact.get("eventIds", [])):
            err(f"artifact {aid} cites an absent event")
        if READ_WORDS.search(str(artifact.get("status", ""))):
            err(f"artifact {aid} status upgrades a mention/report to verified reading or consumption")
        location = str(artifact.get("location", ""))

    # A local path seen in multiple records does not prove object identity.
    path_rows = {}
    for a in artifacts:
        loc = str(a.get("location", ""))
        if loc.startswith("/"):
            path_rows.setdefault(loc, []).append(a)
    for location, rows in path_rows.items():
        if len(rows) > 1 and any(not a.get("namespace") for a in rows):
            err(f"local path {location} is repeated without workspace identity; do not merge artifacts")

    # Scene steps are the reproducible replay sequence; no animation geometry
    # or invented elapsed-work clock is accepted as source evidence.
    scene = observation.get("scene")
    if scene is not None:
        if scene.get("clock") != "recorded event order":
            err("scene clock must use recorded event order")
        expected = sorted(events, key=_order)
        steps = scene.get("steps", [])
        flat_ids = [eid for step in steps for eid in step.get("eventIds", [])]
        expected_ids = [e.get("id") for e in expected]
        if flat_ids != expected_ids:
            err("scene steps must deterministically list each event once in source order")
        if any(step.get("step") != i for i, step in enumerate(steps)):
            err("scene step indices must be contiguous and deterministic")
    return errors


def validate_bundle(bundle: dict, source_path: str | Path | None = None) -> list[str]:
    """Validate the generated embedded bundle; empty list means it passed."""
    errors: list[str] = []
    if not isinstance(bundle, dict):
        return ["bundle must be an object"]
    manifest = bundle.get("manifest", {})
    snapshot = manifest.get("source", {}).get("sha256")
    if not isinstance(snapshot, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot):
        errors.append("manifest source SHA-256 is missing or malformed")
    source_days = {}
    if source_path is not None and snapshot:
        digest = hashlib.sha256()
        with Path(source_path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != snapshot:
            errors.append("manifest source SHA-256 does not match the frozen source file")
    if bundle.get("review", {}).get("humanReview") not in {"pending", "awaiting-human-review"}:
        errors.append("bundle review must remain pending human review")
    observations = bundle.get("observations")
    if not isinstance(observations, list) or not observations:
        errors.append("bundle needs a non-empty observations array")
        return errors
    observation_ids = [o.get("id") for o in observations]
    if len(observation_ids) != len(set(observation_ids)):
        errors.append("observation IDs must be unique")
    for observation in observations:
        errors.extend(validate_observation(observation, snapshot or ""))
    if source_path is not None and snapshot and not any("does not match the frozen source" in e for e in errors):
        # Compare every replay record to the exact source record. viewer's index
        # and read_day APIs keep this bounded to the selected day records.
        old_path = viewer.TRANSCRIPT
        try:
            viewer.TRANSCRIPT = Path(source_path)
            entries = {d["day"]: d for d in viewer.day_index()["days"]}
            for observation in observations:
                for event in observation.get("events", []):
                    day_label, index = event.get("day"), event.get("index")
                    try:
                        if day_label not in source_days:
                            source_days[day_label] = viewer.read_day(entries[day_label])
                        raw = source_days[day_label]["events"][index]
                    except (KeyError, IndexError, TypeError):
                        errors.append(f"{observation.get('id')}: event {event.get('id')} locator is absent from frozen source")
                        continue
                    source_text = raw.get("content") or raw.get("message") or raw.get("goal") or raw.get("thinking") or ""
                    source_speaker = raw.get("speakerName") or raw.get("agentName") or "Unknown"
                    checks = (("text", source_text), ("speaker", source_speaker), ("sourceType", raw.get("type")),
                              ("rawTimestamp", raw.get("timestamp")), ("time", raw.get("time")))
                    for field, expected in checks:
                        if event.get(field) != expected:
                            errors.append(f"{observation.get('id')}: event {event.get('id')} {field} differs from frozen source")
                    if event.get("quote") and event["quote"] not in source_text:
                        errors.append(f"{observation.get('id')}: event {event.get('id')} quote is absent from frozen source")
        finally:
            viewer.TRANSCRIPT = old_path
    for adapter in bundle.get("adapters", []):
        if adapter.get("snapshotId") != snapshot:
            errors.append(f"adapter {adapter.get('id')} is not pinned to the bundle snapshot")
    return errors


def _run_negative_fixtures() -> None:
    """Small fixtures prove the production validator rejects key failure modes."""
    snap = "a" * 64
    def fixture(ts1="2026-01-01T00:00:00Z", ts2="2026-01-01T00:01:00Z", text2="Adopted.", stance2="endorsement"):
        a = {"id": "ev-aaaaaaaaaaaa-d1-e1", "day": 1, "index": 1, "sourceOrder": 1, "speaker": "A", "text": "Source", "quote": "Source", "timestampQuality": "valid-offset", "utcTimestamp": ts1, "track": "expression", "claimId": "c1", "stance": "mention", "observationBasis": "explicit-agent-report", "outcomeStatus": "unknown", "rootFamily": None, "rootReason": None, "isDocumentedCheck": False, "isEndorsement": False}
        b = {"id": "ev-aaaaaaaaaaaa-d1-e2", "day": 1, "index": 2, "sourceOrder": 2, "speaker": "B", "text": text2, "quote": text2, "timestampQuality": "valid-offset" if ts2 else "missing", "utcTimestamp": ts2, "track": "expression", "claimId": "c1", "stance": stance2, "observationBasis": "explicit-agent-report", "outcomeStatus": "unknown", "rootFamily": None, "rootReason": None, "isDocumentedCheck": False, "isEndorsement": stance2 == "endorsement"}
        return {"id": "fixture", "title": "Fixture", "question": "Q", "summary": "S", "phenomenon": "idea-lineage", "status": "exploratory", "reviewStatus": "awaiting-human-review", "snapshotId": snap, "scope": {}, "claims": [{"id": "c1", "text": "Claim", "qualifier": ""}], "events": [a, b], "roots": [{"id": "r1", "basis": "bounded", "caveat": "limited", "eventIds": [a["id"]]}], "edges": [], "artifacts": [], "publication": {"permissionStatus": "unconfirmed", "redactionReview": "pending"}}
    def edge(source, target, evidence, explanation="", quote=""):
        return {"id": "e", "source": source, "target": target, "type": "endorses", "explanation": explanation, "evidenceEventIds": evidence, "observationBasis": "explicit-agent-report", "relationshipStrength": "explicit-attribution", "alternatives": [], "reviewStatus": "unreviewed", "quote": quote}

    obs = fixture()
    obs["edges"] = [edge("ev-aaaaaaaaaaaa-d1-e2", "ev-aaaaaaaaaaaa-d1-e1", ["ev-aaaaaaaaaaaa-d1-e2"])]
    assert any("backward" in e for e in validate_observation(obs, snap)), "future alleged adoption must fail"

    obs = fixture(stance2="challenge", text2="I reject the claim.")
    obs["events"][1]["isEndorsement"] = True
    assert any("endorsement" in e for e in validate_observation(obs, snap)), "challenge cannot be counted as endorsement"

    obs = fixture(ts1="2026-01-01T00:00:00Z", ts2="2026-01-01T00:00:00Z")
    obs["edges"] = [edge("ev-aaaaaaaaaaaa-d1-e1", "ev-aaaaaaaaaaaa-d1-e2", ["ev-aaaaaaaaaaaa-d1-e1"], "reported 2 minutes later")]
    assert any("duration" in e for e in validate_observation(obs, snap)), "tied times cannot support elapsed duration"
    obs = fixture(ts1=None, ts2="2026-01-01T00:01:00Z")
    obs["events"][0]["timestampQuality"] = "missing"; obs["events"][0]["utcTimestamp"] = None
    obs["edges"] = [edge("ev-aaaaaaaaaaaa-d1-e1", "ev-aaaaaaaaaaaa-d1-e2", ["ev-aaaaaaaaaaaa-d1-e1"], "reported 2 hours later")]
    assert any("duration" in e for e in validate_observation(obs, snap)), "missing time cannot support elapsed duration"

    obs = fixture()
    obs["artifacts"] = [{"id": "a", "location": "/tmp/result.csv", "status": "mentioned", "eventIds": ["ev-aaaaaaaaaaaa-d1-e1"]}, {"id": "b", "location": "/tmp/result.csv", "status": "mentioned", "eventIds": ["ev-aaaaaaaaaaaa-d1-e2"]}]
    assert any("workspace identity" in e for e in validate_observation(obs, snap)), "unscoped local paths cannot be merged"

    obs = fixture()
    obs["artifacts"] = [{"id": "a", "location": "https://example.invalid/a", "status": "verified-read", "eventIds": ["ev-aaaaaaaaaaaa-d1-e1"]}]
    assert any("reading or consumption" in e for e in validate_observation(obs, snap)), "a mention cannot become a verified read"

    obs = fixture()
    obs["edges"] = [{"id": "observed", "source": "ev-aaaaaaaaaaaa-d1-e1", "target": "ev-aaaaaaaaaaaa-d1-e2", "type": "observed_modifies_version", "explanation": "", "evidenceEventIds": ["ev-aaaaaaaaaaaa-d1-e1"], "observationBasis": "explicit-agent-report", "relationshipStrength": "explicit-attribution", "alternatives": [], "reviewStatus": "unreviewed"}]
    assert any("requires direct logged evidence" in e for e in validate_observation(obs, snap)), "reported work cannot be promoted to observed work"

    obs = fixture()
    obs["edges"] = [{"id": "candidate", "source": "ev-aaaaaaaaaaaa-d1-e1", "target": "ev-aaaaaaaaaaaa-d1-e2", "type": "candidate_informed_by", "explanation": "", "evidenceEventIds": ["ev-aaaaaaaaaaaa-d1-e1"], "observationBasis": "explicit-agent-report", "relationshipStrength": "explicit-attribution", "alternatives": [], "reviewStatus": "unreviewed"}]
    assert any("cannot be explicit attribution" in e for e in validate_observation(obs, snap)), "candidate transmission cannot be upgraded to explicit"

    obs = fixture()
    obs["events"][1]["rootFamily"] = "r1"; obs["events"][1]["rootReason"] = "same URL"
    assert any("explicit attribution path" in e for e in validate_observation(obs, snap)), "shared URL or incidental edge is not lineage"

    obs = fixture()
    obs["scene"] = {"clock": "recorded event order", "steps": [{"step": 0, "eventIds": ["ev-aaaaaaaaaaaa-d1-e2"]}, {"step": 1, "eventIds": ["ev-aaaaaaaaaaaa-d1-e1"]}]}
    assert any("deterministically" in e for e in validate_observation(obs, snap)), "scene sequence must be deterministic source order"

    bundle = {"manifest": {"source": {"sha256": snap}}, "review": {"humanReview": "pending"}, "observations": [fixture()]}
    import tempfile
    with tempfile.NamedTemporaryFile() as wrong_source:
        wrong_source.write(b"not the frozen source")
        wrong_source.flush()
        assert any("does not match the frozen source" in e for e in validate_bundle(bundle, wrong_source.name)), "source snapshot drift must fail"


class ValidatorFixtures(unittest.TestCase):
    def test_negative_fixtures(self):
        _run_negative_fixtures()


def main() -> int:
    if "--self-test" in sys.argv:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidatorFixtures)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    bundle_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "bundle.json"
    source_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "village-transcript.json"
    if not bundle_path.exists():
        print(f"Bundle not found: {bundle_path}", file=sys.stderr)
        return 2
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    errors = validate_bundle(bundle, source_path)
    if errors:
        print("FIELDNOTES validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"FIELDNOTES validation passed: {len(bundle['observations'])} observations, snapshot {bundle['manifest']['source']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
