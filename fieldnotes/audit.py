#!/usr/bin/env python3
"""Reproducible, local provenance audit for the immutable Village transcript."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viewer import TRANSCRIPT, load_days

PARSER_VERSION = "fieldnotes-audit/1.0.0"
KNOWN_TYPES = {
    "USER_TALK", "AGENT_TALK", "START_USING_COMPUTER", "STOP_USING_COMPUTER",
    "WAIT", "PAUSE", "REQUEST_HUMAN_HELPER", "CANCEL_REQUEST_FOR_HUMAN_HELPER",
    "SEARCH_HISTORY", "CONSOLIDATE",
}
URL_RE = re.compile(r"https?://[^\s\]\[()<>\"']+", re.I)
PATH_RE = re.compile(r"(?<![\w])(?:/[^\s,;()<>\"']{2,}|[\w.-]+\.(?:py|js|ts|json|csv|md|txt|pdf|html|ipynb|xlsx|docx))(?![\w])", re.I)
ID_KEYS = {"agentId", "agent_id", "speakerId", "speaker_id", "instanceId", "instance_id"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_info(path: Path) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(path.parent), *args], stderr=subprocess.DEVNULL, text=True).strip() or None
        except (OSError, subprocess.CalledProcessError):
            return None
    remote = git("config", "--get", "remote.origin.url")
    return {"name": "aidigestorg/ai-village", "remote": remote, "revision": git("rev-parse", "HEAD"),
            "revisionStatus": "local-working-tree-snapshot" if git("status", "--porcelain", "--", path.name) else "committed-or-clean"}


def timestamp_value(value):
    if not isinstance(value, str) or not value.strip():
        return None, "missing"
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None, "malformed-or-no-timezone"
        return parsed.astimezone(timezone.utc), "valid"
    except (ValueError, OverflowError):
        return None, "malformed"


def main() -> None:
    file_path = Path(TRANSCRIPT).resolve()
    type_counts, label_counts, missing_fields, field_presence = Counter(), Counter(), Counter(), Counter()
    labels = defaultdict(lambda: {"count": 0, "first": None, "last": None, "idFields": Counter()})
    agent_labels = defaultdict(lambda: {"count": 0, "first": None, "last": None, "idFields": Counter()})
    total = accepted = quarantined = messages = human = agent = activity = content_records = 0
    day_rows, dates, date_counts = [], set(), Counter()
    times, missing_time, malformed_time = [], 0, 0
    out_of_order = tied = 0
    dup = defaultdict(list)
    artifact_urls = Counter()
    artifact_paths = Counter()
    sample_keys = {}
    id_field_counts = Counter()
    first_last = []
    schema_shapes = Counter()
    for day_obj, day_start, day_end in load_days():
        day_label = day_obj.get("day")
        date = day_obj.get("date")
        dates.add(date)
        date_counts[str(date)] += 1
        evs = day_obj.get("events")
        if not isinstance(evs, list):
            evs = []
        day_times = []
        day_type_counts = Counter()
        day_accepted = day_quarantined = 0
        for index, event in enumerate(evs):
            total += 1
            locator = {"file": file_path.name, "day": day_label, "date": date,
                       "eventIndex": index, "dayByteStart": day_start, "dayByteEnd": day_end}
            if not isinstance(event, dict):
                quarantined += 1; day_quarantined += 1
                continue
            accepted += 1; day_accepted += 1
            typ = event.get("type")
            type_key = typ if isinstance(typ, str) and typ else "<missing>"
            type_counts[type_key] += 1; day_type_counts[type_key] += 1
            schema_shapes["|".join(sorted(event.keys()))] += 1
            sample_keys.setdefault(type_key, sorted(event.keys()))
            field_presence.update(event.keys())
            for key in ("type", "timestamp", "time", "speakerName", "agentName", "content"):
                if key not in event or event[key] is None or event[key] == "":
                    missing_fields[key] += 1
            speaker = event.get("speakerName", event.get("agentName"))
            if isinstance(speaker, str) and speaker.strip():
                label = speaker.strip(); label_counts[label] += 1
                info = labels[label]; info["count"] += 1
                info["first"] = info["first"] or locator
                info["last"] = locator
                if type_key == "AGENT_TALK":
                    ainfo = agent_labels[label]; ainfo["count"] += 1
                    ainfo["first"] = ainfo["first"] or locator
                    ainfo["last"] = locator
            for key, value in event.items():
                if key.lower() in {k.lower() for k in ID_KEYS} and value is not None:
                    id_field_counts[key] += 1
                    if isinstance(speaker, str): labels[speaker.strip()]["idFields"][key] += 1
                    if type_key == "AGENT_TALK" and isinstance(speaker, str): agent_labels[speaker.strip()]["idFields"][key] += 1
            if type_key == "AGENT_TALK": agent += 1; messages += 1
            elif type_key == "USER_TALK": human += 1; messages += 1
            else: activity += 1
            parsed, quality = timestamp_value(event.get("timestamp"))
            if quality == "missing": missing_time += 1
            elif quality != "valid": malformed_time += 1
            else:
                times.append(parsed); day_times.append((parsed, index))
            if isinstance(event.get("content"), str):
                content_records += 1
                content = event["content"]
                fingerprint = hashlib.sha256((str(type_key)+"\0"+str(speaker)+"\0"+content).encode("utf-8")).hexdigest()
                dup[fingerprint].append({**locator, "type": type_key, "speakerLabel": speaker,
                                         "contentBytes": len(content.encode("utf-8"))})
                for u in URL_RE.findall(content):
                    u = u.rstrip(".,!?;:")
                    artifact_urls[u] += 1
                for p in PATH_RE.findall(content):
                    artifact_paths[p] += 1
        day_times.sort(key=lambda x: x[1])
        out_of_order += sum(1 for a, b in zip(day_times, day_times[1:]) if b[0] < a[0])
        time_groups = Counter(t for t, _ in day_times)
        tied += sum(n - 1 for n in time_groups.values() if n > 1)
        day_rows.append({"day": day_label, "date": date, "eventCount": len(evs), "accepted": day_accepted,
                         "quarantined": day_quarantined, "types": dict(sorted(day_type_counts.items())),
                         "sourceSpan": {"byteStart": day_start, "byteEnd": day_end}})

    digest = sha256_file(file_path)
    import_time = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    duplicate_groups = [rows for rows in dup.values() if len(rows) > 1]
    duplicate_groups.sort(key=lambda rows: (-len(rows), rows[0]["day"], rows[0]["eventIndex"]))
    duplicate_count = sum(len(g) for g in duplicate_groups)
    url_distinct = sorted(artifact_urls)
    path_distinct = sorted(artifact_paths)
    source = {"file": file_path.name, "path": str(file_path), "bytes": file_path.stat().st_size,
              "sha256": digest, "importedAt": import_time, "parserVersion": PARSER_VERSION,
              "repo": repo_info(file_path)}
    results_source = json.loads((file_path.parent / "analysis/results.json").read_text()).get("source", {})
    findings_source = json.loads((file_path.parent / "analysis/findings.json").read_text()).get("source", {})
    analysis_provenance = {
        "analysis/results.json": {"declaredSHA256": results_source.get("sha256"), "declaredBytes": results_source.get("bytes"),
                                   "matchesAuditedSnapshot": results_source.get("sha256") == digest and results_source.get("bytes") == file_path.stat().st_size},
        "analysis/findings.json": {"declaredSHA256": findings_source.get("sha256"), "declaredBytes": findings_source.get("bytes"),
                                   "matchesAuditedSnapshot": None if not findings_source.get("sha256") else findings_source.get("sha256") == digest,
                                   "status": "declares source filename and descriptive byte count but no cryptographic digest" if not findings_source.get("sha256") else "digest present"},
        "analysis/review-sample.json": {"declaredSHA256": None, "matchesAuditedSnapshot": None,
                                        "status": "no source provenance block; treat as derivative/unverified"},
    }
    timing = {"timezone": "UTC", "timestampField": "timestamp", "preservedSourceStrings": True,
              "minimum": min(times).isoformat().replace("+00:00", "Z") if times else None,
              "maximum": max(times).isoformat().replace("+00:00", "Z") if times else None,
              "valid": len(times), "missing": missing_time, "malformedOrTimezoneMissing": malformed_time,
              "denominator": accepted, "withinDayOutOfOrderAdjacentPairs": out_of_order,
              "withinDayTiedAdditionalEvents": tied,
              "orderingRule": "source event index retained; timestamps do not create continuity or elapsed active-work time"}
    summary = {"events": total, "messages": messages, "human": human, "dayRecords": len(day_rows),
               "uniqueDates": len(dates), "agentLabels": len(agent_labels), "speakerLabels": len(labels), "accepted": accepted,
               "quarantined": quarantined, "types": dict(sorted(type_counts.items())), "agentAuthored": agent,
               "activityRecords": activity, "unknownTypes": {k: v for k, v in sorted(type_counts.items()) if k not in KNOWN_TYPES},
               "denominators": {"events": total, "eventTypes": accepted, "messages": messages,
                                "humanMessages": human, "agentMessages": agent, "activityRecords": activity,
                                "dayRecords": len(day_rows), "uniqueDates": len(dates),
                                "agentLabels": agent, "speakerLabels": accepted}}
    manifest = {
        "version": "fieldnotes-manifest/1.0", "status": "audited-local-snapshot", "source": source,
        "summary": summary, "timing": timing,
        "coverage": {"days": day_rows, "sourceSpans": "day-level byte spans from viewer.load_days(); event locator is day plus zero-based event index",
                     "acceptedLocatorRule": "<source SHA256>:d<source day label>-e<zero-based event index>",
                     "acceptedLocatorExample": f"{digest[:12]}:d{day_rows[0]['day']}-e0" if day_rows else None,
                     "quarantinePolicy": "non-object event values are quarantined; all other parsed event objects are accepted, including incomplete/unknown types",
                     "missingFields": {"coreFieldMissingCounts": dict(sorted(missing_fields.items())),
                                       "allObservedFields": {key: {"present": n, "missing": accepted - n} for key, n in sorted(field_presence.items())},
                                       "denominator": accepted},
                     "eventSchemaSampleKeysByType": sample_keys,
                     "schemaShapeCounts": dict(schema_shapes.most_common(20))},
        "identity": {"rawSpeakerLabelCount": len(labels), "rawSpeakerLabels": sorted(labels),
                     "agentAuthoredRawLabelCount": len(agent_labels), "agentAuthoredRawLabels": sorted(agent_labels),
                     "labels": {name: {"agentAuthoredEventCount": v["count"], "firstAgentAuthoredEvent": v["first"], "lastAgentAuthoredEvent": v["last"],
                                       "idFieldEvidence": dict(v["idFields"])} for name, v in sorted(agent_labels.items())},
                     "allSpeakerLabels": {name: {"eventCount": v["count"], "firstObserved": v["first"], "lastObserved": v["last"]} for name, v in sorted(labels.items())},
                     "observedAgentIdentifierFields": dict(id_field_counts),
                     "configurationMetadataAvailability": {key: field_presence.get(key, 0) for key in
                         ("model", "modelName", "model_name", "scaffold", "scaffolding", "regime", "version")},
                     "resolvedAgentInstanceCount": 0, "aliasMerges": [],
                     "limitation": "speaker labels are observations, not resolved agent instances; aliases and continuity remain unresolved without explicit identity metadata"},
        "duplicates": {"method": "exact SHA-256 over event type, raw speaker label, and full content; candidates only",
                       "groups": len(duplicate_groups), "candidateRecords": duplicate_count,
                       "examples": duplicate_groups[:200]},
        "artifacts": {"URLMentionOccurrences": sum(artifact_urls.values()), "distinctURLStrings": len(url_distinct),
                      "distinctURLCandidates": url_distinct, "pathMentionOccurrences": sum(artifact_paths.values()),
                      "distinctPathCandidates": path_distinct,
                      "contentRecordsScanned": content_records, "denominator": content_records,
                      "candidateRule": "conservative textual URL/path matches only; no fetch, existence check, identity canonicalization, or confirmed artifacts"},
        "existingAnalyses": analysis_provenance,
        "limitations": ["A day record is not evidence of uninterrupted operation.",
                        "No continuity, availability, causal influence, or agent identity is inferred from labels or gaps.",
                        "Activity records are classified by event type; reported actions are not independently verified.",
                        "URL and path matches are mentions, not proof that an artifact existed, was read, or was reused.",
                        "The local revision and transcript content may differ from upstream; no remote state was fetched."]}
    output = Path(__file__).resolve().parent
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    identities = {"version": "fieldnotes-identities/1.0", "sourceSHA256": digest,
                  "rawSpeakerLabels": manifest["identity"]["labels"], "resolvedAgentInstances": [],
                  "aliasMerges": [], "policy": manifest["identity"]["limitation"]}
    (output / "identities.json").write_text(json.dumps(identities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"source": source, "summary": summary, "timing": timing,
                      "identityFields": dict(id_field_counts), "duplicateGroups": len(duplicate_groups),
                      "distinctURLCandidates": len(url_distinct), "distinctPathCandidates": len(path_distinct),
                      "existingAnalysisMatches": manifest["existingAnalyses"]["analysis/results.json"]["matchesAuditedSnapshot"]}, indent=2))


if __name__ == "__main__":
    main()
