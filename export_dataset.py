#!/usr/bin/env python3
"""Create a reproducible, read-only snapshot for the AI Village analysis dataset.

This exports indexed chat records and current experimental predictions only. It
never opens a model, writes to the source database, or includes source archives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_DEFAULT = ROOT / "message_signatures" / "messages.sqlite"
OUT_DEFAULT = ROOT.parent / "ai-village-analyzed-dataset"
NOTICE = (
    "Exploratory, incomplete, unvalidated model annotations. Categories and "
    "hypotheses are author-defined. Scores are not calibrated and have not "
    "been validated against human annotations. A predicted label is not task "
    "extraction, source verification, or evidence that a reported action succeeded."
)


def read_meta(conn, key):
    row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prune_audit(value, key=""):
    """Retain audit summaries while dropping record-level examples/path data."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            low = str(k).lower()
            if low in {"examples", "samples", "sample", "records", "raw", "rawtext", "fulltext"}:
                continue
            if low in {"path", "file", "absolutepath", "sourcepath", "model_dir", "workdir"}:
                continue
            out[k] = prune_audit(v, k)
        return out
    if isinstance(value, list):
        if len(value) > 40:
            return {"omitted_detail_count": len(value), "note": "Detailed entries omitted from public compact audit export."}
        return [prune_audit(v, key) for v in value]
    if isinstance(value, str) and len(value) > 4000:
        return value[:4000] + " [truncated in compact public audit export]"
    return value


def compact_prediction(row):
    if row["result"] is None:
        return None
    try:
        result = json.loads(row["result"])
    except (TypeError, json.JSONDecodeError):
        result = {"parse_error": True}
    # Keep all prediction fields used in the UI and audit; discard unknown
    # internal payloads to prevent accidental model/runtime data leakage.
    allowed = (
        "primary_candidate", "primary_act", "score_semantics", "scores",
        "candidate_templates", "template", "template_status", "slot_values",
        "extraction_performed", "context_used", "multi_act_status",
        "review_reasons", "truncation",
    )
    out = {k: result[k] for k in allowed if k in result}
    out.update({
        "label": row["label"], "top_score": row["top_score"],
        "margin": row["margin"], "needs_review": bool(row["needs_review"]),
        "truncated": bool(row["truncated"]),
    })
    for key in ("run_id", "execution_epoch_id"):
        if key in result:
            out[key] = result[key]
    runtime = result.get("runtime")
    if isinstance(runtime, dict) and "truncation" in runtime:
        out["runtime_truncation"] = runtime["truncation"]
    return out


def export(db_path: Path, out: Path, shard_size=25000):
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty staging directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    data_dir.mkdir(exist_ok=True)
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("BEGIN")
    try:
        active = read_meta(conn, "active_run") or {}
        progress = read_meta(conn, "progress") or {}
        source_sha = read_meta(conn, "source_sha256")
        total = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        unique_total = conn.execute("SELECT count(DISTINCT text_hash) FROM messages").fetchone()[0]
        run_id = str(active.get("id") or "")
        if run_id:
            unique_pred = conn.execute(
                "SELECT count(*) FROM predictions WHERE run_id=? AND text_hash IN (SELECT DISTINCT text_hash FROM messages)",
                (run_id,),
            ).fetchone()[0]
        else:
            unique_pred = 0
        label_counts = [dict(r) for r in conn.execute("""
            SELECT p.label AS label, count(*) AS source_records,
                   sum(p.needs_review) AS needs_review, sum(p.truncated) AS truncated
            FROM messages m JOIN predictions p ON p.text_hash=m.text_hash AND p.run_id=?
            GROUP BY p.label ORDER BY source_records DESC, p.label
        """, (run_id,))] if run_id else []
        classified = sum(x["source_records"] for x in label_counts)
        run_state = str(progress.get("state") or "unknown")
        if run_state == "complete" and classified < total:
            run_state = "incomplete"
        generated = datetime.now(timezone.utc).isoformat()
        run_record = prune_audit(dict(active))
        manifest = {
            "schema_version": "1.0",
            "dataset": "AI Village analyzed chat messages",
            "generated_at_utc": generated,
            "notice": NOTICE,
            "source": {
                "name": "AI Village transcript chat records",
                "sha256": source_sha,
                "records": total,
                "unique_exact_texts": unique_total,
                "indexed_database": "message_signatures/messages.sqlite (not distributed)",
            },
            "snapshot": {
                "state": run_state,
                "run_id": run_id or None,
                "execution_epoch_id": progress.get("execution_epoch_id"),
                "execution_epoch_number": progress.get("execution_epoch_number"),
                "classified_source_records": classified,
                "pending_source_records": total - classified,
                "classified_unique_texts": unique_pred,
                "pending_unique_texts": max(0, unique_total - unique_pred),
                "coverage_fraction_source_records": classified / total if total else 0,
                "coverage_fraction_unique_texts": unique_pred / unique_total if unique_total else 0,
                "predicted_label_counts_source_records": label_counts,
                "progress_metadata": {k: progress.get(k) for k in (
                    "completed_this_session", "elapsed_seconds", "unique_texts_per_second", "updated_at"
                ) if k in progress},
            },
            "model_run": run_record,
            "human_review": {"status": "not performed", "validated_labels": 0},
            "duplicate_policy": "One row per source record; identical exact text records share their text-hash-level prediction, but retain separate message IDs and source locators.",
            "score_interpretation": "Model-dependent scores are advisory, uncalibrated, and must not be interpreted as probabilities of correctness. Independent entailment scores need not sum to one.",
            "files": [],
        }
        query = conn.execute("""
            SELECT m.id,m.day,m.event_index,m.date,m.timestamp,m.speaker,m.source_type,m.text,m.text_hash,
                   p.label,p.top_score,p.margin,p.needs_review,p.truncated,p.result
            FROM messages m LEFT JOIN predictions p ON p.text_hash=m.text_hash AND p.run_id=?
            ORDER BY m.day,m.event_index
        """, (run_id,))
        shard_num = 0
        rows_in_shard = 0
        gz = None
        binary = None
        current_path = None
        exported = 0
        for r in query:
            if gz is None or rows_in_shard >= shard_size:
                if gz:
                    gz.close()
                    binary.close()
                    manifest["files"].append({"path": f"data/{current_path.name}", "records": rows_in_shard,
                                              "bytes": current_path.stat().st_size, "sha256": sha_file(current_path)})
                shard_num += 1
                rows_in_shard = 0
                current_path = data_dir / f"messages-{shard_num:03d}.jsonl.gz"
                binary = open(current_path, "wb")
                gz = gzip.GzipFile(filename="", mode="wb", fileobj=binary, compresslevel=6, mtime=0)
                # Text wrapper is kept separate so the gzip header is deterministic.
                import io
                gz = io.TextIOWrapper(gz, encoding="utf-8", newline="\n")
            item = {
                "message_id": r["id"],
                "source_locator": {"day": r["day"], "event_index": r["event_index"], "date": r["date"], "timestamp": r["timestamp"]},
                "source_snapshot_sha256": source_sha,
                "speaker_label": r["speaker"], "source_type": r["source_type"],
                "text_hash_sha256": r["text_hash"], "text": r["text"],
                "prediction": compact_prediction(r),
            }
            gz.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            exported += 1
            rows_in_shard += 1
        if gz:
            gz.close()
            binary.close()
            manifest["files"].append({"path": f"data/{current_path.name}", "records": rows_in_shard,
                                      "bytes": current_path.stat().st_size, "sha256": sha_file(current_path)})
        if exported != total:
            raise RuntimeError(f"snapshot row accounting mismatch: exported {exported}, expected {total}")
        analysis = {}
        results_path = ROOT / "analysis" / "results.json"
        findings_path = ROOT / "analysis" / "findings.json"
        if results_path.exists():
            raw = json.loads(results_path.read_text(encoding="utf-8"))
            keep = ("title", "methodVersion", "generatedAt", "source", "summary", "limitations")
            analysis["results"] = {k: raw[k] for k in keep if k in raw}
        if findings_path.exists():
            raw = json.loads(findings_path.read_text(encoding="utf-8"))
            analysis["findings"] = raw
        if analysis:
            write_json(out / "analysis-summary.json", analysis)
        fieldnotes = ROOT / "fieldnotes"
        for src_name, dest_name in (("manifest.json", "fieldnotes-provenance.json"), ("bundle.json", "fieldnotes-evidence-bundle.json")):
            p = fieldnotes / src_name
            if not p.exists():
                continue
            raw = prune_audit(json.loads(p.read_text(encoding="utf-8")))
            if src_name == "manifest.json":
                source = raw.get("source", {})
                source = {k:v for k,v in source.items() if k not in ("path", "file", "absolutePath")}
                compact = {k:raw.get(k) for k in ("schemaVersion", "generatedAt", "summary", "timing", "coverage", "identity", "duplicates", "artifacts", "existingAnalyses", "limitations") if k in raw}
                compact["source"] = source
                raw = compact
            else:
                # The bundle's manifest reference may contain the machine-specific input path.
                m = raw.get("manifest")
                if isinstance(m, dict):
                    m.pop("source", None)
                    m["fullAuditUrl"] = "fieldnotes-provenance.json"
            write_json(out / dest_name, raw)
        manifest["exported_records"] = exported
        manifest["read_snapshot"] = "SQLite read-only transaction; source rows and predictions read from one consistent database snapshot."
        write_json(out / "manifest.json", manifest)
    finally:
        conn.rollback()
        conn.close()
    return manifest


def write_readme(out: Path, manifest: dict):
    s = manifest["snapshot"]
    coverage = s["coverage_fraction_source_records"] * 100
    readme = f'''---
license: other
license_name: ai-village-research-terms
license_link: https://huggingface.co/datasets/aidigestorg/ai-village
language:
  - en
pretty_name: AI Village analyzed chat messages (partial exploratory snapshot)
tags:
  - agents
  - llm-agents
  - conversational-analysis
size_categories:
  - 100K<n<1M
---

# AI Village analyzed chat messages

This dataset joins source chat records from the AI Village transcript to an experimental, author-defined message-act classifier. It preserves one record per indexed source message, its exact source locator and source text, plus the current model annotation when available. Unclassified records are included with `prediction: null`.

## Snapshot status

Generated {manifest['generated_at_utc']}. The classifier snapshot is **{s['state']}**, with {s['classified_source_records']:,} of {manifest['source']['records']:,} source records annotated ({coverage:.2f}%) and {s['pending_source_records']:,} pending. The model run has not been validated against human labels. All annotations are exploratory; none are human-reviewed. This is not a corpus-wide finding or a validated benchmark.

The predictions select one primary candidate act per exact text. The categories and hypotheses are author-defined. Model scores are uncalibrated; independent entailment scores do not sum to one. Long inputs can be truncated at the model limit. Messages may contain multiple communicative acts; this pass does not perform sentence-level multi-act segmentation or extract task fields/arguments. Candidate templates are reusable DSPy-style signature sketches, not extracted values. Duplicate exact texts share a text-level prediction while each source occurrence remains a separate row.

## Files

- `data/messages-*.jsonl.gz`: source records and nullable predictions, split into gzip-compressed JSON Lines shards.
- `manifest.json`: source snapshot hash, model/run provenance, state, row accounting, coverage and shard checksums.
- `analysis-summary.json`: compact summaries of separately authored corpus analyses, when available.
- `fieldnotes-provenance.json` and `fieldnotes-evidence-bundle.json`: compact provenance audit artifacts, with machine-specific filesystem paths removed.

Each JSONL row has `message_id`, `source_locator` (`day`, `event_index`, `date`, original timestamp string), `source_snapshot_sha256`, `speaker_label`, `source_type`, `text_hash_sha256`, `text`, and `prediction`. The source timestamp string is preserved as supplied; it is not normalized here. `prediction` contains the label, score/review/truncation metadata, available model scores and candidate signatures.

## Provenance and upstream terms

The source archive is the [AI Village dataset by AI Digest](https://huggingface.co/datasets/aidigestorg/ai-village); its snapshot SHA-256 and source record count are in `manifest.json`. This analyzed derivative does not redistribute the full source archive. The upstream dataset card specifies research use, no AI training/fine-tuning without written permission, no re-identification, citation of AI Digest / AI Village, and notifying them about resulting publications. Review the upstream terms before using this derivative.

The labeling run uses pretrained inference, with model/runtime and execution-epoch provenance recorded in the manifest. The derivative does not include model weights, the source SQLite database, local environment files, or private credentials. See `manifest.json` for exact model, taxonomy and code hashes and for the run status at export time.
'''
    (out / "README.md").write_text(readme, encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=DB_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()
    manifest = export(args.db, args.out)
    write_readme(args.out, manifest)
    print(json.dumps({"out": str(args.out), "records": manifest["exported_records"],
                      "source_sha256": manifest["source"]["sha256"],
                      "state": manifest["snapshot"]["state"],
                      "classified": manifest["snapshot"]["classified_source_records"],
                      "unique_classified": manifest["snapshot"]["classified_unique_texts"]}, indent=2))


if __name__ == "__main__":
    main()
