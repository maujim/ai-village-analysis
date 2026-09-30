"""Build a private, reproducible context-sensitivity experiment input.

This is a model-free data preparation tool. It does not call a model or start a
cloud job. The target message is preserved exactly in all three conditions;
only-first truncation is prevented by measuring the actual tokenizer encoding
against the longest frozen taxonomy hypothesis.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from message_signatures import store
from message_signatures.cloud_transfer import _canonical_bytes, _file_sha256, _load_taxonomy
from message_signatures.nli_runtime import _load_hypotheses

DEFAULT_DB = store.DB
DEFAULT_TAXONOMY = Path(__file__).with_name("taxonomy.json")
DEFAULT_TOKENIZER = Path(__file__).resolve().parents[1] / "models" / "deberta-base-zeroshot"
DEFAULT_OUTPUT = Path("/private/tmp/ai-village-context-experiment")
MAX_LENGTH = 512
MAX_PREMISE_TOKENS = 384
MAX_TARGET_TOKENS = 160
MAX_CONTEXT_TOKENS = 200
CONDITIONS = ("isolated", "real_context", "shuffled_context")


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    backend = getattr(tokenizer, "_tokenizer", tokenizer)
    encoded = backend.encode(text, add_special_tokens=False)
    return list(encoded.ids)


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    backend = getattr(tokenizer, "_tokenizer", tokenizer)
    return backend.decode(token_ids, skip_special_tokens=True)


def _special_count(tokenizer: Any) -> int:
    return int(tokenizer.num_special_tokens_to_add(pair=True))


def _render_context(tokenizer: Any, contexts: list[dict[str, Any]], target: dict[str, Any],
                    max_context_tokens: int, premise_budget: int) -> tuple[str, int, list[dict[str, Any]]]:
    """Render context and target, trimming context IDs from the left only."""
    # Cap the context body independently, then enforce the tighter exact pair
    # budget after separators and target are included.
    body = "\n".join(f"{c['speaker']}: {c['text']}" for c in contexts)
    body_ids = _encode(tokenizer, body)
    if len(body_ids) > max_context_tokens:
        body_ids = body_ids[-max_context_tokens:]
    target_line = f"{target['speaker']}: {target['text']}"
    target_ids = _encode(tokenizer, target_line)
    kept_context_ids = body_ids
    while True:
        rendered_body = _decode(tokenizer, kept_context_ids).strip()
        premise = f"Previous messages:\n{rendered_body}\nCurrent message:\n{target_line}"
        premise_ids = _encode(tokenizer, premise)
        if len(premise_ids) <= premise_budget and len(premise_ids) <= MAX_PREMISE_TOKENS:
            return premise, len(premise_ids), [dict(c) for c in contexts]
        if not kept_context_ids:
            raise ValueError("target and fixed prompt do not fit the untruncated premise budget")
        # Trimming always consumes context tokens; the target remains an exact
        # suffix and is validated by the caller before serialization.
        kept_context_ids = kept_context_ids[1:]


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    # Stream the chronological index once; avoid sorting large message text
    # through nested SQLite window functions.
    records = []
    seen = set()
    previous_day_count = 0
    day_count = 0
    last_day = None
    cursor = conn.execute("SELECT id,day,event_index,date,timestamp,speaker,source_type,text,text_hash FROM messages ORDER BY day,event_index")
    for row in cursor:
        record = dict(row)
        if record["day"] != last_day:
            previous_day_count += day_count
            day_count = 0
            last_day = record["day"]
        if (day_count >= 2 and previous_day_count >= 2 and
                record["text_hash"] not in seen and
                isinstance(record["text"], str) and record["text"].strip()):
            record["prior_count"] = day_count
            record["shuffle_pool_count"] = previous_day_count
            records.append(record)
            seen.add(record["text_hash"])
        day_count += 1
    return records


def _chronological_previous(conn: sqlite3.Connection, target: dict[str, Any]) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT id,day,event_index,date,timestamp,speaker,source_type,text,text_hash
        FROM messages WHERE day=? AND event_index<? ORDER BY event_index DESC LIMIT 2
    """, (target["day"], target["event_index"])).fetchall()
    return [dict(r) for r in reversed(rows)]


def _shuffled_candidates(all_messages: list[dict[str, Any]], target: dict[str, Any], rng: random.Random,
                         reference_tokens: int, tokenizer: Any,
                         token_cache: dict[str, int]) -> list[dict[str, Any]]:
    # Sample from earlier, different-day chat events. Select a close token
    # length pair, then shuffle their order as the irrelevant-context control.
    candidates = [r for r in all_messages
                  if (r["day"] < target["day"] or
                      (r["day"] == target["day"] and r["event_index"] < target["event_index"]))
                  and r["day"] != target["day"] and isinstance(r["text"], str) and r["text"].strip()]
    if len(candidates) < 2:
        raise ValueError("not enough earlier cross-day messages for shuffled context")
    # Bounded random subset keeps preparation work modest and deterministic.
    if len(candidates) > 128:
        candidates = rng.sample(candidates, 128)
    for candidate in candidates:
        if candidate["id"] not in token_cache:
            token_cache[candidate["id"]] = len(_encode(tokenizer, f"{candidate['speaker']}: {candidate['text']}"))
        candidate["_context_tokens"] = token_cache[candidate["id"]]
    candidates.sort(key=lambda r: (abs(r["_context_tokens"] - reference_tokens / 2),
                                   r["day"], r["event_index"]))
    chosen = candidates[: min(32, len(candidates))]
    pairs = [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    pair = min(pairs, key=lambda p: (abs(p[0]["_context_tokens"] + p[1]["_context_tokens"] - reference_tokens),
                                    rng.random()))
    pair = [dict(pair[0]), dict(pair[1])]
    rng.shuffle(pair)
    return pair


def build_experiment(db_path: str | Path = DEFAULT_DB,
                     output_dir: str | Path = DEFAULT_OUTPUT,
                     tokenizer: Any | None = None,
                     tokenizer_path: str | Path = DEFAULT_TOKENIZER,
                     taxonomy_path: str | Path = DEFAULT_TAXONOMY,
                     sample_size: int = 128,
                     seed: int = 20260930,
                     framed_isolated: bool = False) -> dict[str, Any]:
    """Create the three-condition JSONL, manifest, and fixed sample audit."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    taxonomy, taxonomy_sha = _load_taxonomy(taxonomy_path)
    hypotheses = _load_hypotheses(taxonomy_path)
    max_hyp_tokens = max(len(_encode(tokenizer, h)) for h in hypotheses.values())
    premise_budget = MAX_LENGTH - max_hyp_tokens - _special_count(tokenizer)
    if premise_budget < 1:
        raise ValueError("taxonomy leaves no source-message token budget")

    conn = _open_readonly(Path(db_path))
    try:
        conn.execute("BEGIN")
        active = json.loads(conn.execute("SELECT value FROM metadata WHERE key='active_run'").fetchone()[0])
        source_sha = json.loads(conn.execute("SELECT value FROM metadata WHERE key='source_sha256'").fetchone()[0])
        candidates = _records(conn)
        all_messages = [dict(r) for r in conn.execute(
            "SELECT id,day,event_index,date,timestamp,speaker,source_type,text,text_hash FROM messages"
        )]
        rng = random.Random(seed)
        shuffled_token_cache: dict[str, int] = {}
        # Shuffle candidates before tokenizer filtering. Stopping after N
        # eligible rows gives a uniform sample without tokenizing the entire
        # archive on every experiment build.
        rng.shuffle(candidates)
        targets: list[dict[str, Any]] = []
        for record in candidates:
            # Keep the indexed source string byte-for-byte; strip is used only
            # to exclude whitespace-only messages, never to rewrite a target.
            record["target_tokens"] = len(_encode(tokenizer, f"{record['speaker']}: {record['text']}"))
            if record["target_tokens"] > MAX_TARGET_TOKENS:
                continue
            targets.append(record)
            if len(targets) == sample_size:
                break
        if len(targets) < sample_size:
            raise ValueError(f"only {len(targets)} eligible target messages; need {sample_size}")
        targets.sort(key=lambda r: (r["day"], r["event_index"]))

        rows: list[dict[str, Any]] = []
        sample_audit: list[dict[str, Any]] = []
        for index, target in enumerate(targets):
            real_context = _chronological_previous(conn, target)
            if len(real_context) != 2:
                raise ValueError("eligible target lost its two preceding same-day messages")
            real_len = sum(len(_encode(tokenizer, f"{c['speaker']}: {c['text']}")) for c in real_context)
            shuffled = _shuffled_candidates(all_messages, target, rng, real_len, tokenizer,
                                            shuffled_token_cache)
            if len(shuffled) != 2:
                raise ValueError("could not form two-message shuffled context")
            sample_audit.append({
                "sample_index": index, "target_id": target["id"], "target_text_hash": target["text_hash"],
                "day": target["day"], "event_index": target["event_index"],
                "target_tokens": target["target_tokens"],
                "real_context_ids": [c["id"] for c in real_context],
                "shuffled_context_ids": [c["id"] for c in shuffled],
            })
            contexts_by_condition = {
                "isolated": [], "real_context": real_context, "shuffled_context": shuffled,
            }
            for condition in CONDITIONS:
                if condition == "isolated":
                    premise = (f"Previous messages:\n\nCurrent message:\n{target['speaker']}: {target['text']}"
                               if framed_isolated else target["text"])
                    premise_tokens = len(_encode(tokenizer, premise))
                    kept_context: list[dict[str, Any]] = []
                else:
                    premise, premise_tokens, kept_context = _render_context(
                        tokenizer, contexts_by_condition[condition], target,
                        MAX_CONTEXT_TOKENS, premise_budget,
                    )
                target_line = f"{target['speaker']}: {target['text']}"
                if condition == "isolated":
                    target_present = premise.endswith(target_line) if framed_isolated else premise == target["text"]
                else:
                    target_present = premise.endswith(target_line)
                if not target_present:
                    raise ValueError(f"target was altered or omitted in {condition}")
                if premise_tokens > premise_budget or premise_tokens > MAX_PREMISE_TOKENS:
                    raise ValueError("premise exceeds fixed all-hypothesis token budget")
                text_hash = _sha_text(premise)
                rows.append({
                    "id": f"ctx-{index:04d}-{condition}", "text_hash": text_hash,
                    "base_text_hash": target["text_hash"], "condition": condition,
                    "target_id": target["id"], "target_text_hash": target["text_hash"],
                    "day": target["day"], "event_index": target["event_index"],
                    "date": target.get("date"), "timestamp": target.get("timestamp"),
                    "speaker": target.get("speaker"), "target": target["text"],
                    "text": premise, "target_tokens": target["target_tokens"],
                    "premise_tokens": premise_tokens,
                    "context_tokens_before_trim": real_len if condition == "real_context" else
                        (sum(len(_encode(tokenizer, f"{c['speaker']}: {c['text']}")) for c in shuffled)
                         if condition == "shuffled_context" else 0),
                    "context_tokens_cap": MAX_CONTEXT_TOKENS,
                    "context": [{k: c.get(k) for k in ("id", "day", "event_index", "timestamp", "speaker", "text", "text_hash")}
                                for c in kept_context],
                })

        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=False)
        payload_path = output / "context-input.jsonl.gz"
        with payload_path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz:
            for row in rows:
                gz.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
        audit_path = output / "sample-audit.json"
        audit_path.write_text(json.dumps(sample_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model_assets = active.get("model_assets_sha256", {})
        payload = {
            "schemaVersion": 1, "kind": "context-sensitivity-experiment",
            "experimentId": hashlib.sha256(_canonical_bytes({
                "sourceSha256": source_sha, "taxonomySha256": taxonomy_sha,
                "seed": seed, "sampleSize": sample_size, "sample": sample_audit,
                "framedIsolated": framed_isolated,
            })).hexdigest()[:24],
            "runId": active.get("id"), "model": active.get("model"),
            "modelSha256": active.get("model_sha256"), "modelAssetsSha256": model_assets,
            "sourceSha256": source_sha, "taxonomySha256": taxonomy_sha,
            "scoreSemantics": active.get("score_semantics", "independent_entailment"),
            "question": active.get("question"), "maxLength": MAX_LENGTH,
            "maxPremiseTokens": MAX_PREMISE_TOKENS, "maxInputTokens": MAX_PREMISE_TOKENS,
            "maxTargetTokens": MAX_TARGET_TOKENS,
            "maxContextTokens": MAX_CONTEXT_TOKENS,
            "tokenizer": "DeBERTa tokenizer pinned by modelAssetsSha256",
            "tokenizerAssetsSha256": {k: v for k, v in model_assets.items()
                                      if k.startswith(("tokenizer", "special_tokens", "added_tokens", "spm"))},
            "tokenizerPath": str(Path(tokenizer_path).resolve()),
            "conditions": list(CONDITIONS), "hypothesisCount": len(hypotheses),
            "isolatedFraming": "matched_context_wrapper" if framed_isolated else "raw_target",
            "seed": seed, "targetCount": sample_size, "rowCount": len(rows),
            "inputFile": payload_path.name, "inputSha256": _file_sha256(payload_path),
            "count": len(rows),
            "rowsPerCondition": {c: sample_size for c in CONDITIONS},
            "payload": {"file": payload_path.name, "count": len(rows), "sha256": _file_sha256(payload_path)},
            "sampleAudit": {"file": audit_path.name, "sha256": _file_sha256(audit_path)},
            "manifestSha256": "",
        }
        payload["manifestSha256"] = hashlib.sha256(_canonical_bytes({k: v for k, v in payload.items() if k != "manifestSha256"})).hexdigest()
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--framed-isolated", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_experiment(args.db, args.output, tokenizer_path=args.tokenizer,
                                      taxonomy_path=args.taxonomy, sample_size=args.sample_size,
                                      seed=args.seed, framed_isolated=args.framed_isolated), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
