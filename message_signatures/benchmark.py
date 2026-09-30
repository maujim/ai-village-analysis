"""Reproducible local NLI parity/speed benchmark.

The fixed source sample is chosen by text SHA-256 order, independent of the
classifier's pending/completed queue. The legacy baseline is loaded directly
from git commit 3703d866ea into a temporary module and receives explicit model
and taxonomy paths. This command is opt-in; importing this module runs no model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "models" / "deberta-base-zeroshot"
TAXONOMY_PATH = HERE / "taxonomy.json"
DB_PATH = HERE / "messages.sqlite"
BASELINE_REVISION = "3703d866ea"
QUESTION = "What is the primary communicative function of this message? Choose the best matching act."
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "ai-village-nli-benchmark-baseline.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hash() -> str:
    import viewer

    return sha256_file(Path(viewer.TRANSCRIPT))


def fixed_hash_sample(sample_size: int, db_path: Path = DB_PATH, expected_source_hash: str | None = None) -> list[dict[str, str]]:
    """Select unique source messages in deterministic text-hash order."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not db_path.is_file():
        raise FileNotFoundError(f"Message index not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key='source_sha256'").fetchone()
        indexed_source = json.loads(row[0]) if row else None
        if expected_source_hash and indexed_source != expected_source_hash:
            raise RuntimeError("Message index source hash does not match the transcript snapshot")
        rows = connection.execute(
            "SELECT text_hash, MIN(text) AS text FROM messages GROUP BY text_hash ORDER BY text_hash LIMIT ?",
            (sample_size,),
        ).fetchall()
    finally:
        connection.close()
    if len(rows) < sample_size:
        raise ValueError(f"Requested {sample_size} unique source texts; only {len(rows)} are indexed")
    return [{"text_hash": str(key), "text": str(text)} for key, text in rows]


def edge_fixtures() -> list[dict[str, str]]:
    """Small fixed cases covering empty, unicode, and long-input boundaries."""
    return [
        {"id": "fixture-empty", "text": ""},
        {"id": "fixture-unicode-punctuation", "text": "‘We checked it’—or did we? ✅ café…"},
        {"id": "fixture-near-limit", "text": "status " * 480},
        {"id": "fixture-truncated-long", "text": ("A recorded status update includes a source and a scoped result. " * 160).strip()},
    ]


def _scores(result: Mapping[str, Any]) -> Mapping[str, float]:
    values = result.get("probabilities") or result.get("scores")
    if not isinstance(values, Mapping) or not values:
        raise ValueError("runtime returned no act probabilities")
    return values


def _top_and_review(result: Mapping[str, Any], text: str) -> tuple[str, bool]:
    scores = _scores(result)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[0]
    margin = float(scores[top]) - float(scores[ranked[1]]) if len(ranked) > 1 else 0.0
    review = float(scores[top]) < 0.55 or margin < 0.15 or bool(result.get("truncated")) or not text.strip()
    return top, review


def _truncation_signature(result: Mapping[str, Any]) -> tuple[Any, ...]:
    truncation = result.get("truncation") or {}
    per_hypothesis = truncation.get("per_hypothesis") or {}
    return (
        bool(result.get("truncated") or truncation.get("message")),
        tuple(sorted(
            (str(act), bool(info.get("message")), int(info.get("message_tokens_dropped", -1)))
            for act, info in per_hypothesis.items()
        )),
        int(truncation.get("message_tokens_dropped_max", -1)),
    )


def compare_rows(texts: Sequence[str], baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(texts) != len(baseline) or len(texts) != len(candidate):
        raise ValueError("Comparison row count does not match messages")
    if not texts:
        raise ValueError("Cannot compare an empty benchmark set")
    deltas: list[float] = []
    top_matches = trunc_matches = review_matches = 0
    per_message = []
    for text, old, new in zip(texts, baseline, candidate):
        old_scores, new_scores = _scores(old), _scores(new)
        if set(old_scores) != set(new_scores):
            raise ValueError("Baseline and candidate scored different act IDs")
        row_deltas = [abs(float(old_scores[key]) - float(new_scores[key])) for key in old_scores]
        deltas.extend(row_deltas)
        old_top, old_review = _top_and_review(old, text)
        new_top, new_review = _top_and_review(new, text)
        top_equal = old_top == new_top
        trunc_equal = _truncation_signature(old) == _truncation_signature(new)
        review_equal = old_review == new_review
        top_matches += int(top_equal)
        trunc_matches += int(trunc_equal)
        review_matches += int(review_equal)
        per_message.append({
            "topLabelAgreement": top_equal,
            "truncationEqual": trunc_equal,
            "reviewFlagAgreement": review_equal,
            "maxAbsoluteScoreDelta": max(row_deltas, default=0.0),
        })
    return {
        "messagesCompared": len(texts),
        "actScoresCompared": len(deltas),
        "topLabelAgreement": top_matches,
        "topLabelAgreementFraction": top_matches / len(texts),
        "maximumAbsoluteScoreDelta": max(deltas, default=0.0),
        "meanAbsoluteScoreDelta": sum(deltas) / len(deltas),
        "truncationEquality": trunc_matches,
        "truncationEqualityFraction": trunc_matches / len(texts),
        "reviewFlagAgreement": review_matches,
        "reviewFlagAgreementFraction": review_matches / len(texts),
        "perMessage": per_message,
    }


def _options_and_version() -> tuple[dict[str, str], str | None]:
    data = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    return {str(act["id"]): str(act["definition"]) for act in data["acts"]}, data.get("version")


def _load_baseline_module(temporary_dir: Path):
    source = subprocess.run(
        ["git", "show", f"{BASELINE_REVISION}:message_signatures/nli_runtime.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    baseline_path = temporary_dir / "nli_runtime_baseline.py"
    baseline_path.write_bytes(source)
    name = "_ai_village_nli_baseline"
    spec = importlib.util.spec_from_file_location(name, baseline_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load temporary baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, source


def _cache_key(source_sha: str, model_sha: str, taxonomy_sha: str, baseline_code_sha: str, device: str, pair_batch_size: int) -> str:
    material = {
        "source": source_sha,
        "model": model_sha,
        "taxonomy": taxonomy_sha,
        "baselineCode": baseline_code_sha,
        "device": device,
        "pairBatchSize": pair_batch_size,
        "maxLength": 512,
        "question": QUESTION,
    }
    return sha256_bytes(json.dumps(material, sort_keys=True).encode())


def _read_cache(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schemaVersion": 1, "runs": {}}


def _write_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_runtime(runtime: Any, texts: Sequence[str], options: Mapping[str, str]) -> tuple[list[dict[str, Any]], float]:
    start = time.perf_counter()
    outputs = runtime.predict(texts, options, QUESTION)
    elapsed = time.perf_counter() - start
    if len(outputs) != len(texts):
        raise ValueError("Runtime output count does not match message count")
    return outputs, elapsed


def run_benchmark(
    *,
    sample_size: int = 48,
    backend: str = "torch",
    device: str = "mps",
    baseline_pair_batch_size: int = 32,
    pair_batch_size: int = 32,
    output_path: Path | None = None,
    cache_path: Path = DEFAULT_CACHE,
    compare_fixtures: bool = True,
) -> dict[str, Any]:
    import viewer
    try:
        from .nli_runtime import NLIRuntime
    except ImportError:
        from nli_runtime import NLIRuntime

    current_source_hash = source_hash()
    sample = fixed_hash_sample(sample_size, expected_source_hash=current_source_hash)
    fixtures = edge_fixtures() if compare_fixtures else []
    all_rows = [{"id": f"source-{row['text_hash'][:16]}", **row} for row in sample]
    all_rows += fixtures
    texts = [row["text"] for row in all_rows]
    sample_sha = sha256_bytes("\n".join(row["text_hash"] for row in sample).encode())
    model_sha = sha256_file(MODEL_DIR / "model.safetensors")
    taxonomy_sha = sha256_file(TAXONOMY_PATH)
    optimized_code_sha = sha256_file(HERE / "nli_runtime.py")
    options, taxonomy_version = _options_and_version()
    device_key = device

    with tempfile.TemporaryDirectory(prefix="ai-village-nli-benchmark-") as temp:
        temp_dir = Path(temp)
        baseline_module, baseline_source = _load_baseline_module(temp_dir)
        baseline_code_sha = sha256_bytes(baseline_source)
        key = _cache_key(current_source_hash, model_sha, taxonomy_sha, baseline_code_sha, device_key, baseline_pair_batch_size)
        cache = _read_cache(cache_path)
        run_cache = cache.setdefault("runs", {}).setdefault(key, {"rows": {}})
        row_cache = run_cache.setdefault("rows", {})
        # Fixture keys are also keyed by their exact text hash.
        text_keys = [sha256_bytes(text.encode()) for text in texts]
        missing_indices = [i for i, msg_key in enumerate(text_keys) if msg_key not in row_cache]
        baseline_load_seconds = 0.0
        baseline_source_inference_seconds = 0.0
        baseline_fixture_inference_seconds = 0.0
        baseline_cache_hits = len(texts) - len(missing_indices)
        if missing_indices:
            # Match the production runner's Torch CPU thread setting.
            import torch
            torch.set_num_threads(4)
            load_start = time.perf_counter()
            baseline_runtime = baseline_module.NLIRuntime(
                model_dir=MODEL_DIR,
                taxonomy_path=TAXONOMY_PATH,
                device=device,
                batch_size=baseline_pair_batch_size,
                max_length=512,
            )
            baseline_load_seconds = time.perf_counter() - load_start
            missing_source = [i for i in missing_indices if i < len(sample)]
            missing_fixtures = [i for i in missing_indices if i >= len(sample)]
            for indices, timing_key in (
                (missing_source, "source"),
                (missing_fixtures, "fixtures"),
            ):
                if not indices:
                    continue
                outputs, elapsed = _run_runtime(baseline_runtime, [texts[i] for i in indices], options)
                if timing_key == "source":
                    baseline_source_inference_seconds = elapsed
                else:
                    baseline_fixture_inference_seconds = elapsed
                for i, result in zip(indices, outputs):
                    row_cache[text_keys[i]] = result
            _write_cache(cache_path, cache)
            del baseline_runtime
        baseline_results = [row_cache[msg_key] for msg_key in text_keys]

    candidate_kwargs: dict[str, Any] = {
        "model_dir": MODEL_DIR,
        "taxonomy_path": TAXONOMY_PATH,
        "device": device,
        "max_length": 512,
        "pair_batch_size": pair_batch_size,
    }
    if backend != "torch":
        candidate_kwargs["backend"] = backend
    # Keep Torch runtime comparisons on the same thread configuration as the
    # corpus runner. MLX does not use Torch for model inference.
    import torch
    torch.set_num_threads(4)
    candidate_load_start = time.perf_counter()
    candidate_runtime = NLIRuntime(**candidate_kwargs)
    candidate_load_seconds = time.perf_counter() - candidate_load_start
    candidate_source_results, candidate_source_inference_seconds = _run_runtime(
        candidate_runtime, texts[:len(sample)], options
    )
    candidate_fixture_results: list[dict[str, Any]] = []
    candidate_fixture_inference_seconds = 0.0
    if fixtures:
        candidate_fixture_results, candidate_fixture_inference_seconds = _run_runtime(
            candidate_runtime, texts[len(sample):], options
        )
    candidate_results = candidate_source_results + candidate_fixture_results
    candidate_metadata = candidate_runtime.metadata()
    candidate_metadata.setdefault("backend", backend)

    comparison = compare_rows(texts, baseline_results, candidate_results)
    source_comparison = compare_rows(
        texts[:len(sample)], baseline_results[:len(sample)], candidate_results[:len(sample)]
    )
    fixture_comparison = None
    if fixtures:
        fixture_comparison = {
            "fixtureIds": [row["id"] for row in fixtures],
            **compare_rows(
            texts[len(sample):], baseline_results[len(sample):], candidate_results[len(sample):]
            ),
        }
    result = {
        "schemaVersion": 1,
        "model": {
            "path": str(MODEL_DIR),
            "sha256": model_sha,
            "taxonomySha256": taxonomy_sha,
            "taxonomyVersion": taxonomy_version,
        },
        "source": {
            "path": str(Path(viewer.TRANSCRIPT)),
            "sha256": current_source_hash,
            "sampleMethod": "unique messages ordered by ascending SHA-256 text_hash in indexed corpus",
            "sampleCount": len(sample),
            "sampleHash": sample_sha,
            "sampleTextHashes": [row["text_hash"] for row in sample],
        },
        "code": {
            "baselineRevision": BASELINE_REVISION,
            "baselineRuntimeSha256": baseline_code_sha,
            "candidateRuntimeSha256": optimized_code_sha,
            "candidateMLXBackendSha256": sha256_file(HERE / "mlx_deberta.py") if backend == "mlx" else None,
            "benchmarkHarnessSha256": sha256_file(HERE / "benchmark.py"),
        },
        "settings": {
            "baselineBackend": "torch",
            "backend": backend,
            "device": device,
            "baselinePairBatchSize": baseline_pair_batch_size,
            "pairBatchSize": pair_batch_size,
            "torchThreads": 4,
            "maxLength": 512,
            "scoreSemantics": "independent_entailment",
            "reviewThresholds": {"topBelow": 0.55, "marginBelow": 0.15},
        },
        "timing": {
            "baselineLoadSeconds": baseline_load_seconds,
            "baselineSourceInferenceSecondsThisInvocation": baseline_source_inference_seconds,
            "baselineFixtureInferenceSecondsThisInvocation": baseline_fixture_inference_seconds,
            "baselineCacheHits": baseline_cache_hits,
            "baselineCachedRows": len(row_cache),
            "baselineCachePath": str(cache_path),
            "candidateLoadSeconds": candidate_load_seconds,
            "candidateSourceInferenceSeconds": candidate_source_inference_seconds,
            "candidateFixtureInferenceSeconds": candidate_fixture_inference_seconds,
            "sourceMessagesPerSecond": len(sample) / candidate_source_inference_seconds if candidate_source_inference_seconds else None,
        },
        "sourceSampleComparison": source_comparison,
        "edgeFixtureComparison": fixture_comparison,
        "combinedComparison": comparison,
        "candidateRuntime": candidate_metadata,
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--baseline-pair-batch-size", type=int, default=32)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-fixtures", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        sample_size=args.sample_size,
        backend=args.backend,
        device=args.device,
        baseline_pair_batch_size=args.baseline_pair_batch_size,
        pair_batch_size=args.pair_batch_size,
        output_path=args.output,
        cache_path=args.baseline_cache,
        compare_fixtures=not args.no_fixtures,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
