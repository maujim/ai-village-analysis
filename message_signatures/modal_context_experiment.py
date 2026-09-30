"""One-off, bounded Modal worker for the context sensitivity experiment.

This module is isolated from ``modal_worker.py`` so the frozen classification
worker and its parity provenance remain unchanged. The input bundle is prepared
locally, uploaded to the private Volume, and addressed by a manifest-pinned
path/hash/count. This worker does not access the SQLite store.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

try:
    import modal
except ModuleNotFoundError:  # Model-free helpers and tests need no Modal SDK.
    modal = None  # type: ignore[assignment]

from message_signatures.modal_worker import (
    CudaNliRuntime, MODEL_DIR, MODEL_NAME, MODEL_VOLUME_NAME, TAXONOMY_CONTAINER_PATH,
    VOLUME_MOUNT, sha256_file,
)

HERE = Path(__file__).resolve().parent
CONDITIONS = ("isolated", "real_context", "shuffled_context")
MAX_INPUT_ROWS = 384
MAX_INPUT_TOKENS = 384
MAX_LENGTH = 512
PAIR_BATCH_SIZE = 96
MESSAGE_BATCH_SIZE = 24
MAX_WALL_SECONDS = 180
GPU_TYPE = "L4"
QUESTION = "What is the primary communicative function of this message? Choose the best matching act."
DEFAULT_INPUT_ROOT = "jobpayloads"


def canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_input_manifest(manifest: Mapping[str, Any]) -> None:
    required = ("experimentId", "inputFile", "inputSha256", "count", "sourceSha256",
                "taxonomySha256", "model", "modelSha256", "modelAssetsSha256", "maxInputTokens",
                "maxLength", "manifestSha256", "question")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("Context input manifest missing: " + ", ".join(missing))
    if int(manifest["count"]) != MAX_INPUT_ROWS or int(manifest["maxInputTokens"]) != MAX_INPUT_TOKENS:
        raise ValueError("Context manifest row/token bounds differ from frozen experiment")
    if int(manifest["maxLength"]) != MAX_LENGTH:
        raise ValueError(f"Context inference maxLength must be {MAX_LENGTH}")
    if manifest.get("targetCount") != 128 or manifest.get("rowsPerCondition") != {c: 128 for c in CONDITIONS}:
        raise ValueError("Context manifest must describe 128 targets per condition")
    if str(manifest["model"]) != MODEL_NAME:
        raise ValueError("Context manifest model does not match frozen NLI model")
    if canonical_sha256({key: value for key, value in manifest.items() if key != "manifestSha256"}) != manifest["manifestSha256"]:
        raise ValueError("Context input manifest SHA-256 mismatch")
    for key in ("inputSha256", "sourceSha256", "taxonomySha256", "modelSha256"):
        value = str(manifest[key])
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if PurePosixPath(str(manifest["inputFile"])).name != str(manifest["inputFile"]):
        raise ValueError("inputFile must be a simple filename")


def safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"Expected a safe relative Volume path, got {value!r}")
    return str(path)


def validate_spec(spec: Mapping[str, Any]) -> None:
    required = ("experimentId", "inputPath", "inputFile", "inputSha256", "count", "sourceSha256",
                "model", "modelSha256", "modelAssetsSha256", "taxonomySha256",
                "inputManifestSha256", "gpuHourlyRateUsd", "maxInputTokens", "maxLength")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError("Context experiment spec missing: " + ", ".join(missing))
    if spec["model"] != MODEL_NAME:
        raise ValueError("Unexpected model identity")
    if int(spec["count"]) != MAX_INPUT_ROWS:
        raise ValueError(f"Context experiment must contain exactly {MAX_INPUT_ROWS} rows")
    if int(spec["maxInputTokens"]) != MAX_INPUT_TOKENS or int(spec["maxLength"]) != MAX_LENGTH:
        raise ValueError("Context input/inference token limits differ from the frozen experiment")
    if not isinstance(spec["modelAssetsSha256"], dict):
        raise ValueError("modelAssetsSha256 must be an object")
    for key in ("inputSha256", "sourceSha256", "modelSha256", "taxonomySha256", "inputManifestSha256"):
        value = str(spec[key])
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if not math.isfinite(float(spec["gpuHourlyRateUsd"])) or float(spec["gpuHourlyRateUsd"]) < 0:
        raise ValueError("gpuHourlyRateUsd must be a finite nonnegative number")
    safe_relative_path(str(spec["inputPath"]))
    expected_path = str(PurePosixPath(DEFAULT_INPUT_ROOT) / str(spec["experimentId"]) / str(spec["inputFile"]))
    if str(spec["inputPath"]) != expected_path:
        raise ValueError(f"inputPath must be {expected_path!r}")


def read_experiment_rows(path: Path, expected_sha256: str, expected_count: int) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ValueError("Context input compressed SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    by_target: dict[str, set[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank context JSONL row at line {line_no}")
            row = json.loads(line)
            for key in ("id", "base_text_hash", "condition", "target", "text", "text_hash"):
                if key not in row:
                    raise ValueError(f"Context row {line_no} is missing {key}")
            row_id = str(row["id"])
            if not row_id or row_id in ids:
                raise ValueError(f"Context row id is empty or duplicated at line {line_no}")
            ids.add(row_id)
            condition = str(row["condition"])
            if condition not in CONDITIONS:
                raise ValueError(f"Unknown context condition {condition!r} at line {line_no}")
            if not isinstance(row["target"], str) or not row["target"] or not isinstance(row["text"], str):
                raise ValueError(f"Context row {line_no} needs nonempty target and string text")
            framed_target = f"Previous messages:\n\nCurrent message:\n{row.get('speaker')}: {row['target']}"
            if condition == "isolated" and row["text"] not in (row["target"], framed_target):
                raise ValueError(f"Isolated condition must send the target alone at line {line_no}")
            if condition != "isolated" and not row["text"].endswith(row["target"]):
                raise ValueError(f"Context premise must preserve the entire target at its end (line {line_no})")
            actual_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            if actual_hash != row["text_hash"]:
                raise ValueError(f"Context premise/text_hash mismatch at line {line_no}")
            base_hash = str(row["base_text_hash"])
            if len(base_hash) != 64 or any(c not in "0123456789abcdef" for c in base_hash):
                raise ValueError(f"Invalid base_text_hash at line {line_no}")
            by_target.setdefault(base_hash, set()).add(condition)
            rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"Context input has {len(rows)} rows; expected {expected_count}")
    if len(by_target) != expected_count // len(CONDITIONS) or any(conditions != set(CONDITIONS) for conditions in by_target.values()):
        raise ValueError("Expected exactly one row for each of three conditions per target")
    return rows


def validate_runtime_assets(spec: Mapping[str, Any], model_path: Path = Path(MODEL_DIR),
                            taxonomy_path: Path = TAXONOMY_CONTAINER_PATH) -> None:
    if sha256_file(model_path / "model.safetensors") != spec["modelSha256"]:
        raise ValueError("Mounted model weights differ from experiment manifest")
    assets = {
        path.name: sha256_file(path) for path in sorted(model_path.glob("*"))
        if path.is_file() and path.suffix in (".json", ".model", ".jinja")
    }
    if assets != spec["modelAssetsSha256"]:
        raise ValueError("Mounted model/tokenizer assets differ from experiment manifest")
    if sha256_file(taxonomy_path) != spec["taxonomySha256"]:
        raise ValueError("Mounted taxonomy differs from experiment manifest")


def score_experiment(spec: Mapping[str, Any], *, model_path: Path = Path(MODEL_DIR),
                     taxonomy_path: Path = TAXONOMY_CONTAINER_PATH,
                     runtime_class: Any = CudaNliRuntime,
                     volume_mount: str = VOLUME_MOUNT) -> dict[str, Any]:
    """Score all 384 condition rows, returning a complete or explicit partial payload."""
    validate_spec(spec)
    started = time.monotonic()
    read_seconds = model_seconds = infer_seconds = 0.0
    rows_out: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    try:
        input_path = Path(volume_mount) / safe_relative_path(str(spec["inputPath"]))
        phase = time.monotonic()
        rows = read_experiment_rows(input_path, str(spec["inputSha256"]), int(spec["count"]))
        read_seconds = time.monotonic() - phase
        validate_runtime_assets(spec, model_path, taxonomy_path)
        phase = time.monotonic()
        runtime = runtime_class(str(model_path), str(taxonomy_path), PAIR_BATCH_SIZE)
        model_seconds = time.monotonic() - phase
        for row in rows:
            n_tokens = len(runtime.tokenizer_backend.encode(str(row["text"]), add_special_tokens=False).ids)
            if n_tokens > MAX_INPUT_TOKENS:
                raise ValueError(f"Premise {row['id']} exceeds the {MAX_INPUT_TOKENS}-token context limit")
        runtime.execution_id = str(spec["experimentId"])
        runtime.source_sha256 = str(spec["sourceSha256"])
        runtime.taxonomy_sha256 = str(spec["taxonomySha256"])
        meta = runtime.metadata(__file__)
        texts = [str(row["text"]) for row in rows]
        # One condition per input row; preserve row identity and source locators.
        deadline = time.time() + MAX_WALL_SECONDS
        for start in range(0, len(rows), MESSAGE_BATCH_SIZE):
            batch_rows = rows[start:start + MESSAGE_BATCH_SIZE]
            phase = time.monotonic()
            predictions = runtime.predict([str(row["text"]) for row in batch_rows], deadline)
            infer_seconds += time.monotonic() - phase
            for row, pred in zip(batch_rows, predictions):
                if set(pred["scores"]) != set(runtime.act_ids):
                    raise ValueError("Runtime did not score all taxonomy acts")
                if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in pred["scores"].values()):
                    raise ValueError("Runtime returned invalid entailment score")
                rows_out.append({
                    "id": row["id"], "base_text_hash": row["base_text_hash"],
                    "condition": row["condition"], "text_hash": row["text_hash"],
                    "scores": pred["scores"], "truncation": pred["truncation"],
                    "target": row["target"], "target_tokens": row.get("target_tokens"),
                    "context_tokens": row.get("context_tokens"),
                    "source_locator": row.get("source_locator"),
                    "context_locators": row.get("context_locators", []),
                    "target_id": row.get("target_id"), "target_text_hash": row.get("target_text_hash"),
                    "day": row.get("day"), "event_index": row.get("event_index"),
                    "date": row.get("date"), "timestamp": row.get("timestamp"),
                    "speaker": row.get("speaker"), "context": row.get("context", []),
                })
        if len(rows_out) != len(rows):
            raise ValueError("Context output count does not match input count")
        complete = True
        error = None
    except Exception as exc:
        complete = False
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started
    inference_rows = len(rows_out)
    return {
        "schemaVersion": 1, "experimentId": str(spec["experimentId"]),
        "inputManifestSha256": spec["inputManifestSha256"], "inputSha256": spec["inputSha256"],
        "sourceSha256": spec["sourceSha256"], "model": MODEL_NAME,
        "modelSha256": spec["modelSha256"], "modelAssetsSha256": spec["modelAssetsSha256"],
        "taxonomySha256": spec["taxonomySha256"], "backend": "cuda", "gpuRequested": GPU_TYPE,
        "question": spec.get("question"), "maxInputTokens": MAX_INPUT_TOKENS,
        "maxLength": MAX_LENGTH, "complete": complete, "count": inference_rows,
        "rows": rows_out, "error": error,
        "performance": {
            "inputReadSeconds": round(read_seconds, 4), "modelLoadSeconds": round(model_seconds, 4),
            "inferenceSeconds": round(infer_seconds, 4), "elapsedSeconds": round(elapsed, 4),
            "messagesScored": inference_rows, "pairsScored": inference_rows * 12,
            "messagesPerInferenceSecond": round(inference_rows / infer_seconds, 4) if infer_seconds else None,
            "gpuHourlyRateUsd": float(spec["gpuHourlyRateUsd"]),
            "estimatedGpuCostUsd": round(elapsed / 3600 * float(spec["gpuHourlyRateUsd"]), 6),
            "costEstimateBasis": "worker scoring-body duration × supplied L4 hourly rate; excludes startup, platform overhead, and storage; not an invoice",
        },
        "runtime": meta,
    }


def write_output(path: str | Path, result: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    payload = dict(result)
    payload["resultSha256"] = canonical_sha256(payload)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    temp.write_text(serialized, encoding="utf-8")
    temp.replace(target)
    checksum_path = target.with_suffix(target.suffix + ".sha256")
    checksum_tmp = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
    checksum_tmp.write_text(hashlib.sha256(serialized.encode("utf-8")).hexdigest() + "\n", encoding="ascii")
    checksum_tmp.replace(checksum_path)


if modal is not None:
    app = modal.App("ai-village-context-experiment")
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install("torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2",
                        "safetensors==0.8.0", "numpy==2.5.3")
        .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("message_signatures")
        .add_local_file(str(HERE / "taxonomy.json"), str(TAXONOMY_CONTAINER_PATH))
    )

    @app.function(image=image, gpu=GPU_TYPE, cpu=4, memory=16_384, max_containers=1,
                  timeout=MAX_WALL_SECONDS, retries=0, scaledown_window=10,
                  volumes={VOLUME_MOUNT: model_volume.with_mount_options(read_only=True)})
    def score_context(spec: dict[str, Any]) -> dict[str, Any]:
        return score_experiment(spec)

    @app.local_entrypoint()
    def main(input_manifest: str, input_path: str, gpu_hourly_rate_usd: float, output: str) -> None:
        manifest = json.loads(Path(input_manifest).read_text(encoding="utf-8"))
        validate_input_manifest(manifest)
        expected_input_path = str(PurePosixPath(DEFAULT_INPUT_ROOT) / str(manifest["experimentId"]) / str(manifest["inputFile"]))
        if input_path != expected_input_path:
            raise ValueError(f"--input-path must be {expected_input_path!r}")
        spec = {
            "experimentId": manifest["experimentId"], "inputPath": input_path,
            "inputFile": manifest["inputFile"], "inputSha256": manifest["inputSha256"],
            "count": manifest["count"], "sourceSha256": manifest["sourceSha256"],
            "model": MODEL_NAME, "modelSha256": manifest["modelSha256"],
            "modelAssetsSha256": manifest["modelAssetsSha256"], "taxonomySha256": manifest["taxonomySha256"],
            "inputManifestSha256": manifest["manifestSha256"], "gpuHourlyRateUsd": gpu_hourly_rate_usd,
            "maxInputTokens": manifest["maxInputTokens"], "maxLength": manifest["maxLength"],
            "question": manifest["question"],
        }
        result = score_context.remote(spec)
        write_output(output, result)
        if not result["complete"]:
            raise RuntimeError(f"Context experiment incomplete: {result['error']}")
else:
    app = None
    score_context = None
