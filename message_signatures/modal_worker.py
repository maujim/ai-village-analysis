"""Bounded Modal CUDA inference for the frozen DeBERTa NLI annotations.

This module is a transfer/inference worker only. It never opens or writes the
local SQLite database. The default invocation scores the parity pilots; pending
corpus shards require an explicit mode. Model and input files must already be
present in the named private Modal Volume. Nothing is downloaded from Hugging
Face by a running worker.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import modal

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c"
MODEL_VOLUME_NAME = "ai-village-classification-20260930"
VOLUME_MOUNT = "/mnt/village"
MODEL_DIR = f"{VOLUME_MOUNT}/model"
PAYLOAD_ROOT = f"{VOLUME_MOUNT}/jobpayloads"
MAX_LENGTH = 512
QUESTION = "What is the primary communicative function of this message? Choose the best matching act."
DEFAULT_PAIR_BATCH_SIZE = 256
MESSAGE_BATCH_SIZE = 32
MAX_WALL_SECONDS = 600
MAX_CONTAINERS = 8
REVIEW_TOP_BELOW = 0.55
REVIEW_MARGIN_BELOW = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name in ("", ".", ".."):
        raise ValueError(f"Expected a simple shard filename, got {value!r}")
    return path.name


def validate_export_manifest(manifest: Mapping[str, Any]) -> None:
    """Check inference-critical fields without trusting filenames or row data."""
    required = ("schemaVersion", "exportId", "runId", "sourceSha256", "model",
                "modelSha256", "modelAssetsSha256", "taxonomySha256", "actSpecs",
                "scoreSemantics", "maxLength", "pendingShards", "pilotShards")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError("Export manifest is missing: " + ", ".join(missing))
    payload = {key: value for key, value in manifest.items() if key != "manifestSha256"}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("manifestSha256"):
        raise ValueError("Export manifest SHA-256 mismatch")
    if manifest["scoreSemantics"] != "independent_entailment":
        raise ValueError("This worker only supports independent NLI entailment scores")
    if int(manifest["maxLength"]) != MAX_LENGTH:
        raise ValueError(f"Expected maxLength={MAX_LENGTH}")
    if manifest["model"] != MODEL_NAME:
        raise ValueError(f"Unexpected model identity: {manifest['model']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["modelSha256"])):
        raise ValueError("modelSha256 must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["taxonomySha256"])):
        raise ValueError("taxonomySha256 must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest["sourceSha256"])):
        raise ValueError("sourceSha256 must be a lowercase SHA-256 digest")
    specs = manifest["actSpecs"]
    if not isinstance(specs, list) or len(specs) != 12:
        raise ValueError("Expected the frozen 12-act taxonomy")
    ids = [str(act.get("id", "")) for act in specs]
    if len(set(ids)) != 12 or any(not act.get("hypothesis") for act in specs):
        raise ValueError("Taxonomy IDs must be unique and each act needs a hypothesis")
    seen: set[str] = set()
    for group_name, expected_kind in (("pendingShards", "pending"), ("pilotShards", "parity_only")):
        if not isinstance(manifest[group_name], list):
            raise ValueError(f"{group_name} must be a list")
        for shard in manifest[group_name]:
            name = _safe_name(str(shard.get("file", "")))
            if name in seen:
                raise ValueError(f"Duplicate shard filename in manifest: {name}")
            seen.add(name)
            if shard.get("kind") != expected_kind:
                raise ValueError(f"Unexpected kind for {name}: {shard.get('kind')!r}")
            if expected_kind == "parity_only" and shard.get("importable") is not False:
                raise ValueError("Parity pilot shards must be marked importable=false")
            if int(shard.get("count", -1)) < 1:
                raise ValueError(f"Invalid row count for {name}")
            if not re.fullmatch(r"[0-9a-f]{64}", str(shard.get("sha256", ""))):
                raise ValueError(f"Invalid compressed shard SHA-256 for {name}")


def select_shards(
    manifest: Mapping[str, Any], mode: str = "pilot", max_shards: int = 0,
    all_pending: bool = False,
) -> list[dict[str, Any]]:
    """Select pilot parity files by default; never mix them with importable work."""
    validate_export_manifest(manifest)
    if max_shards < 0:
        raise ValueError("max_shards cannot be negative")
    if mode == "pilot":
        selected = [s for s in manifest["pilotShards"] if s.get("importable") is False]
        if not selected:
            raise ValueError("Export has no pilot shards")
        # The 68-text set is a superset of the fixed 48-text parity sample and
        # gives the first bounded Modal run the strongest diagnostic coverage.
        selected.sort(key=lambda item: (int(item["count"]) != 68, -int(item["count"])))
        selected = selected[:1]
    elif mode == "pending":
        selected = [s for s in manifest["pendingShards"] if s.get("importable") is True]
        if not selected:
            raise ValueError("Export has no pending shards")
        if max_shards == 0 and not all_pending:
            raise ValueError("Full pending execution requires explicit all_pending=True")
    else:
        raise ValueError("mode must be 'pilot' or 'pending'")
    if max_shards:
        selected = selected[:max_shards]
    return [dict(s) for s in selected]


def read_gzip_jsonl(path: Path, expected_count: int, expected_sha256: str) -> list[dict[str, str]]:
    """Validate a frozen shard and return its unique text/hash records."""
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"Compressed input hash mismatch: {path.name}")
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row in {path.name}:{line_number}")
            row = json.loads(line)
            text_hash, text = row.get("text_hash"), row.get("text")
            if not isinstance(text_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", text_hash):
                raise ValueError(f"Invalid text_hash in {path.name}:{line_number}")
            if not isinstance(text, str):
                raise ValueError(f"Non-string text in {path.name}:{line_number}")
            if text_hash in seen:
                raise ValueError(f"Duplicate text_hash in {path.name}: {text_hash}")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != text_hash:
                raise ValueError(f"Text/hash mismatch in {path.name}:{line_number}")
            seen.add(text_hash)
            records.append({"text_hash": text_hash, "text": text})
    if len(records) != expected_count:
        raise ValueError(f"{path.name} has {len(records)} rows; expected {expected_count}")
    return records


def make_work_items(
    manifest: Mapping[str, Any], shards: Sequence[Mapping[str, Any]],
    volume_prefix: str, deadline_unix: float, pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
    execution_id: str | None = None,
) -> list[dict[str, Any]]:
    """Build small map inputs that point at already-uploaded private-volume data."""
    validate_export_manifest(manifest)
    if pair_batch_size < 1:
        raise ValueError("pair_batch_size must be positive")
    execution_id = execution_id or uuid.uuid4().hex
    prefix = PurePosixPath(volume_prefix)
    if prefix.is_absolute() or any(part in ("", ".", "..") for part in prefix.parts):
        raise ValueError("volume_prefix must be a safe relative path")
    items = []
    for shard in shards:
        filename = _safe_name(str(shard["file"]))
        items.append({
            "export_id": str(manifest["exportId"]),
            "execution_id": execution_id,
            "run_id": str(manifest["runId"]),
            "source_sha256": str(manifest["sourceSha256"]),
            "model": str(manifest["model"]),
            "model_sha256": str(manifest["modelSha256"]),
            "model_assets_sha256": dict(manifest["modelAssetsSha256"]),
            "taxonomy_sha256": str(manifest["taxonomySha256"]),
            "score_semantics": str(manifest["scoreSemantics"]),
            "max_length": int(manifest["maxLength"]),
            "act_specs": [dict(act) for act in manifest["actSpecs"]],
            "shard": dict(shard),
            "volume_input_path": str(prefix / str(manifest["exportId"]) / filename),
            "deadline_unix": float(deadline_unix),
            "pair_batch_size": pair_batch_size,
            "message_batch_size": MESSAGE_BATCH_SIZE,
        })
    return items


class CudaNliRuntime:
    """CUDA/FP16 adapter with the exact taxonomy and truncation policy."""
    def __init__(self, model_dir: str, taxonomy_path: str, pair_batch_size: int):
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        from message_signatures.nli_runtime import _label_key, _prepare_pair

        if not torch.cuda.is_available():
            raise RuntimeError("Modal NLI worker requires an available CUDA device")
        torch.set_num_threads(4)
        self.torch = torch
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.pair_batch_size = pair_batch_size
        self.max_length = MAX_LENGTH
        self.model_dir = Path(model_dir)
        self.taxonomy_path = Path(taxonomy_path)
        self._prepare_pair = _prepare_pair

        taxonomy = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        self.act_specs = taxonomy["acts"]
        self.hypotheses = {str(a["id"]): str(a["hypothesis"]).strip() for a in self.act_specs}
        if len(self.hypotheses) != 12 or any(not v for v in self.hypotheses.values()):
            raise ValueError("Expected 12 unique, nonempty taxonomy hypotheses")
        self.act_ids = list(self.hypotheses)

        self.config = AutoConfig.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.id2label = {int(k): str(v) for k, v in self.config.id2label.items()}
        self.label2id = {_label_key(str(k)): int(v) for k, v in self.config.label2id.items()}
        entailment = [i for i, label in self.id2label.items() if _label_key(label) == "entailment"]
        if len(entailment) != 1 or self.label2id.get("entailment") != entailment[0]:
            raise ValueError("Local model config does not identify entailment unambiguously")
        if set(self.id2label) != set(range(int(self.config.num_labels))):
            raise ValueError("Model config label IDs are incomplete")
        self.entailment_id = entailment[0]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer_backend = self.tokenizer._tokenizer
        self.hypothesis_encodings = {
            act_id: self.tokenizer_backend.encode(text, add_special_tokens=False)
            for act_id, text in self.hypotheses.items()
        }
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            config=self.config, torch_dtype=self.dtype,
        ).eval().to(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def metadata(self, code_path: str | Path) -> dict[str, Any]:
        import importlib.metadata
        torch = self.torch
        from message_signatures import nli_runtime
        p = Path(code_path)
        execution_id = getattr(self, "execution_id", None)
        return {
            "executionId": execution_id,
            "backend": "cuda",
            "device": f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})",
            "dtype": "float16",
            "parameterCount": self.parameter_count,
            "modelSha256": sha256_file(self.model_dir / "model.safetensors"),
            "modelAssetsSha256": {
                path.name: sha256_file(path) for path in sorted(self.model_dir.glob("*"))
                if path.is_file() and path.suffix in (".json", ".model", ".jinja")
            },
            "taxonomySha256": getattr(self, "taxonomy_sha256", sha256_file(self.taxonomy_path)),
            "codeSha256": sha256_file(p),
            "runtimeSha256": sha256_file(Path(nli_runtime.__file__)),
            "dependencies": {
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "safetensors": importlib.metadata.version("safetensors"),
                "numpy": importlib.metadata.version("numpy"),
                "modal": importlib.metadata.version("modal"),
            },
            "sourceSha256": getattr(self, "source_sha256", None),
            "taxonomySha256": getattr(self, "taxonomy_sha256", None),
            "maxLength": self.max_length,
            "pairBatchSize": self.pair_batch_size,
            "messageBatchSize": MESSAGE_BATCH_SIZE,
            "scoreSemantics": "independent_entailment",
            "calibrated": False,
            "localFilesOnly": True,
            "trustRemoteCode": False,
        }

    def predict(self, texts: Sequence[str], deadline_unix: float) -> list[dict[str, Any]]:
        from message_signatures.nli_runtime import _length_order
        if time.time() >= deadline_unix:
            raise TimeoutError("Wall deadline reached before scoring this shard")
        messages = [str(t) for t in texts]
        enc_by_text = {
            text: self.tokenizer_backend.encode(text, add_special_tokens=False)
            for text in dict.fromkeys(messages)
        }
        pairs: list[dict[str, Any]] = []
        trunc_by_message: list[dict[str, dict[str, Any]]] = [dict() for _ in messages]
        for message_index, text in enumerate(messages):
            source = enc_by_text[text]
            for act_id in self.act_ids:
                features, dropped = self._prepare_pair(
                    self.tokenizer, source, self.hypothesis_encodings[act_id], MAX_LENGTH
                )
                trunc_by_message[message_index][act_id] = {
                    "message": dropped > 0,
                    "message_tokens": len(source.ids),
                    "message_tokens_dropped": dropped,
                    "hypothesis": False,
                }
                pairs.append({"message_index": message_index, "act_id": act_id, "features": features})
        order = _length_order([item["features"] for item in pairs], enabled=True)
        pair_scores = [0.0] * len(pairs)
        for start in range(0, len(order), self.pair_batch_size):
            if time.time() >= deadline_unix:
                raise TimeoutError("Wall deadline reached during shard scoring")
            batch_indices = order[start:start + self.pair_batch_size]
            features = [pairs[index]["features"] for index in batch_indices]
            encoded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
            encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.float()
                probs = self.torch.softmax(logits, dim=-1)[:, self.entailment_id]
            values = probs.cpu().tolist()
            for pair_index, score in zip(batch_indices, values):
                pair_scores[pair_index] = float(score)
        by_message: list[dict[str, float]] = [dict() for _ in messages]
        for pair, score in zip(pairs, pair_scores):
            by_message[pair["message_index"]][pair["act_id"]] = score
        results = []
        for scores, truncation in zip(by_message, trunc_by_message):
            results.append({
                "scores": scores,
                "truncation": {
                    "truncated": any(item["message"] for item in truncation.values()),
                    "perHypothesis": truncation,
                },
            })
        return results


_RUNTIME: CudaNliRuntime | None = None
_RUNTIME_KEY: tuple[str, str, int] | None = None


def _score_one_shard(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Pure worker body separated from Modal decorators for unit testing."""
    global _RUNTIME, _RUNTIME_KEY
    shard = spec["shard"]
    filename = _safe_name(str(shard["file"]))
    output_rows: list[dict[str, Any]] = []
    runtime_meta: dict[str, Any] = {}
    try:
        if spec["score_semantics"] != "independent_entailment" or int(spec["max_length"]) != MAX_LENGTH:
            raise ValueError("Shard semantics/length do not match this worker")
        input_path = Path(VOLUME_MOUNT) / spec["volume_input_path"]
        records = read_gzip_jsonl(input_path, int(shard["count"]), str(shard["sha256"]))
        taxonomy_path = HERE / "taxonomy.json"
        if sha256_file(taxonomy_path) != spec["taxonomy_sha256"]:
            raise ValueError("Bundled taxonomy hash differs from frozen export")
        model_path = Path(MODEL_DIR)
        actual_model_sha = sha256_file(model_path / "model.safetensors")
        if actual_model_sha != spec["model_sha256"]:
            raise ValueError("Mounted model weights do not match frozen export SHA-256")
        actual_assets = {
            path.name: sha256_file(path) for path in sorted(model_path.glob("*"))
            if path.is_file() and path.suffix in (".json", ".model", ".jinja")
        }
        if actual_assets != spec["model_assets_sha256"]:
            raise ValueError("Mounted model/tokenizer assets do not match frozen export")
        key = (str(model_path), actual_model_sha, int(spec["pair_batch_size"]))
        if _RUNTIME is None or _RUNTIME_KEY != key:
            _RUNTIME = CudaNliRuntime(str(model_path), str(taxonomy_path), int(spec["pair_batch_size"]))
            _RUNTIME_KEY = key
        runtime = _RUNTIME
        if runtime.hypotheses != {str(a["id"]): str(a["hypothesis"]).strip() for a in spec["act_specs"]}:
            raise ValueError("Export hypotheses differ from mounted taxonomy")
        if runtime.id2label != {0: "entailment", 1: "not_entailment"}:
            raise ValueError(f"Unexpected NLI label map: {runtime.id2label}")
        runtime.execution_id = str(spec["execution_id"])
        runtime.source_sha256 = str(spec["source_sha256"])
        runtime.taxonomy_sha256 = str(spec["taxonomy_sha256"])
        runtime_meta = runtime.metadata(__file__)
        texts = [row["text"] for row in records]
        for offset in range(0, len(texts), int(spec.get("message_batch_size", MESSAGE_BATCH_SIZE))):
            if time.time() >= float(spec["deadline_unix"]):
                raise TimeoutError("Wall deadline reached while processing this shard")
            batch = texts[offset:offset + int(spec.get("message_batch_size", MESSAGE_BATCH_SIZE))]
            predictions = runtime.predict(batch, float(spec["deadline_unix"]))
            for source, prediction in zip(records[offset:offset + len(batch)], predictions):
                if set(prediction["scores"]) != set(runtime.act_ids):
                    raise ValueError("Runtime did not score every taxonomy act")
                if any(not math.isfinite(s) or not 0.0 <= s <= 1.0 for s in prediction["scores"].values()):
                    raise ValueError("Model returned invalid entailment scores")
                output_rows.append({
                    "text_hash": source["text_hash"],
                    "scores": prediction["scores"],
                    "truncation": prediction["truncation"],
                    "runtime": runtime_meta,
                })
        if len(output_rows) != len(records):
            raise ValueError("Result count does not match input shard")
        return {"ok": True, "complete": True, "kind": shard["kind"], "shardId": shard["id"],
                "inputFile": filename,
                "inputSha256": shard["sha256"], "count": len(records), "rows": output_rows,
                "cloudExecution": runtime_meta}
    except Exception as exc:
        return {"ok": False, "complete": False, "kind": shard.get("kind"),
                "shardId": shard.get("id"), "inputFile": filename,
                "inputSha256": shard.get("sha256"), "count": int(shard.get("count", 0)),
                "rows": output_rows, "cloudExecution": runtime_meta,
                "error": f"{type(exc).__name__}: {exc}"}


if modal is not None:
    app = modal.App("ai-village-nli-modal")
    model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME)
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(
            "torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2",
            "safetensors==0.8.0", "numpy==2.5.3",
        )
        .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
              "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
        .add_local_python_source("message_signatures")
        .add_local_file(str(HERE / "taxonomy.json"), "/root/message_signatures/taxonomy.json")
    )

    @app.function(
        image=image,
        gpu="H100",
        cpu=4,
        memory=16_384,
        max_containers=MAX_CONTAINERS,
        timeout=MAX_WALL_SECONDS,
        scaledown_window=10,
        retries=0,
        volumes={VOLUME_MOUNT: model_volume.with_mount_options(read_only=True)},
    )
    def score_shard(spec: dict[str, Any]) -> dict[str, Any]:
        return _score_one_shard(spec)

    @app.local_entrypoint()
    def main(
        export_dir: str,
        output_dir: str,
        mode: str = "pilot",
        max_shards: int = 0,
        all_pending: bool = False,
        wall_seconds: int = MAX_WALL_SECONDS,
        pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
        volume_prefix: str = "jobpayloads",
    ) -> None:
        run_local(export_dir, output_dir, mode, max_shards, all_pending,
                  wall_seconds, pair_batch_size, volume_prefix)
else:  # pragma: no cover - makes pure utility functions importable without SDK.
    app = None
    score_shard = None


def _atomic_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as out:
        for row in rows:
            out.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)
    return sha256_file(path)


def run_local(
    export_dir: str | Path,
    output_dir: str | Path,
    mode: str = "pilot",
    max_shards: int = 0,
    all_pending: bool = False,
    wall_seconds: int = MAX_WALL_SECONDS,
    pair_batch_size: int = DEFAULT_PAIR_BATCH_SIZE,
    volume_prefix: str = "jobpayloads",
    *,
    map_function: Any = None,
) -> dict[str, Any]:
    """Run already-uploaded shard files and checkpoint each completed result locally."""
    if wall_seconds < 1 or wall_seconds > MAX_WALL_SECONDS:
        raise ValueError(f"wall_seconds must be 1..{MAX_WALL_SECONDS}")
    export_path = Path(export_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()
    manifest_path = export_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_export_manifest(manifest)
    selected = select_shards(manifest, mode, max_shards, all_pending)
    if mode == "pending" and max_shards == 0 and not all_pending:
        raise ValueError("Refusing an uncapped pending run without --all-pending")
    if map_function is None:
        if score_shard is None:
            raise RuntimeError("Modal SDK is unavailable; install the pinned Modal CLI first")
        map_function = score_shard

    # Verify each local export artifact before dispatch, then workers verify the
    # same digest again against the private Volume copy.
    for shard in selected:
        filename = _safe_name(str(shard["file"]))
        read_gzip_jsonl(export_path / filename, int(shard["count"]), str(shard["sha256"]))

    start = time.monotonic()
    deadline_unix = time.time() + wall_seconds
    execution_id = uuid.uuid4().hex
    specs = make_work_items(manifest, selected, volume_prefix, deadline_unix, pair_batch_size, execution_id)
    def bounded_specs():
        for spec in specs:
            if time.monotonic() - start >= wall_seconds:
                break
            yield spec
    completed: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    iterator = map_function.map(bounded_specs(), order_outputs=False, return_exceptions=True)
    for result in iterator:
        if isinstance(result, BaseException):
            errors.append({"error": f"{type(result).__name__}: {result}"})
            continue
        if not result.get("ok"):
            if result.get("rows"):
                stem = Path(str(result.get("inputFile", "partial"))).name
                if stem.endswith(".jsonl.gz"):
                    stem = stem[:-9]
                target = out_path / f"partial-{stem}.jsonl.gz"
                partial_sha = _atomic_gzip_jsonl(target, result["rows"])
                partial.append({
                    "id": result.get("shardId"), "inputFile": result.get("inputFile"),
                    "inputSha256": result.get("inputSha256"),
                    "outputFile": target.name, "outputSha256": partial_sha,
                    "count": len(result["rows"]), "kind": result.get("kind"),
                    "cloudExecution": result.get("cloudExecution", {}),
                })
            errors.append({"inputFile": result.get("inputFile"), "error": result.get("error")})
            continue
        if len(result.get("rows", [])) != int(result["count"]):
            errors.append({"inputFile": result.get("inputFile"), "error": "result cardinality mismatch"})
            continue
        stem = Path(str(result["inputFile"])).stem
        if stem.endswith(".jsonl"):
            stem = stem[:-6]
        if result["kind"] == "parity_only":
            target = out_path / f"parity-{stem}.jsonl.gz"
        else:
            target = out_path / f"result-{stem}.jsonl.gz"
        result_sha = _atomic_gzip_jsonl(target, result["rows"])
        completed.append({
            "id": result["shardId"], "inputFile": result["inputFile"],
            "inputSha256": result["inputSha256"], "count": result["count"],
            "outputFile": target.name, "outputSha256": result_sha,
            "kind": result["kind"], "cloudExecution": result["cloudExecution"],
        })

    all_selected_done = len(completed) == len(selected) and not errors
    all_pending_selected = mode == "pending" and len(selected) == len(manifest["pendingShards"])
    result_manifest = {
        "schemaVersion": 1,
        "exportId": manifest["exportId"], "runId": manifest["runId"],
        "manifestSha256": manifest["manifestSha256"],
        "sourceSha256": manifest["sourceSha256"], "model": manifest["model"],
        "modelSha256": manifest["modelSha256"],
        "modelAssetsSha256": manifest["modelAssetsSha256"],
        "taxonomySha256": manifest["taxonomySha256"],
        "maxLength": MAX_LENGTH, "scoreSemantics": "independent_entailment",
        "mode": mode, "complete": all_selected_done and (mode == "pilot" or all_pending_selected),
        "selectedShardCount": len(selected), "completedShardCount": len(completed),
        "elapsedSeconds": round(time.monotonic() - start, 3),
        "wallBudgetSeconds": wall_seconds, "pairBatchSize": pair_batch_size,
        "shards": completed, "partialShards": partial, "errors": errors,
        "cloudExecution": _common_execution_metadata(completed),
    }
    result_manifest["resultsManifestSha256"] = _result_manifest_sha256(result_manifest)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_tmp = out_path / "results-manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(result_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_tmp.replace(out_path / "results-manifest.json")
    return result_manifest


def _common_execution_metadata(completed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not completed:
        return {}
    keys = set(completed[0]["cloudExecution"])
    common = {key: completed[0]["cloudExecution"][key] for key in keys}
    for item in completed[1:]:
        meta = item["cloudExecution"]
        for key in list(common):
            if key in meta and meta[key] != common[key]:
                common.pop(key)
    return common


def _result_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "resultsManifestSha256"}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
