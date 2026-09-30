"""Model-free safety and bundle-contract tests for the Modal CUDA worker."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import tempfile
import unittest
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from typing import Any

from message_signatures import modal_worker as worker


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: dict[str, Any], excluded: str) -> str:
    payload = {key: item for key, item in value.items() if key != excluded}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return digest(raw)


def make_manifest(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    acts = [
        {"id": f"act{i}", "definition": f"definition {i}",
         "hypothesis": f"The message act is {i}.", "signature": "input -> output"}
        for i in range(12)
    ]
    pending_rows = [{"text_hash": digest(b"one"), "text": "one"},
                    {"text_hash": digest(b"two"), "text": "two"}]
    pilot_rows = [{"text_hash": digest(f"pilot-{i}".encode()), "text": f"pilot-{i}"}
                  for i in range(68)]
    shards = []
    for filename, rows, kind, importable in (
        ("pending-00000.jsonl.gz", pending_rows, "pending", True),
        ("pilot-48.jsonl.gz", pilot_rows[:48], "parity_only", False),
        ("pilot-68.jsonl.gz", pilot_rows, "parity_only", False),
    ):
        path = directory / filename
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        shards.append({"id": filename.removesuffix(".jsonl.gz"), "file": filename,
                       "count": len(rows), "sha256": worker.sha256_file(path),
                       "kind": kind, "importable": importable})
    manifest = {
        "schemaVersion": 1, "exportId": "export-test", "runId": "run-test",
        "sourceSha256": digest(b"source"), "engine": "nli",
        "model": worker.MODEL_NAME, "modelSha256": digest(b"weights"),
        "modelAssetsSha256": {"tokenizer.json": digest(b"tokenizer")},
        "tokenizerAssetsSha256": {"tokenizer.json": digest(b"tokenizer")},
        "taxonomySha256": digest(b"taxonomy"), "question": "choose",
        "actSpecs": acts, "scoreSemantics": "independent_entailment",
        "probabilitiesCalibrated": False, "maxLength": 512,
        "truncationPolicy": {"strategy": "only_first", "limit": 512},
        "reviewThresholds": {"top_score_below": .55, "margin_below": .15},
        "pendingShards": [shards[0]], "pilotShards": shards[1:],
    }
    manifest["manifestSha256"] = canonical_hash(manifest, "manifestSha256")
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, pending_rows


def fake_execution(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "executionId": "execution-test", "backend": "cuda", "device": "cuda:0 (NVIDIA H100)",
        "dtype": "float16", "modelSha256": manifest["modelSha256"],
        "modelAssetsSha256": manifest["modelAssetsSha256"],
        "sourceSha256": manifest["sourceSha256"], "taxonomySha256": manifest["taxonomySha256"],
        "runtimeSha256": digest(b"runtime"), "codeSha256": digest(b"worker"),
        "dependencies": {"torch": "2.x"}, "maxLength": 512,
        "scoreSemantics": "independent_entailment",
    }


class FakeMap:
    def __init__(self, results: list[dict[str, Any]]):
        self.results = results

    def map(self, specs, **kwargs):
        self.seen_specs = list(specs)
        self.kwargs = kwargs
        return iter(self.results)

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


class ModalWorkerTests(unittest.TestCase):
    def test_taxonomy_uses_explicit_packaged_container_path(self):
        self.assertEqual(str(worker.TAXONOMY_CONTAINER_PATH), "/root/message_signatures/taxonomy.json")
        self.assertEqual(worker.TAXONOMY_CONTAINER_PATH.name, "taxonomy.json")

    def test_cloud_execution_uses_exact_h100_identity(self):
        self.assertEqual(worker.GPU_TYPE, "H100!")

    def test_modal_version_metadata_can_be_missing_in_injected_runtime(self):
        with patch.object(worker, "modal", SimpleNamespace()):
            with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError("modal")):
                self.assertEqual(
                    worker.modal_sdk_version(),
                    "platform-injected; distribution metadata unavailable",
                )

    def test_worker_helpers_import_without_optional_modal_sdk(self):
        code = (
            "from message_signatures import modal_worker; "
            "assert modal_worker.modal is None; "
            "assert modal_worker.select_shards is not None; "
            "print('optional Modal import fallback ok')"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", code], cwd=Path(__file__).resolve().parents[1],
            text=True, capture_output=True, check=True,
        )
        self.assertIn("optional Modal import fallback ok", result.stdout)

    def test_default_pilot_selects_uploaded_68_superset_and_never_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = make_manifest(Path(temp))
            selected = worker.select_shards(manifest, mode="pilot", max_shards=1)
            self.assertEqual([s["file"] for s in selected], ["pilot-68.jsonl.gz"])
            self.assertFalse(selected[0]["importable"])

    def test_uncapped_pending_mode_requires_explicit_all_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = make_manifest(Path(temp))
            with self.assertRaisesRegex(ValueError, "explicit all_pending"):
                worker.select_shards(manifest, mode="pending")
            self.assertEqual(len(worker.select_shards(manifest, "pending", all_pending=True)), 1)

    def test_safe_volume_paths_and_export_manifest_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = make_manifest(Path(temp))
            worker.validate_export_manifest(manifest)
            with self.assertRaises(ValueError):
                worker._safe_name("../pending.jsonl.gz")
            bad = dict(manifest)
            bad["sourceSha256"] = digest(b"other")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                worker.validate_export_manifest(bad)

    def test_gzip_shard_hashes_and_text_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, rows = make_manifest(Path(temp))
            shard = manifest["pendingShards"][0]
            observed = worker.read_gzip_jsonl(Path(temp) / shard["file"], shard["count"], shard["sha256"])
            self.assertEqual(observed, rows)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                worker.read_gzip_jsonl(Path(temp) / shard["file"], shard["count"], digest(b"wrong"))

    def test_local_checkpoint_writer_emits_importer_schema_and_pins_export_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_dir, output_dir = root / "export", root / "out"
            export_dir.mkdir()
            manifest, rows = make_manifest(export_dir)
            acts = [act["id"] for act in manifest["actSpecs"]]
            result_rows = [{"text_hash": row["text_hash"],
                            "scores": {act: .25 for act in acts},
                            "truncation": {"truncated": False, "perHypothesis": {}},
                            "runtime": fake_execution(manifest)} for row in rows]
            response = {"ok": True, "complete": True, "kind": "pending",
                        "shardId": "pending-00000", "inputFile": "pending-00000.jsonl.gz",
                        "inputSha256": manifest["pendingShards"][0]["sha256"],
                        "count": len(rows), "rows": result_rows,
                        "performance": {"inputReadSeconds": .2, "modelLoadSeconds": 1.5,
                                        "inferenceSeconds": 2.0, "elapsedSeconds": 4.0,
                                        "messagesScored": 2, "pairsScored": 24,
                                        "messagesPerInferenceSecond": 1.0},
                        "cloudExecution": fake_execution(manifest)}
            fake_map = FakeMap([response])
            result = worker.run_local(export_dir, output_dir, mode="pending", all_pending=True,
                                      wall_seconds=600, gpu_type="L4", map_function=fake_map)
            self.assertTrue(result["complete"])
            self.assertEqual(result["manifestSha256"], manifest["manifestSha256"])
            self.assertEqual(result["shards"][0]["id"], "pending-00000")
            self.assertEqual(result["shards"][0]["count"], len(rows))
            self.assertTrue((output_dir / "result-pending-00000.jsonl.gz").is_file())
            self.assertEqual(result["gpuTypeRequested"], "L4")
            self.assertEqual(fake_map.options, {"gpu": "L4"})
            self.assertEqual(result["performance"]["messagesScored"], 2)
            self.assertEqual(result["performance"]["pairsScored"], 24)
            self.assertEqual(result["performance"]["messagesPerInferenceSecond"], 1.0)
            self.assertEqual(result["resultsManifestSha256"], canonical_hash(result, "resultsManifestSha256"))

    def test_partial_shard_is_separately_checkpointed_and_manifest_incomplete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_dir, output_dir = root / "export", root / "out"
            export_dir.mkdir()
            manifest, rows = make_manifest(export_dir)
            partial_response = {"ok": False, "complete": False, "kind": "pending",
                                "shardId": "pending-00000", "inputFile": "pending-00000.jsonl.gz",
                                "inputSha256": manifest["pendingShards"][0]["sha256"], "count": len(rows),
                                "rows": [{"text_hash": rows[0]["text_hash"], "scores": {},
                                          "truncation": {"truncated": False}}],
                                "cloudExecution": fake_execution(manifest), "error": "TimeoutError"}
            result = worker.run_local(export_dir, output_dir, mode="pending", all_pending=True,
                                      map_function=FakeMap([partial_response]))
            self.assertFalse(result["complete"])
            self.assertEqual(result["shards"], [])
            self.assertEqual(result["partialShards"][0]["count"], 1)
            self.assertTrue((output_dir / "partial-pending-00000.jsonl.gz").is_file())


if __name__ == "__main__":
    unittest.main()
