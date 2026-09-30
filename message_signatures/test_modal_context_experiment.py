"""Model-free checks for the isolated Modal context experiment worker."""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from message_signatures import modal_context_experiment as worker


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_input(path: Path) -> tuple[list[dict], str]:
    rows = []
    for i in range(128):
        target = f"target message number {i}"
        base_hash = digest(target)
        for condition, premise in (
            ("isolated", target),
            ("real_context", f"prior real messages {i} {target}"),
            ("shuffled_context", f"length matched shuffled messages {i} {target}"),
        ):
            rows.append({
                "id": f"target-{i}-{condition}", "base_text_hash": base_hash,
                "condition": condition, "target": target, "text": premise,
                "text_hash": digest(premise), "target_tokens": 5, "context_tokens": 0,
                "source_locator": f"day-{i}", "context_locators": [],
            })
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return rows, worker.sha256_file(path)


class FakeRuntime:
    def __init__(self, model_dir, taxonomy_path, pair_batch_size):
        self.tokenizer_backend = SimpleNamespace(encode=lambda text, add_special_tokens=False: SimpleNamespace(ids=text.split()))
        self.act_ids = [f"act{i}" for i in range(12)]
        self.hypotheses = {key: "hypothesis" for key in self.act_ids}

    def metadata(self, path):
        return {"executionId": "ctx-test", "codeSha256": digest("code")}

    def predict(self, texts, deadline):
        return [{"scores": {key: 0.5 for key in self.act_ids},
                 "truncation": {"truncated": False, "perHypothesis": {}}} for _ in texts]


def make_spec(path: Path, input_sha: str) -> dict:
    return {
        "experimentId": "experiment-test", "inputPath": "jobpayloads/experiment-test/context-input.jsonl.gz",
        "inputFile": "context-input.jsonl.gz", "inputSha256": input_sha, "count": 384,
        "sourceSha256": digest("source"), "model": worker.MODEL_NAME,
        "modelSha256": digest("weights"), "modelAssetsSha256": {"tokenizer.json": digest("tokenizer")},
        "taxonomySha256": digest("taxonomy"), "inputManifestSha256": digest("manifest"),
        "gpuHourlyRateUsd": 1.0, "maxInputTokens": 384, "maxLength": 512,
    }


class ModalContextExperimentTests(unittest.TestCase):
    def test_context_jsonl_hashes_and_triplets_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.jsonl.gz"
            rows, zipped_hash = write_input(path)
            result = worker.read_experiment_rows(path, zipped_hash, 384)
            self.assertEqual(len(result), 384)
            self.assertEqual(len({row["base_text_hash"] for row in result}), 128)
            with self.assertRaisesRegex(ValueError, "compressed SHA-256"):
                worker.read_experiment_rows(path, digest("wrong"), 384)

    def test_manifest_digest_is_canonical_and_pins_limits(self):
        manifest = {
            "experimentId": "e", "inputFile": "context-input.jsonl.gz", "inputSha256": digest("input"),
            "count": 384, "sourceSha256": digest("source"), "taxonomySha256": digest("tax"),
            "model": worker.MODEL_NAME, "modelSha256": digest("model"),
            "modelAssetsSha256": {"tokenizer.json": digest("tok")},
            "maxInputTokens": 384, "maxLength": 512,
            "question": worker.QUESTION, "targetCount": 128,
            "rowsPerCondition": {condition: 128 for condition in worker.CONDITIONS},
        }
        manifest["manifestSha256"] = worker.canonical_sha256(manifest)
        worker.validate_input_manifest(manifest)
        manifest["count"] = 1
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch|row/token bounds"):
            worker.validate_input_manifest(manifest)

    def test_worker_returns_all_condition_scores_and_cost_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            mount = Path(temp)
            input_path = mount / "jobpayloads/experiment-test/context-input.jsonl.gz"
            input_path.parent.mkdir(parents=True)
            _, input_sha = write_input(input_path)
            spec = make_spec(input_path, input_sha)
            with patch.object(worker, "validate_runtime_assets", return_value=None):
                result = worker.score_experiment(spec, model_path=mount, taxonomy_path=mount / "taxonomy.json",
                                                 runtime_class=FakeRuntime, volume_mount=temp)
            self.assertTrue(result["complete"], result["error"])
            self.assertEqual(result["count"], 384)
            self.assertEqual(result["performance"]["pairsScored"], 4608)
            self.assertEqual(result["performance"]["gpuHourlyRateUsd"], 1.0)
            self.assertEqual(len(result["rows"]), 384)
            self.assertEqual({row["condition"] for row in result["rows"]}, set(worker.CONDITIONS))

    def test_context_target_must_be_preserved_at_end_of_premise(self):
        row = {"id": "x", "base_text_hash": digest("target"), "condition": "real_context",
               "target": "target", "text": "target omitted", "text_hash": digest("target omitted")}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "preserve the entire target"):
                worker.read_experiment_rows(path, worker.sha256_file(path), 1)

    def test_output_is_atomic_and_checksummed(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "result.json"
            worker.write_output(target, {"complete": True, "rows": []})
            payload = json.loads(target.read_text(encoding="utf-8"))
            self.assertTrue(payload["complete"])
            self.assertEqual(payload["resultSha256"], worker.canonical_sha256({k: v for k, v in payload.items() if k != "resultSha256"}))
            self.assertEqual((Path(str(target) + ".sha256")).read_text().strip(), worker.sha256_file(target))


if __name__ == "__main__":
    unittest.main()
