"""Offline, model-free robustness checks for the NLI benchmark harness."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from message_signatures.benchmark import (
    _cache_key,
    compare_rows,
    edge_fixtures,
    fixed_hash_sample,
    sha256_bytes,
)


def _result(scores: dict[str, float], *, truncated: bool = False, dropped: int = 0) -> dict:
    return {
        "probabilities": scores,
        "truncated": truncated,
        "truncation": {
            "message": truncated,
            "message_tokens_dropped_max": dropped,
            "per_hypothesis": {
                act: {"message": truncated, "message_tokens_dropped": dropped}
                for act in scores
            },
        },
    }


class FixedHashSampleTests(unittest.TestCase):
    def test_sample_is_hash_ordered_unique_and_validates_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "messages.sqlite"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
                "CREATE TABLE messages(text_hash TEXT NOT NULL,text TEXT NOT NULL);"
            )
            expected_source = "a" * 64
            connection.execute("INSERT INTO metadata VALUES (?,?)", ("source_sha256", json.dumps(expected_source)))
            texts = ["delta", "alpha", "charlie", "bravo", "alpha"]
            for text in texts:
                key = hashlib.sha256(text.encode()).hexdigest()
                connection.execute("INSERT INTO messages VALUES (?,?)", (key, text))
            connection.commit()
            connection.close()

            first = fixed_hash_sample(3, database, expected_source)
            second = fixed_hash_sample(3, database, expected_source)
            self.assertEqual(first, second)
            self.assertEqual([item["text_hash"] for item in first], sorted(item["text_hash"] for item in first))
            self.assertEqual(len({item["text_hash"] for item in first}), 3)
            with self.assertRaises(RuntimeError):
                fixed_hash_sample(3, database, "b" * 64)


class ComparisonTests(unittest.TestCase):
    def test_comparison_reports_score_top_truncation_and_review_agreement(self) -> None:
        scores = {"request": 0.7, "question": 0.2, "status": 0.1}
        baseline = [_result(scores), _result(scores, truncated=True, dropped=8)]
        candidate = [_result(scores), _result(scores, truncated=True, dropped=8)]
        result = compare_rows(["please inspect", "long source"], baseline, candidate)
        self.assertEqual(result["topLabelAgreement"], 2)
        self.assertEqual(result["maximumAbsoluteScoreDelta"], 0.0)
        self.assertEqual(result["meanAbsoluteScoreDelta"], 0.0)
        self.assertEqual(result["truncationEquality"], 2)
        self.assertEqual(result["reviewFlagAgreement"], 2)

    def test_comparison_detects_numeric_and_truncation_drift(self) -> None:
        baseline = [_result({"request": 0.7, "question": 0.3})]
        candidate = [_result({"request": 0.61, "question": 0.39}, truncated=True, dropped=3)]
        result = compare_rows(["text"], baseline, candidate)
        self.assertEqual(result["topLabelAgreement"], 1)
        self.assertAlmostEqual(result["maximumAbsoluteScoreDelta"], 0.09)
        self.assertEqual(result["truncationEquality"], 0)

    def test_edge_fixtures_cover_empty_unicode_and_long_truncation(self) -> None:
        fixtures = {item["id"]: item["text"] for item in edge_fixtures()}
        self.assertEqual(fixtures["fixture-empty"], "")
        self.assertIn("✅", fixtures["fixture-unicode-punctuation"])
        self.assertGreater(len(fixtures["fixture-truncated-long"]), 512)

    def test_cache_key_changes_with_backend_relevant_settings(self) -> None:
        base = _cache_key("s", "m", "t", "c", "mps", 32)
        self.assertNotEqual(base, _cache_key("s", "m", "t", "c", "cpu", 32))
        self.assertNotEqual(base, _cache_key("s", "m", "t", "c", "mps", 8))
        self.assertEqual(sha256_bytes(b"sample"), hashlib.sha256(b"sample").hexdigest())


if __name__ == "__main__":
    unittest.main()
