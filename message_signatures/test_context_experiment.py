import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from message_signatures.context_experiment import build_experiment


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    """Small deterministic tokenizer stand-in; tests do not load model files."""
    def encode(self, text, add_special_tokens=False):
        return _Encoding(text.split())

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def num_special_tokens_to_add(self, pair=True):
        return 3


def _fixture(tmp: Path):
    db = tmp / "fixture.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("""
      CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE messages(id TEXT PRIMARY KEY,day INTEGER,event_index INTEGER,date TEXT,
        timestamp TEXT,speaker TEXT,source_type TEXT,text TEXT,text_hash TEXT);
    """)
    active = {
        "id": "run-fixture", "model": "fixture-model", "model_sha256": "a" * 64,
        "model_assets_sha256": {"tokenizer.json": "b" * 64},
        "score_semantics": "independent_entailment", "question": "fixed question",
    }
    conn.execute("INSERT INTO metadata VALUES (?,?)", ("active_run", json.dumps(active)))
    conn.execute("INSERT INTO metadata VALUES (?,?)", ("source_sha256", json.dumps("c" * 64)))
    for day in range(1, 5):
        for event_index in range(5):
            text = f"day{day} message{event_index} exact."
            digest = hashlib.sha256(text.encode()).hexdigest()
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)", (
                f"d{day}-e{event_index}", day, event_index, f"2025-01-{day:02d}",
                f"2025-01-{day:02d}T12:{event_index:02d}:00Z", "Ada", "USER_TALK", text, digest,
            ))
    conn.commit()
    conn.close()
    taxonomy = tmp / "taxonomy.json"
    taxonomy.write_text(json.dumps({"acts": [
        {"id": f"act{i}", "hypothesis": f"message is about category {i}", "signature": "x -> y"}
        for i in range(12)
    ]}), encoding="utf-8")
    return db, taxonomy


class ContextExperimentTests(unittest.TestCase):
    def test_builds_three_paired_conditions_with_untruncated_exact_target(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            db, taxonomy = _fixture(tmp)
            out = tmp / "experiment"
            manifest = build_experiment(db, out, tokenizer=_Tokenizer(), tokenizer_path=tmp / "fake-tokenizer",
                                        taxonomy_path=taxonomy, sample_size=2, seed=13)
            self.assertEqual(manifest["count"], 6)
            self.assertEqual(manifest["rowsPerCondition"], {
                "isolated": 2, "real_context": 2, "shuffled_context": 2,
            })
            self.assertEqual(manifest["inputSha256"], manifest["payload"]["sha256"])
            with gzip.open(out / manifest["inputFile"], "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({row["text_hash"] for row in rows}), 6)
            for row in rows:
                self.assertEqual(hashlib.sha256(row["text"].encode()).hexdigest(), row["text_hash"])
                if row["condition"] == "isolated":
                    self.assertEqual(row["text"], row["target"])
                else:
                    self.assertTrue(row["text"].endswith(f"{row['speaker']}: {row['target']}"))
                    self.assertEqual(len(row["context"]), 2)
                self.assertLessEqual(row["premise_tokens"], manifest["maxPremiseTokens"])

    def test_rejects_target_above_cap_before_writing_export(self):
        class LongTargetTokenizer(_Tokenizer):
            def encode(self, text, add_special_tokens=False):
                if text.startswith("Ada:") and "Current" not in text:
                    return _Encoding(["t"] * 161)
                return super().encode(text, add_special_tokens)

        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            db, taxonomy = _fixture(tmp)
            # All targets are ineligible, so the builder must fail closed and
            # leave no partial experiment directory behind.
            out = tmp / "experiment"
            with self.assertRaisesRegex(ValueError, "eligible target"):
                build_experiment(db, out, tokenizer=LongTargetTokenizer(), tokenizer_path=tmp / "fake",
                                 taxonomy_path=taxonomy, sample_size=2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
