import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import export_dataset


class DatasetExportTests(unittest.TestCase):
    def test_read_snapshot_accounts_rows_and_keeps_prediction_epoch_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tiny.sqlite"
            out = root / "export"
            conn = sqlite3.connect(db)
            conn.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE messages (
                    id TEXT PRIMARY KEY, day INTEGER, event_index INTEGER, date TEXT,
                    timestamp TEXT, speaker TEXT, source_type TEXT, text TEXT, text_hash TEXT
                );
                CREATE TABLE predictions (
                    run_id TEXT, text_hash TEXT, label TEXT, top_score REAL, margin REAL,
                    needs_review INTEGER, truncated INTEGER, result TEXT,
                    PRIMARY KEY(run_id,text_hash)
                );
            """)
            active = {"id": "run-1", "runtime_settings": {"model_dir": "/secret/model", "device": "cpu"}}
            progress = {"state": "running", "run_id": "run-1", "execution_epoch_id": "epoch-2", "execution_epoch_number": 2}
            for key, value in (("active_run", active), ("progress", progress), ("source_sha256", "a" * 64)):
                conn.execute("INSERT INTO metadata VALUES (?,?)", (key, json.dumps(value)))
            conn.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)", (
                "msg-1", 4, 7, "2025-04-04", "time-string", "agent-A", "AGENT_TALK", "sample", "hash-1"))
            result = {
                "primary_candidate": "status_report", "score_semantics": "independent_entailment",
                "run_id": "run-1", "execution_epoch_id": "epoch-2",
                "runtime": {"truncation": {"message": True, "per_hypothesis": {"status_report": {"message": True}}}},
            }
            conn.execute("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)", (
                "run-1", "hash-1", "status_report", 0.61, 0.12, 1, 1, json.dumps(result)))
            conn.commit()
            before = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
            conn.close()

            manifest = export_dataset.export(db, out, shard_size=1)
            export_dataset.write_readme(out, manifest)

            self.assertEqual(manifest["exported_records"], before)
            self.assertEqual(manifest["snapshot"]["classified_source_records"], 1)
            self.assertEqual(manifest["snapshot"]["execution_epoch_id"], "epoch-2")
            self.assertEqual(manifest["model_run"]["runtime_settings"], {"device": "cpu"})
            self.assertEqual(sum(f["records"] for f in manifest["files"]), 1)
            shard = out / manifest["files"][0]["path"]
            with gzip.open(shard, "rt", encoding="utf-8") as stream:
                row = json.loads(stream.readline())
            prediction = row["prediction"]
            self.assertEqual(prediction["run_id"], "run-1")
            self.assertEqual(prediction["execution_epoch_id"], "epoch-2")
            self.assertTrue(prediction["runtime_truncation"]["per_hypothesis"]["status_report"]["message"])
            self.assertEqual(row["source_locator"]["timestamp"], "time-string")
            self.assertNotIn("gated: true", (out / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
