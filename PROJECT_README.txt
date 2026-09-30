AI Village Research
===================

Start the local site from this directory:
  python3 viewer.py
  http://127.0.0.1:8765/

The root is the research hub. Its four views are:
  /village/       Original transcript browser, with day/event deep links
  /analysis.html  Quantitative analysis and daily rhythm
  /fieldnotes/    Source-linked investigations and local review
  /signatures/    Experimental local message classification and DSPy templates

Legacy /?day=...&event=... links redirect to the Village browser. Shared visual
styles live in shared/site.css. Page logic is dependency-free browser JavaScript.
The server uses the Python standard library, caches recently read days, supports
gzip responses, and revalidates static pages/styles with ETags.

Source data
-----------
The original dataset documentation remains in README.md and SCHEMA.md.
The local viewer reads village-transcript.json and builds .viewer-index.json.
Source: https://huggingface.co/datasets/aidigestorg/ai-village
The served transcript contains 375,426 events across 404 day records. The
signature index includes 183,485 chat records and 178,052 unique exact texts.
Snapshot identifiers accompany the analyses; source changes require rebuilding.

Local classification
--------------------
See message_signatures/README.txt for the installed model, dependencies,
commands, and limitations. The current full run scores all remaining unique
texts with the local DeBERTa base NLI model on Apple MPS. Each completed batch
is saved to SQLite. Pause/resume controls are in /signatures/.

message_signatures/run-checkpoint.json is a dated, compact progress snapshot,
not a claim of full completion. The read-only completion exporter automatically
writes message_signatures/full-run-report.json when the running job ends:
  .venv-signatures/bin/python message_signatures/export_report.py --wait

The live page and database hold current progress. A finished computation still
produces experimental labels: this classifier has not been validated against
human annotations, and reusable templates are not extracted task arguments.

Git and local artifacts
-----------------------
origin is git@github.com:maujim/ai-village-analysis.git; upstream preserves the
Hugging Face dataset remote. Python/HTML/CSS, analysis JSON, model configuration
and download provenance are committed. Model weights, the virtual environment,
runtime logs, locks, and SQLite database stay local and are ignored by Git.
The project history is separate from the upstream dataset export history.

Checks
------
  .venv-signatures/bin/python -m unittest test_viewer message_signatures.test_store message_signatures.test_control message_signatures.test_export_report fieldnotes.test_server
  python3 analysis/validate.py
  python3 fieldnotes/validate.py

HTTP tests use a temporary loopback server and isolated fixtures. Process-control
tests mock process creation and signals; they do not interrupt the real job.
