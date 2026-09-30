MESSAGE SIGNATURES — local message-act research, not task extraction

Open http://127.0.0.1:8765/signatures/ while `python3 viewer.py` is running.
The signatures page can pause or resume its local experimental NLI run. Resume
uses the configured local model and the currently indexed transcript; it does
not publish results or execute any task described in a message. The command-line
examples below let you choose a bounded pilot or explicitly start a full run.

What is indexed and what is scored
----------------------------------
The frozen source snapshot contains 183,485 AGENT_TALK and USER_TALK message
records. Exact-text hashing yields 178,052 unique message texts. The index keeps
every source record and locator; identical text can share one model inference.
Consequently, “classified messages” counts matching source records, while the
runner's progress counts unique text inputs. Activity events are outside this
chat-message index. The source SHA-256, message hash, day, and event index keep
results tied to their transcript snapshot.

The current model is the local ~184M-parameter
MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c NLI checkpoint. For each message,
it scores each of twelve author-defined act hypotheses independently. These
scores are neither mutually exclusive nor calibrated probabilities of the
message's true function. The top-scoring act is only a candidate; the runner's
review flag is a provisional heuristic, not evidence that an unflagged result
is correct. No task arguments are extracted and no action is verified.

Current run state and limits
----------------------------
The full-corpus run started on 30 September 2026, resuming the base-model pilot
under run ID d64365bb56121748c722. Check the live page for current completion
and throughput; early short messages run much faster than long messages, so
the remaining-time estimate changes. The original 64-unique-text pilot covered
394 source records in 133.51 seconds. Its 104-hour extrapolation is retained
as historical diagnostic evidence, not the live completion forecast.

The worker is independent of the web server and commits each batch. An idle-
sleep guard stays active until the worker exits. Closing the laptop or shutting
down can still interrupt computation; resume from the UI or command line.
The read-only export_report.py --wait process writes full-run-report.json on
completion or an explicit terminal failure/partial state, with provenance,
both count denominators, and up to 24 exact-message examples. It does not
classify anything itself. This file remains a compact summary of experimental
outputs; the SQLite database retains every per-message score and template.

The pilot already exposes a material failure: a “village” resume was assigned
assertion with an entailment score of 0.903 and was not flagged for review.
Other short diagnostics also confuse acknowledgment/check reporting. This is
not an accuracy estimate, but it is enough to show that the score and review
thresholds have not been validated for this corpus. Nano-Jev (~33M) and Qwen3
(~600M) experiments were also weak on the small authored diagnostics. The simpler
Qwen prompt still chose the first category on all 12 cases and retained the same
top act in only 1/12 rotated-choice comparisons. A one-case CPU float32 check
matched the MPS top act (maximum score difference 0.0020). Exact prompts and
results are preserved in simple_jev_diagnostic.json. Do not treat
current pilot labels or high scores as reliable classifications.

Bounded and full runs (from repository root)
--------------------------------------------
Build or verify the source index first:
  python3 message_signatures/store.py

Run a bounded 64-new-unique-text experiment:
  .venv-signatures/bin/python message_signatures/run.py --engine nli --device mps --limit 64

Explicitly start scoring every remaining unique text for the active model/config:
  .venv-signatures/bin/python message_signatures/run.py --engine nli --device mps

`--limit` bounds new unique texts in that invocation; `--day` can further bound
which pending texts are selected. The full-run command is intentionally explicit.
Predictions resume only when source snapshot, model assets, tokenizer/config,
taxonomy, runner/runtime code, device, and dependency provenance match. A
changed configuration has its own run ID. Progress and predictions are
committed by batch; a process lock prevents concurrent scoring writers. Ctrl-C
or the viewer's Pause control requests a stop after the current batch. The
viewer Resume control starts the configured full run. Closing the viewer does
not stop inference.

Model use and method
--------------------
Both model and tokenizer are loaded from local files with
`local_files_only=True` and `trust_remote_code=False`. The runtime makes no
outbound model calls. Model downloads were inbound from Hugging Face; weights
remain under `models/`.

The twelve conversational acts and their reusable DSPy-style signature
skeletons were authored before classification; they were not discovered from
the transcript. `adapter.py` provides a DSPy routing module, and `signatures.py`
defines contracts for possible later stages. In the current pass, a message is
assigned one primary act candidate. A message can contain several acts, and a
single label does not represent all of its content.

A defensible path from messages to reusable task contracts is:
  1. Manually define and revise act categories against representative source
     examples; keep ambiguous cases and category boundaries explicit.
  2. Segment mixed messages into verbatim communicative-act spans, preserving
     offsets and source IDs rather than forcing one class per whole message.
  3. Use preceding turns as context when a reply depends on what it answers.
  4. Extract task arguments only when supported by source spans; leave missing
     values unknown and retain evidence links for each field.
  5. Cluster only source-grounded, compatible input/output contracts, then
     review whether the resulting DSPy signatures generalize across examples.
  6. Evaluate on a human-adjudicated, held-out sample with documented
     disagreement, class-wise performance, and validated review thresholds.

Classification is not task extraction. A predicted act or signature template
does not supply message-specific arguments, prove that a reported check ran, or
show that an external action succeeded. Thanks may be social acknowledgment;
it is not by itself an executable task.

Diagnostics and references
-------------------------
pilot.json records the Nano-Jev experiment; jev_pilot.json records Qwen3 with
first-answer-token scoring; nli_pilot.json records the DeBERTa xsmall comparison;
base-pilot.json records the stronger DeBERTa base comparison. Their synthetic
examples are manually authored inspection diagnostics, not a gold set or a
corpus accuracy measure. The current base pilot is persisted in the local
signature database. No full-corpus classification has been run.

https://dspy.ai/3.1.3/learn/programming/signatures/
https://huggingface.co/sdmlai/nano-jev
https://huggingface.co/Qwen/Qwen3-0.6B
https://github.com/lookski/openjev
https://huggingface.co/MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33
Transcript attribution: AI Digest / AI Village, aidigestorg/ai-village.
