# Modal CUDA continuation runbook

This is an operator guide for continuing the existing local message-signature run on CUDA. It does not create a new run or reclassify existing prediction rows. Cloud work is only a proposal until a numeric parity pilot is reviewed and approved.

## Frozen inference contract

- Logical run and source snapshot stay fixed. The export manifest pins the run ID, source SHA-256, model weights/assets, taxonomy, tokenizer inputs, score semantics, and 512-token truncation policy.
- Model: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c`, 184,423,682 parameters, CUDA FP16. The class hypothesis set and order come from the frozen 12-act taxonomy.
- Scores remain independent entailment scores (they do not sum to one), uncalibrated, and subject to the existing review thresholds. Per-hypothesis truncation detail is retained.
- Modal workers use the private Volume `ai-village-classification-20260930`; the configured worker requests `H100!` (fixed H100 selection, no automatic upgrade), a maximum of 8 containers, a 600-second function timeout, and zero automatic retries. Workers mount the Volume read-only, use local files only, and do not read or write the local SQLite database.
- Pending corpus shards and parity-only pilot shards are separate. Pilot shards must never be imported as predictions. Import is append-only: a shard is rejected if any of its text hashes already has a prediction in the run.

## Operator sequence

0. Cloud prerequisites: install the optional pinned client and authenticate the Modal CLI only when preparing a cloud run:

   ```sh
   .venv-signatures/bin/python -m pip install -r message_signatures/requirements-modal.txt
   modal setup
   ```

   Modal requires a valid payment method for GPU execution. **The current workspace is blocked at this prerequisite: no payment method is configured, so no H100 inference has been launched and no GPU spend has occurred.** Stop here until the account owner has completed billing setup; then resume the already-authorized pilot. Modal documents the payment requirement in its [GPU guide](https://modal.com/docs/guide/gpu) and [billing guide](https://modal.com/docs/guide/billing).

1. Pause the local classifier from the signatures page and verify its state is `paused`, `interrupted`, or otherwise idle. Do not export while the local process is still scoring; this avoids wasting a frozen shard on text that the local run classifies before cloud import.

2. Export a new frozen bundle. Use the active run ID shown by the status endpoint or UI, and choose an output path that does not already exist:

   ```sh
   RUN_ID='<active logical run id>'
   BUNDLE="message_signatures/cloud-exports/modal-${RUN_ID}"
   .venv-signatures/bin/python -m message_signatures.cloud_transfer export \
     --run-id "$RUN_ID" --shard-size 20000 --output "$BUNDLE"
   ```

   The export reads the local database through a read-only SQLite transaction. It writes a hashed manifest, importable `pendingShards`, and non-importable `pilotShards` (`kind: parity_only`). The 20,000-text shard target avoids hundreds of small Modal calls; begin with at most 8 pending shards per invocation so a wave fits the 8-container cap, then re-export after imports to get a fresh manifest of the remaining work. Handle the exported text as sensitive source data.

3. Prepare the named private Volume before dispatch. It must already contain the exact model and tokenizer files under `model/` (mounted as `/mnt/village/model/`). Upload the selected pilot file beneath `jobpayloads/<exportId>/`; the worker resolves inputs from that directory. For example, after confirming the export ID and Volume contents:

   ```sh
   EXPORT_ID='<export id from manifest.json>'
   modal volume put ai-village-classification-20260930 \
     "$BUNDLE/pilot-68.jsonl.gz" \
     "jobpayloads/$EXPORT_ID/pilot-68.jsonl.gz"
   ```

   `modal volume put` uploads local files to a Modal Volume; the scoring function mounts the same Volume read-only. Upload only the bundle files needed for the selected run. Do not upload the source SQLite database, unrelated archives, local environment files, or credentials.

4. Run the parity-only pilot first. The default export includes a 68-row parity shard: it contains the fixed 48-message sample plus 20 additional texts. The worker orders the 68-row shard first; it does not offer an ID filter. Compare all 68 rows against the root-reviewed local baseline (including the fixed 48 common hashes) and use the extra 20 only as supplemental diagnostics. Keep the invocation within the 600-second wall budget:

   ```sh
   RESULTS="message_signatures/cloud-results/${EXPORT_ID}-pilot"
   modal run message_signatures/modal_worker.py \
     --export-dir "$BUNDLE" --output-dir "$RESULTS" \
     --mode pilot --max-shards 1 --wall-seconds 600
   ```

   Inspect `results-manifest.json`, every output shard checksum/count, CUDA device and FP16 metadata, model/tokenizer/taxonomy hashes, and all pilot scores. Produce the read-only comparison report:

   ```sh
   .venv-signatures/bin/python -m message_signatures.modal_parity \
     --export-dir "$BUNDLE" --results-dir "$RESULTS" \
     --local-baseline /private/tmp/ai-village-modal-pilot/pilot-local-baseline.json \
     --output "$RESULTS/parity-report.json"
   ```

   A report marked `pass` requires maximum absolute score delta ≤ 0.02 plus exact top-label, review-status, and truncation agreement for every message. It is a comparison report, **not an approval artifact**. The pilot is not a quality evaluation; do not infer classification accuracy from it. Do not process or import pending corpus shards before the root reviewer records a passing numeric comparison.

5. Only after explicit parity approval, run bounded pending batches. Upload their pending shard files to the corresponding Volume directory, then use an explicit shard cap and 600-second budget. Increase the cap only from observed complete-shard throughput; each Modal invocation has its own dispatch and timeout risk. `--all-pending` is an explicit opt-in for uncapped pending work and should not be used for the initial run.

6. Pause the local classifier before each import. The importer verifies that the frozen source, active run, model assets, taxonomy, truncation/scoring semantics, parity approval, input hashes, result hashes, and row counts still match. Import one `pending-xxxxx` shard at a time:

   ```sh
   .venv-signatures/bin/python -m message_signatures.cloud_transfer import-shard \
     --bundle "$BUNDLE" --results "$RESULTS" \
     --shard-id '<pending shard id>' \
     --parity-approval '<root-approved parity approval JSON>'
   ```

   The importer records the CUDA execution as a separate provenance epoch. It never overwrites existing predictions. A conflict, hash mismatch, missing shard, active local runner, or unapproved result must stop the import.

7. Refresh the run report and signatures UI after imports. Confirm unique prediction totals and remaining unique texts, then resume local scoring only if that is the chosen next step. Do not describe the corpus as complete unless every unique source text is classified and the report confirms completion.

## Interface gate

The frozen bundle contains `pendingShards` and `pilotShards`; pilot entries are marked `kind: parity_only` and `importable: false`. Result entries preserve the source shard ID, count, input filename/hash, output filename/hash, and CUDA execution metadata required by `cloud_transfer.import_shard`. Run the local contract tests before a paid pilot:

```sh
.venv-signatures/bin/python -m unittest \
  message_signatures.test_modal_worker \
  message_signatures.test_cloud_transfer
```

The cloud pilot is already authorized. Launch remains blocked only by the missing Modal payment method described above; after billing setup, continue with the pilot sequence.

The Modal worker currently sets the resource limits above in `modal_worker.py`. Modal CLI [volume upload](https://modal.com/docs/cli/latest/volume) and [`modal run`](https://modal.com/docs/cli/latest/run) references describe the commands used here. See [Modal Volume guidance](https://modal.com/docs/guide/volumes) for Volume persistence and read-only mount behavior.
