"""Bounded synthetic diagnostic and candidate-order invariance probe for NLI.

Uses only the twelve authored examples in pilot.py; it does not label or sample
the message corpus. Expected labels are diagnostic hints, never a gold set.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pilot import CASES, QUESTION
from nli_runtime import NLIRuntime

TAXONOMY = HERE / "taxonomy.json"
MODEL_DIR = ROOT / "models" / "deberta-xsmall-zeroshot"


def taxonomy_options() -> dict[str, str]:
    data = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    acts = data.get("acts")
    if not isinstance(acts, list) or len(acts) != 12:
        raise ValueError("taxonomy.json must define exactly twelve acts")
    return {str(act["id"]): str(act.get("hypothesis") or "") for act in acts}


def run_pilot(device: str = "auto", batch_size: int = 32) -> dict:
    options = taxonomy_options()
    load_start = time.perf_counter()
    runtime = NLIRuntime(model_dir=MODEL_DIR, device=device, batch_size=batch_size)
    load_seconds = time.perf_counter() - load_start
    texts = [str(case["text"]) for case in CASES]

    start = time.perf_counter()
    baseline = runtime.predict(texts, options, QUESTION)
    canonical_seconds = time.perf_counter() - start

    # Reorder candidates while holding every source/hypothesis pair constant.
    # A pairwise NLI classifier should not depend on dictionary order.
    reverse_options = dict(reversed(list(options.items())))
    start = time.perf_counter()
    reordered = runtime.predict(texts, reverse_options, QUESTION)
    reordered_seconds = time.perf_counter() - start

    predictions = []
    max_abs_diff = 0.0
    stable_top = 0
    for case, first, second in zip(CASES, baseline, reordered):
        differences = {
            act_id: abs(float(first["probabilities"][act_id]) - float(second["probabilities"][act_id]))
            for act_id in options
        }
        row_max_diff = max(differences.values(), default=0.0)
        max_abs_diff = max(max_abs_diff, row_max_diff)
        ranked = sorted(first["probabilities"].items(), key=lambda item: item[1], reverse=True)
        reverse_ranked = sorted(second["probabilities"].items(), key=lambda item: item[1], reverse=True)
        same_top = bool(ranked and reverse_ranked and ranked[0][0] == reverse_ranked[0][0])
        stable_top += int(same_top)
        predictions.append({
            "id": case["id"],
            "message": case["text"],
            "diagnosticExpectedAct_notGold": case["expectedAct"],
            "topCandidate": ranked[0][0] if ranked else None,
            "rankedCandidates": [{"act": act, "entailmentScore": score} for act, score in ranked],
            "orderInvariantTopCandidate": same_top,
            "maxAbsScoreDifferenceAfterReordering": row_max_diff,
            "truncated": first["truncated"],
            "truncation": first["truncation"],
            "scoreSemantics": "independent_entailment; scores do not sum to one",
            "calibrated": False,
        })
    return {
        "title": "Local DeBERTa NLI message-act synthetic diagnostic",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
        "modelPath": str(MODEL_DIR),
        "method": "For each taxonomy act, pair the source message as premise with the act hypothesis; read the entailment class softmax probability independently.",
        "scoring": {
            "semantics": "independent_entailment",
            "candidateScoresSumToOne": False,
            "calibrated": False,
            "labelMappingReadFromModelConfig": runtime.metadata()["label_mapping"],
            "entailmentLabel": runtime.metadata()["entailment_label"],
            "truncation": "Pair tokenization uses truncation=only_first at 512 tokens; taxonomy hypotheses and pair special tokens are retained. Per-act source-token truncation counts are reported.",
        },
        "diagnosticSet": {
            "kind": "same twelve synthetic authored examples as pilot.py",
            "expectedLabels": "Manually authored inspection hints, not gold labels; no accuracy metric is computed.",
            "count": len(CASES),
            "acts": list(options),
        },
        "orderInvariance": {
            "method": "Compare same per-act premise/hypothesis pairs with candidate mapping insertion order reversed.",
            "messagesCompared": len(CASES),
            "topCandidateAgreement": stable_top,
            "maxAbsoluteScoreDifference": max_abs_diff,
            "within1e-5": max_abs_diff <= 1e-5,
            "note": "A numerical order-invariance diagnostic, not act accuracy or calibration.",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "runtime": runtime.metadata(),
        "benchmark": {
            "syntheticMessages": len(CASES),
            "taxonomyActs": len(options),
            "crossEncoderPairsPerPass": len(CASES) * len(options),
            "loadSeconds": load_seconds,
            "canonicalPassSeconds": canonical_seconds,
            "reorderedPassSeconds": reordered_seconds,
            "batchSize": batch_size,
        },
        "predictions": predictions,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run_pilot(args.device, args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
