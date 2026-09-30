"""Run the fixed synthetic act diagnostic through the local Jev logits readout.

Uses exactly the twelve authored examples in pilot.py. Expected acts are
inspection hints, not gold labels; no accuracy metric is calculated. A second
pass rotates the A-L choice order per message to reveal prompt-order
sensitivity. This does not label or sample the transcript corpus.
"""
from __future__ import annotations

import argparse
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

from pilot import CASES, QUESTION  # shared synthetic cases; no corpus sampling
from jev_runtime import JevRuntime

TAXONOMY = HERE / "taxonomy.json"
MODEL_DIR = ROOT / "models" / "qwen3-0.6b"


def taxonomy_options(path: Path) -> dict[str, str]:
    taxonomy = json.loads(path.read_text(encoding="utf-8"))
    acts = taxonomy.get("acts")
    if not isinstance(acts, list) or len(acts) != 12:
        raise ValueError("taxonomy.json must provide exactly 12 acts")
    options: dict[str, str] = {}
    for act in acts:
        act_id = str(act["id"])
        definition = str(act.get("definition") or "").strip()
        if not definition or act_id in options:
            raise ValueError("act IDs must be unique and definitions nonempty")
        options[act_id] = definition
    return options


def run_pilot(
    device: str = "auto",
    batch_size: int = 4,
    model_dir: Path = MODEL_DIR,
    taxonomy_path: Path = TAXONOMY,
    max_message_tokens: int = 768,
) -> dict:
    options = taxonomy_options(taxonomy_path)
    runtime_start = time.perf_counter()
    runtime = JevRuntime(
        model_dir=model_dir,
        device=device,
        batch_size=batch_size,
        max_message_tokens=max_message_tokens,
        taxonomy_path=taxonomy_path,
    )
    load_seconds = time.perf_counter() - runtime_start
    texts = [str(case["text"]) for case in CASES]

    first_start = time.perf_counter()
    baseline = runtime.predict(texts, options, QUESTION)
    first_pass_seconds = time.perf_counter() - first_start

    # One shifted choice ordering per diagnostic message, covering varied
    # rotations while keeping this a bounded 12-message stability probe.
    rotations = [((i % 11) + 1) for i in range(len(texts))]
    rotated_start = time.perf_counter()
    rotated = runtime.predict(texts, options, QUESTION, choice_rotation=rotations)
    rotated_pass_seconds = time.perf_counter() - rotated_start

    predictions = []
    for case, base, shift in zip(CASES, baseline, rotated):
        ranked = sorted(base["probabilities"].items(), key=lambda item: item[1], reverse=True)
        shifted_ranked = sorted(shift["probabilities"].items(), key=lambda item: item[1], reverse=True)
        predictions.append({
            "id": case["id"],
            "message": case["text"],
            "diagnosticExpectedAct_notGold": case["expectedAct"],
            "topAct": base["top_option"],
            "rankedActs": [{"act": act, "score": score} for act, score in ranked],
            "rotatedChoiceOrder": shift["choice_rotation"],
            "rotatedTopAct": shift["top_option"],
            "rotationStableTopAct": base["top_option"] == shift["top_option"],
            "rotatedRankedActs": [{"act": act, "score": score} for act, score in shifted_ranked],
            "truncated": base["truncated"],
            "truncation": base["truncation"],
            "probabilitiesUncalibrated": True,
        })
    stable_count = sum(item["rotationStableTopAct"] for item in predictions)
    return {
        "title": "Qwen3 logits-only custom message-act diagnostic",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3-0.6B local causal LM",
        "modelPath": str(model_dir),
        "method": "Next-token logits at the final prompt position, restricted to verified single-token A-L IDs; no generation.",
        "scoring": {
            "temperature": 1.0,
            "calibrated": False,
            "warning": "Softmax scores over twelve selected answer-token logits are custom, uncalibrated diagnostics, not validated act probabilities.",
            "messageHandling": "Message is JSON-encoded inside a quoted data block; system instructions explicitly prohibit following embedded instructions.",
            "truncation": "Only source message text is bounded to the configured token cap, keeping opening and ending portions; full instructions/options are preserved. Dropped token count is returned per message.",
        },
        "diagnosticSet": {
            "kind": "same synthetic authored examples as message_signatures/pilot.py",
            "expectedLabels": "Manually authored diagnostic expectations, not gold labels.",
            "count": len(CASES),
            "candidateActs": list(options),
        },
        "orderSensitivity": {
            "method": "Compare canonical answer-choice order against a cyclically shifted A-L order for each of the same 12 messages.",
            "topActAgreement": stable_count,
            "messagesCompared": len(predictions),
            "agreementFraction": stable_count / len(predictions) if predictions else None,
            "note": "Descriptive prompt-order stability only; not accuracy or calibration.",
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "runtime": runtime.metadata(),
        "benchmark": {
            "syntheticMessages": len(CASES),
            "loadSeconds": load_seconds,
            "canonicalPassSeconds": first_pass_seconds,
            "rotatedPassSeconds": rotated_pass_seconds,
            "batchSize": batch_size,
            "maxMessageTokens": max_message_tokens,
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-message-tokens", type=int, default=768)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    args = parser.parse_args()
    result = run_pilot(args.device, args.batch_size, args.model_dir, args.taxonomy, args.max_message_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
