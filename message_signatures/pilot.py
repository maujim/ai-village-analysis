"""Small synthetic diagnostic for custom-act Nano-Jev option scoring.

This intentionally does not label the transcript or estimate accuracy. The
expected acts below are human-authored test prompts, not a gold set.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .nano_runtime import NanoRuntime
except ImportError:  # Direct script execution from this directory.
    from nano_runtime import NanoRuntime


SIGNATURES = Path(__file__).resolve().parent
TAXONOMY = SIGNATURES / "taxonomy.json"
DEFAULT_OUTPUT = SIGNATURES / "pilot.json"
QUESTION = "Which primary communicative act does this message perform?"

# Deliberately short and unambiguous examples spanning the v1 act taxonomy.
# These expected values only help inspect obvious confusions; they are not a
# validation set, and no accuracy score is computed.
CASES = [
    {"id": "request-01", "text": "Mira, please review the README and tell me whether the setup steps are complete.", "expectedAct": "instruction_request"},
    {"id": "question-01", "text": "Did you inspect the latest test output?", "expectedAct": "question"},
    {"id": "proposal-01", "text": "We could run the same check on the second branch before merging.", "expectedAct": "proposal_plan"},
    {"id": "commitment-01", "text": "I will rerun the unit tests and post the failing case here.", "expectedAct": "commitment_assignment"},
    {"id": "offer-01", "text": "I can help compare the two drafts if that would be useful.", "expectedAct": "offer_help"},
    {"id": "status-01", "text": "The export is still blocked because the archive has not finished downloading.", "expectedAct": "status_report"},
    {"id": "check-01", "text": "I ran the parser on the sample file; it returned 12 rows and no errors.", "expectedAct": "check_report"},
    {"id": "assertion-01", "text": "The configuration file lists three supported event types.", "expectedAct": "assertion"},
    {"id": "endorsement-01", "text": "I agree with your proposal to keep the source links in the evidence bundle.", "expectedAct": "endorsement"},
    {"id": "qualification-01", "text": "This appears to work for the sample export, although we have not checked the full archive.", "expectedAct": "qualification"},
    {"id": "challenge-01", "text": "That count is incorrect: the log shows 14 entries, not 12.", "expectedAct": "challenge_correction"},
    {"id": "acknowledgment-01", "text": "Thanks for checking that and sharing the result.", "expectedAct": "acknowledgment"},
]


def taxonomy_options(data: dict) -> tuple[list[str], dict[str, str]]:
    acts = data.get("acts")
    if not isinstance(acts, list) or not acts:
        raise ValueError("taxonomy.json must contain a nonempty 'acts' list")
    options: dict[str, str] = {}
    ids: list[str] = []
    for act in acts:
        act_id = str(act["id"])
        definition = str(act.get("definition") or "")
        hypothesis = str(act.get("hypothesis") or definition)
        cues = "; ".join(str(x) for x in act.get("positiveCues", []))
        exclusions = "; ".join(str(x) for x in act.get("exclude", []))
        option = f"{hypothesis} Definition: {definition} Positive cues: {cues} Exclude: {exclusions}"
        ids.append(act_id)
        options[act_id] = option
    return ids, options


def run_device(device: str, options: dict[str, str], act_ids: list[str], max_length: int, batch_size: int) -> dict:
    load_start = time.perf_counter()
    runtime = NanoRuntime(device=device, max_length=max_length, batch_size=batch_size)
    load_seconds = time.perf_counter() - load_start
    texts = [case["text"] for case in CASES]
    # First pass includes lazy backend/kernel setup; second pass is the steady
    # repeat measurement for this tiny, fixed diagnostic batch.
    first_start = time.perf_counter()
    first = runtime.predict(texts, options, QUESTION)
    first_pass = time.perf_counter() - first_start
    second_start = time.perf_counter()
    scored = runtime.predict(texts, options, QUESTION)
    second_pass = time.perf_counter() - second_start

    predictions = []
    for case, output in zip(CASES, scored):
        ordered = sorted(output["probabilities"].items(), key=lambda item: item[1], reverse=True)
        decoded = []
        for act_id, probability in ordered:
            decoded.append({"act": act_id, "probability": probability})
        predictions.append({
            "id": case["id"],
            "message": case["text"],
            "diagnosticExpectedAct_notGold": case["expectedAct"],
            "topAct": decoded[0]["act"],
            "rankedActs": decoded,
            "truncated": output["truncated"],
            "truncation": output["truncation"],
            "truncationByOption": output["truncation_by_option"],
            "probabilitiesUncalibrated": True,
        })
    metadata = runtime.metadata()
    return {
        "runtime": metadata,
        "benchmark": {
            "syntheticMessages": len(CASES),
            "candidateActs": len(act_ids),
            "crossEncoderPairs": len(CASES) * len(act_ids),
            "loadSeconds": load_seconds,
            "firstPassSeconds": first_pass,
            "steadySecondPassSeconds": second_pass,
            "steadyMillisecondsPerMessage": second_pass * 1000 / len(CASES),
        },
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-length", type=int, choices=(384, 512), default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--devices", nargs="+", choices=("cpu", "mps"), default=("cpu", "mps"))
    args = parser.parse_args()

    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    act_ids, options = taxonomy_options(taxonomy)
    import torch
    cpu_threads = min(8, torch.get_num_threads())
    torch.set_num_threads(cpu_threads)

    runs: dict[str, dict] = {}
    for device in args.devices:
        try:
            runs[device] = run_device(device, options, act_ids, args.max_length, args.batch_size)
        except Exception as exc:
            runs[device] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "note": "No fallback or remote model download was attempted.",
            }
    output = {
        "title": "Nano-Jev custom message-act synthetic pilot",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": "Nano-Jev 1.0 local 33.4M-parameter sequence classifier",
        "modelPath": "models/nano-jev",
        "modelCard": "models/nano-jev/README.md",
        "inputFormat": "question: <question> option: <taxonomy act> paired with the message as sequence two",
        "scoring": {
            "method": "One scalar logit per paired message-option; softmax across act options at temperature 1.0.",
            "temperature": 1.0,
            "calibrated": False,
            "probabilityWarning": "These custom taxonomy options differ from Nano-Jev's trained relevance/sufficiency/groundedness decisions. Scores are uncalibrated advisory rankings, not validated probabilities or act labels.",
            "truncation": "The paired message is tokenized with only_second truncation at the configured max length; an overlong question is shortened only as needed to preserve the option text. Both flags are reported per message and option.",
        },
        "diagnosticSet": {
            "kind": "synthetic authored examples",
            "expectedLabels": "Manually authored diagnostic expectations, not gold labels; no accuracy metric is computed.",
            "count": len(CASES),
            "taxonomyVersion": taxonomy.get("version"),
            "acts": act_ids,
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpuThreads": cpu_threads},
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("benchmark", v) for k, v in runs.items()}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
