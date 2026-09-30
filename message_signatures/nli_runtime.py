"""Offline DeBERTa NLI scorer for independent taxonomy-act entailment scores.

Each source message is paired with each taxonomy hypothesis. Scores are the
model's entailment class probability for that pair; they are independent across
acts and are not normalized into a forced-choice distribution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "deberta-base-zeroshot"
DEFAULT_TAXONOMY = Path(__file__).resolve().with_name("taxonomy.json")
MAX_LENGTH = 512


def _load_hypotheses(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    acts = data.get("acts")
    if not isinstance(acts, list) or len(acts) != 12:
        raise ValueError("taxonomy must define exactly 12 acts")
    hypotheses = {str(a["id"]): str(a.get("hypothesis") or "").strip() for a in acts}
    if len(hypotheses) != 12 or any(not h for h in hypotheses.values()):
        raise ValueError("taxonomy act IDs must be unique and hypotheses nonempty")
    return hypotheses


def _label_key(label: str) -> str:
    return "_".join(label.strip().casefold().replace("-", " ").split())


class NLIRuntime:
    """Score premise/hypothesis entailment using local-only model weights."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = MAX_LENGTH,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 8 or max_length > MAX_LENGTH:
            raise ValueError(f"max_length must be between 8 and {MAX_LENGTH}")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Local NLI model directory not found: {self.model_dir}")
        self.taxonomy_path = Path(taxonomy_path).expanduser().resolve()
        self.hypotheses = _load_hypotheses(self.taxonomy_path)
        self.act_ids = list(self.hypotheses)
        self.batch_size = batch_size
        self.max_length = max_length

        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps"):
            raise ValueError("device must be 'auto', 'cpu', or 'mps'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but no MPS backend is available")
        self.device = torch.device(device)
        self.dtype = torch.float16 if device == "mps" else torch.float32

        config = AutoConfig.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        id2label = {int(k): str(v) for k, v in config.id2label.items()}
        label2id = {_label_key(str(k)): int(v) for k, v in config.label2id.items()}
        normalized = {_label_key(v): k for k, v in id2label.items()}
        entailment_ids = [idx for idx, name in id2label.items() if _label_key(name) == "entailment"]
        if len(entailment_ids) != 1:
            raise ValueError(f"Expected exactly one config label named entailment; observed id2label={id2label}")
        entailment_id = entailment_ids[0]
        if label2id.get("entailment") != entailment_id:
            raise ValueError(f"Config id2label/label2id disagree about entailment: {id2label}, {config.label2id}")
        if set(id2label) != set(range(int(config.num_labels))):
            raise ValueError(f"Config label indices are incomplete: {id2label}")
        self.config_labels = id2label
        self.entailment_id = entailment_id

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False,
            config=config, torch_dtype=self.dtype,
        ).eval().to(self.device)
        if int(self.model.config.num_labels) != len(self.config_labels):
            raise ValueError("Loaded classifier head does not match the observed config labels")

    def predict(
        self,
        texts: Sequence[str] | Iterable[str],
        options: Mapping[str, str],
        question: str = "Which communicative-act hypothesis is supported by the source message?",
    ) -> list[dict[str, Any]]:
        if isinstance(texts, str):
            texts = [texts]
        messages = [str(x) for x in texts]
        if not messages:
            return []
        if not isinstance(options, Mapping):
            raise TypeError("options must map taxonomy act IDs to descriptions")
        normalized_options = {str(k): str(v) for k, v in options.items()}
        if set(normalized_options) != set(self.act_ids) or len(normalized_options) != 12:
            raise ValueError("options must contain each taxonomy act ID exactly once")
        # The taxonomy's hypothesis field is the NLI hypothesis. The options
        # mapping is accepted for API compatibility and checked for exact IDs.
        question = str(question)
        if not question.strip():
            raise ValueError("question must be nonempty")

        pairs: list[tuple[int, str, str]] = []
        for mi, message in enumerate(messages):
            # Canonicalize execution order so a caller's mapping insertion order
            # cannot change batch composition or numeric padding effects.
            for act_id in self.act_ids:
                pairs.append((mi, act_id, message))

        score_rows: list[tuple[int, str, float]] = []
        trunc_by_message: list[dict[str, dict[str, Any]]] = [dict() for _ in messages]
        special_count = int(self.tokenizer.num_special_tokens_to_add(pair=True))
        hypothesis_ids = {
            act_id: self.tokenizer.encode(hypothesis, add_special_tokens=False)
            for act_id, hypothesis in self.hypotheses.items()
        }
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            premises = [item[2] for item in batch]
            hypotheses = [self.hypotheses[item[1]] for item in batch]
            encoded = self.tokenizer(
                premises, hypotheses,
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # With only_first truncation, the hypothesis and pair-special tokens
            # are retained. This computes the exact number of source tokens
            # omitted by the configured pair budget.
            for mi, act_id, message in batch:
                source_len = len(self.tokenizer.encode(message, add_special_tokens=False))
                available = self.max_length - len(hypothesis_ids[act_id]) - special_count
                dropped = max(0, source_len - available)
                trunc_by_message[mi][act_id] = {
                    "message": dropped > 0,
                    "message_tokens": source_len,
                    "message_tokens_dropped": dropped,
                    "hypothesis": False,
                }
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.float()
                probs = self.torch.softmax(logits, dim=-1)[:, self.entailment_id]
            score_rows.extend(
                (mi, act_id, float(prob))
                for (mi, act_id, _), prob in zip(batch, probs.cpu().tolist())
            )

        by_message: list[dict[str, float]] = [dict() for _ in messages]
        for mi, act_id, score in score_rows:
            by_message[mi][act_id] = score
        results = []
        for scores, truncation in zip(by_message, trunc_by_message):
            truncated = any(item["message"] for item in truncation.values())
            results.append({
                "probabilities": scores,
                "score_semantics": "independent_entailment",
                "probabilities_sum_to_one": False,
                "truncated": truncated,
                "truncation": {
                    "message": truncated,
                    "message_tokens_dropped_max": max((x["message_tokens_dropped"] for x in truncation.values()), default=0),
                    "per_hypothesis": truncation,
                    "hypothesis": False,
                    "question": False,
                },
                "device": str(self.device),
                "calibrated": False,
                "label_mapping": self.config_labels,
                "entailment_label": self.config_labels[self.entailment_id],
            })
        return results

    def metadata(self) -> dict[str, Any]:
        from importlib.metadata import version
        try:
            transformers_version = version("transformers")
        except Exception:
            transformers_version = "unknown"
        return {
            "model_dir": str(self.model_dir),
            "model_type": getattr(self.model.config, "model_type", None),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "label_mapping": self.config_labels,
            "entailment_label": self.config_labels[self.entailment_id],
            "score_semantics": "independent_entailment",
            "calibrated": False,
            "transformers_version": transformers_version,
            "torch_version": self.torch.__version__,
            "local_files_only": True,
            "trust_remote_code": False,
        }
