"""Offline DeBERTa NLI scorer for independent taxonomy-act entailment scores.

Each source message is paired with each taxonomy hypothesis. Scores are the
model's entailment class probability for that pair; they are independent across
acts and are not normalized into a forced-choice distribution.
"""
from __future__ import annotations

import json
from copy import deepcopy
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


def _prepare_pair(tokenizer: Any, premise_encoding: Any, hypothesis_encoding: Any, max_length: int):
    """Post-process cached Rust-tokenizer encodings with only-first truncation."""
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or not hasattr(backend, "post_process"):
        raise TypeError("NLI optimization requires a tokenizer backend with pair post-processing")
    special_count = int(tokenizer.num_special_tokens_to_add(pair=True))
    premise_budget = max(0, max_length - len(hypothesis_encoding.ids) - special_count)
    premise_count = len(premise_encoding.ids)
    dropped = max(0, premise_count - premise_budget)
    premise = deepcopy(premise_encoding)
    if dropped:
        premise.truncate(premise_budget, stride=0, direction=tokenizer.truncation_side)
    pair = backend.post_process(premise, hypothesis_encoding, add_special_tokens=True)
    return {
        "input_ids": pair.ids,
        "attention_mask": pair.attention_mask,
        "token_type_ids": pair.type_ids,
    }, dropped


def _length_order(encoded_pairs: Sequence[Mapping[str, Any]], enabled: bool = True) -> list[int]:
    """Return stable pair indices, grouping similar sequence lengths together."""
    indices = list(range(len(encoded_pairs)))
    if enabled:
        indices.sort(key=lambda i: (len(encoded_pairs[i]["input_ids"]), i))
    return indices


class NLIRuntime:
    """Score premise/hypothesis entailment using local-only model weights."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = MAX_LENGTH,
        taxonomy_path: str | Path = DEFAULT_TAXONOMY,
        pair_batch_size: int | None = None,
        length_bucketing: bool = True,
        backend: str = "torch",
    ) -> None:
        effective_pair_batch_size = batch_size if pair_batch_size is None else pair_batch_size
        if effective_pair_batch_size < 1:
            raise ValueError("pair_batch_size must be positive")
        if max_length < 8 or max_length > MAX_LENGTH:
            raise ValueError(f"max_length must be between 8 and {MAX_LENGTH}")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Local NLI model directory not found: {self.model_dir}")
        self.taxonomy_path = Path(taxonomy_path).expanduser().resolve()
        self.hypotheses = _load_hypotheses(self.taxonomy_path)
        self.act_ids = list(self.hypotheses)
        self.batch_size = effective_pair_batch_size  # Backward-compatible alias.
        self.pair_batch_size = effective_pair_batch_size
        self.length_bucketing = bool(length_bucketing)
        self.max_length = max_length
        if backend not in ("torch", "mlx"):
            raise ValueError("backend must be torch or mlx")
        self.backend = backend

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
        if backend == "mlx" and device != "mps":
            raise ValueError("The MLX backend requires the local Apple GPU")

        config = AutoConfig.from_pretrained(str(self.model_dir), local_files_only=True, trust_remote_code=False)
        self.config = config
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
        self.tokenizer_backend = getattr(self.tokenizer, "_tokenizer", None)
        if self.tokenizer_backend is None or not hasattr(self.tokenizer_backend, "post_process"):
            raise TypeError("The local DeBERTa tokenizer must expose its Rust pair post-processor")
        self.hypothesis_encodings = {
            act_id: self.tokenizer_backend.encode(hypothesis, add_special_tokens=False)
            for act_id, hypothesis in self.hypotheses.items()
        }
        if backend == "mlx":
            import mlx.core as mx
            from message_signatures.mlx_deberta import MLXDeberta
            self.mx = mx
            mx.set_cache_limit(512 * 1024 * 1024)
            self.model = MLXDeberta(self.model_dir)
            self.parameter_count = self.model.parameter_count
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_dir), local_files_only=True, trust_remote_code=False,
                config=config, torch_dtype=self.dtype,
            ).eval().to(self.device)
            self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if int(config.num_labels) != len(self.config_labels):
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

        # Tokenize each distinct premise once per predict call and reuse the
        # fixed taxonomy hypothesis IDs across every call.
        premise_encodings_by_text = {
            message: self.tokenizer_backend.encode(message, add_special_tokens=False)
            for message in dict.fromkeys(messages)
        }
        pairs: list[dict[str, Any]] = []
        trunc_by_message: list[dict[str, dict[str, Any]]] = [dict() for _ in messages]
        for mi, message in enumerate(messages):
            # Canonicalize candidate order so caller mapping order cannot change
            # padding or batching. Results are scattered back by input index.
            source_encoding = premise_encodings_by_text[message]
            for act_id in self.act_ids:
                prepared, dropped = _prepare_pair(
                    self.tokenizer, source_encoding, self.hypothesis_encodings[act_id], self.max_length
                )
                trunc_by_message[mi][act_id] = {
                    "message": dropped > 0,
                    "message_tokens": len(source_encoding.ids),
                    "message_tokens_dropped": dropped,
                    "hypothesis": False,
                }
                pairs.append({"message_index": mi, "act_id": act_id, "features": prepared})

        order = _length_order([pair["features"] for pair in pairs], self.length_bucketing)
        scores_by_pair = [0.0] * len(pairs)
        for start in range(0, len(order), self.pair_batch_size):
            batch_indices = order[start:start + self.pair_batch_size]
            features = [pairs[index]["features"] for index in batch_indices]
            # Padding is performed only after similar-length pairs are grouped.
            if self.backend == "mlx":
                encoded = self.tokenizer.pad(features, padding=True, return_tensors="np")
                logits = self.model(self.mx.array(encoded["input_ids"]), self.mx.array(encoded["attention_mask"]))
                probs = self.mx.softmax(logits.astype(self.mx.float32), axis=-1)[:, self.entailment_id]
                values = probs.tolist()
            else:
                encoded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with self.torch.inference_mode():
                    logits = self.model(**encoded).logits.float()
                    probs = self.torch.softmax(logits, dim=-1)[:, self.entailment_id]
                values = probs.cpu().tolist()
            for pair_index, probability in zip(batch_indices, values):
                scores_by_pair[pair_index] = float(probability)

        by_message: list[dict[str, float]] = [dict() for _ in messages]
        for pair, score in zip(pairs, scores_by_pair):
            by_message[pair["message_index"]][pair["act_id"]] = score
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
                "backend": self.backend,
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
            "model_type": getattr(self.config, "model_type", None),
            "backend": self.backend,
            "backend_version": version("mlx") if self.backend == "mlx" else self.torch.__version__,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "batch_size": self.batch_size,
            "pair_batch_size": self.pair_batch_size,
            "length_bucketing": self.length_bucketing,
            "max_length": self.max_length,
            "label_mapping": self.config_labels,
            "entailment_label": self.config_labels[self.entailment_id],
            "score_semantics": "independent_entailment",
            "calibrated": False,
            "transformers_version": transformers_version,
            "torch_version": self.torch.__version__,
            "local_files_only": True,
            "trust_remote_code": False,
            "tokenization_strategy": "cache distinct premise tokens and taxonomy hypothesis tokens; sort pairs by encoded length before padded batches",
        }
