"""Offline Nano-Jev scoring for custom message-act options.

The model was trained for RAG decisions, not conversational act labels. These
scores are uncalibrated diagnostics; they are not validated act probabilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "nano-jev"


class NanoRuntime:
    """Load the local Nano-Jev sequence classifier once and score option sets.

    Inputs follow the model card's paired cross-encoder format: the first
    sequence is ``question: <question> option: <option>`` and the second is the
    message. Softmax is applied across options for each message at T=1.0. Because
    these are custom labels outside the model's trained decisions, probabilities
    are explicitly uncalibrated.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        if max_length not in (384, 512):
            raise ValueError("max_length must be 384 or 512")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {self.model_dir}")
        self.max_length = max_length
        self.batch_size = batch_size

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps"):
            raise ValueError("device must be 'auto', 'cpu', or 'mps'")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but this PyTorch build/device has no available MPS backend")
        self.device = torch.device(device)

        # Both loads are local-only; custom model code is never executed.
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_dir), local_files_only=True, trust_remote_code=False
        ).eval().to(self.device)
        self.temperature = 1.0
        self.calibrated = False

    def _first_sequence(self, question: str, option: str) -> tuple[str, bool]:
        """Keep the option text intact; trim only an overlong question prefix."""
        tokenizer = self.tokenizer
        first = f"question: {question} option: {option}"
        pair_specials = tokenizer.num_special_tokens_to_add(pair=True)
        # Reserve one message token: fast BERT tokenizers can reject
        # only_second truncation if the first sequence leaves room for none.
        first_budget = self.max_length - pair_specials - 1
        first_ids = tokenizer(first, add_special_tokens=False)["input_ids"]
        if len(first_ids) <= first_budget:
            return first, False

        marker_ids = tokenizer("question: ", add_special_tokens=False)["input_ids"]
        option_ids = tokenizer(f" option: {option}", add_special_tokens=False)["input_ids"]
        q_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
        q_budget = max(0, first_budget - len(marker_ids) - len(option_ids))
        shortened = tokenizer.decode(q_ids[:q_budget], skip_special_tokens=True)
        candidate = f"question: {shortened} option: {option}"
        # Tokenizer boundary merges can shift the estimate; shorten until the
        # actual first sequence fits, preserving the complete option suffix.
        while q_budget and len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) > first_budget:
            q_budget -= 1
            shortened = tokenizer.decode(q_ids[:q_budget], skip_special_tokens=True)
            candidate = f"question: {shortened} option: {option}"
        if len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) > first_budget:
            raise ValueError("An option is too long to fit the selected model context window")
        return candidate, True

    def predict(
        self,
        texts: Sequence[str] | Iterable[str],
        options: Sequence[str] | Mapping[str, str],
        question: str,
    ) -> list[dict[str, Any]]:
        """Return option probabilities and real truncation flags per message.

        ``truncation.message`` means the paired second sequence was shortened
        by the tokenizer's actual ``only_second`` truncation. A long question
        may separately be shortened to preserve the option in the first
        sequence; that is reported as ``truncation.question``. Pass a mapping
        from stable option IDs to descriptions to get ID-keyed probabilities.
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        if isinstance(options, Mapping):
            option_keys = [str(key) for key in options]
            option_texts = [f"{key}: {value}" for key, value in options.items()]
        else:
            option_keys = [str(o) for o in options]
            option_texts = option_keys.copy()
        question = str(question)
        if not texts:
            return []
        if not option_keys or any(not o.strip() for o in option_texts):
            raise ValueError("options must contain at least one nonempty label")
        if len(set(option_keys)) != len(option_keys):
            raise ValueError("options must be unique so probability keys remain unambiguous")
        if not question.strip():
            raise ValueError("question must be nonempty")

        first_sequences: list[str] = []
        second_sequences: list[str] = []
        question_truncated: list[bool] = []
        message_truncated: list[bool] = []
        per_option_truncation: list[dict[str, dict[str, bool]]] = []
        for message in texts:
            msg_ids = self.tokenizer(message, add_special_tokens=False)["input_ids"]
            per_option = []
            for option in option_texts:
                first, q_was_truncated = self._first_sequence(question, option)
                first_ids = self.tokenizer(first, add_special_tokens=False)["input_ids"]
                raw_pair_length = (
                    len(first_ids) + len(msg_ids)
                    + self.tokenizer.num_special_tokens_to_add(pair=True)
                )
                per_option.append((first, q_was_truncated, raw_pair_length > self.max_length))
            first_sequences.extend(item[0] for item in per_option)
            second_sequences.extend([message] * len(option_texts))
            question_truncated.append(any(item[1] for item in per_option))
            message_truncated.append(any(item[2] for item in per_option))
            per_option_truncation.append({
                option: {"question": item[1], "message": item[2]}
                for option, item in zip(option_keys, per_option)
            })

        probability_rows: list[list[float]] = []
        pair_cursor = 0
        for start in range(0, len(texts), self.batch_size):
            stop = min(len(texts), start + self.batch_size)
            count = (stop - start) * len(option_keys)
            first_batch = first_sequences[pair_cursor:pair_cursor + count]
            second_batch = second_sequences[pair_cursor:pair_cursor + count]
            pair_cursor += count
            encoded = self.tokenizer(
                first_batch,
                second_batch,
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(stop - start, len(option_keys))
                probs = self.torch.softmax(logits / self.temperature, dim=-1).cpu().tolist()
            probability_rows.extend(probs)

        result: list[dict[str, Any]] = []
        for i, (message, row) in enumerate(zip(texts, probability_rows)):
            probabilities = {option: float(prob) for option, prob in zip(option_keys, row)}
            result.append({
                "probabilities": probabilities,
                "top_option": max(probabilities, key=probabilities.get),
                "truncated": question_truncated[i] or message_truncated[i],
                "truncation": {
                    "question": question_truncated[i],
                    "message": message_truncated[i],
                },
                "truncation_by_option": per_option_truncation[i],
                "device": str(self.device),
                "temperature": self.temperature,
                "calibrated": False,
                "calibration_note": "Custom message-act options are outside the trained Nano-Jev decisions; softmax scores are uncalibrated advisory rankings.",
            })
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "device": str(self.device),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "calibrated": self.calibrated,
            "transformers_version": self._version("transformers"),
            "torch_version": self.torch.__version__,
        }

    @staticmethod
    def _version(package: str) -> str:
        try:
            from importlib.metadata import version
            return version(package)
        except Exception:
            return "unknown"
