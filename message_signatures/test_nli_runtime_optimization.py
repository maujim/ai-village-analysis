"""Tokenization-only checks for the optimized NLI preprocessing path.

These focused tests load the local tokenizer but do not load classifier weights
or run inference, so they do not contend with full-corpus model scoring.
"""
from __future__ import annotations

import unittest

try:
    from .nli_runtime import DEFAULT_MODEL_DIR, MAX_LENGTH, _length_order, _prepare_pair
except ImportError:
    from nli_runtime import DEFAULT_MODEL_DIR, MAX_LENGTH, _length_order, _prepare_pair


@unittest.skipUnless(DEFAULT_MODEL_DIR.is_dir(), "local DeBERTa model assets are unavailable")
class PairPreparationParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(DEFAULT_MODEL_DIR), local_files_only=True, trust_remote_code=False
        )

    def _assert_matches_string_tokenizer(self, premise: str, hypothesis: str, max_length: int = MAX_LENGTH) -> None:
        tokenizer = self.tokenizer
        premise_encoding = tokenizer._tokenizer.encode(premise, add_special_tokens=False)
        hypothesis_encoding = tokenizer._tokenizer.encode(hypothesis, add_special_tokens=False)
        optimized, dropped = _prepare_pair(tokenizer, premise_encoding, hypothesis_encoding, max_length)
        baseline = tokenizer(
            premise,
            hypothesis,
            padding=False,
            truncation="only_first",
            max_length=max_length,
        )
        self.assertEqual(optimized["input_ids"], baseline["input_ids"])
        self.assertEqual(optimized["attention_mask"], baseline["attention_mask"])
        if "token_type_ids" in baseline:
            self.assertEqual(optimized["token_type_ids"], baseline["token_type_ids"])
        source_budget = max_length - len(hypothesis_encoding.ids) - tokenizer.num_special_tokens_to_add(pair=True)
        self.assertEqual(dropped, max(0, len(premise_encoding.ids) - source_budget))

    def test_regular_pair_matches_baseline(self) -> None:
        self._assert_matches_string_tokenizer(
            "Please inspect the report and tell me whether the figures match.",
            "The speaker is asking another actor to perform a specified action.",
        )

    def test_unicode_punctuation_and_empty_premise_match_baseline(self) -> None:
        self._assert_matches_string_tokenizer(
            "‘We checked it’—or did we? ✅ café…",
            "The speaker is challenging or correcting a prior proposition.",
        )
        self._assert_matches_string_tokenizer(
            "",
            "The speaker is primarily acknowledging another participant socially.",
        )

    def test_overlength_premise_only_truncation_matches_baseline(self) -> None:
        premise = ("The agent inspected a saved report and recorded the observed result. " * 180).strip()
        self._assert_matches_string_tokenizer(
            premise,
            "The speaker reports that a check was performed and gives its observed result.",
        )

    def test_short_configured_length_preserves_hypothesis(self) -> None:
        self._assert_matches_string_tokenizer(
            "One two three four five six seven eight nine ten eleven twelve.",
            "The speaker is reporting current task state.",
            max_length=32,
        )

    def test_pair_preparation_and_padding_leave_cached_encodings_unchanged(self) -> None:
        tokenizer = self.tokenizer
        backend = tokenizer._tokenizer
        premise = backend.encode("‘Hi, café!’ — check? ✅", add_special_tokens=False)
        hypothesis = backend.encode(
            "The speaker reports a check and its observed result.", add_special_tokens=False
        )
        premise_ids = list(premise.ids)
        hypothesis_ids = list(hypothesis.ids)

        features, _ = _prepare_pair(tokenizer, premise, hypothesis, 32)
        tokenizer.pad([features], padding=True, return_tensors="np")
        self.assertEqual(premise.ids, premise_ids)
        self.assertEqual(hypothesis.ids, hypothesis_ids)

        long_premise = backend.encode("The source says the check was performed. " * 100,
                                      add_special_tokens=False)
        long_ids = list(long_premise.ids)
        _prepare_pair(tokenizer, long_premise, hypothesis, 32)
        self.assertEqual(long_premise.ids, long_ids)


class LengthBucketingTests(unittest.TestCase):
    def test_order_is_stable_and_bucketing_can_be_disabled(self) -> None:
        features = [{"input_ids": list(range(n))} for n in (9, 3, 6, 3)]
        self.assertEqual(_length_order(features), [1, 3, 2, 0])
        self.assertEqual(_length_order(features, enabled=False), [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
