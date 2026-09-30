"""Inference-only MLX forward for the pinned DeBERTa-v3-base NLI checkpoint.

This intentionally supports only the local 12-layer checkpoint configuration.
It returns the original two classifier logits; tokenization and NLI probability
semantics remain the responsibility of the caller.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    from safetensors import safe_open
except ImportError as exc:  # pragma: no cover - depends on optional local runtime
    raise ImportError("MLX DeBERTa requires mlx and safetensors") from exc


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "deberta-base-zeroshot"
WEIGHTS_NAME = "model.safetensors"
_SUPPORTED = {
    "model_type": "deberta-v2",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 512,
    "max_relative_positions": -1,
    "position_buckets": 256,
    "relative_attention": True,
    "position_biased_input": False,
    "type_vocab_size": 0,
    "share_att_key": True,
    "pos_att_type": ["p2c", "c2p"],
    "norm_rel_ebd": "layer_norm",
    "hidden_act": "gelu",
    "pooler_hidden_act": "gelu",
}


def _check_config(config: dict[str, Any]) -> None:
    for key, expected in _SUPPORTED.items():
        observed = config.get(key)
        if observed != expected:
            raise ValueError(f"Unsupported DeBERTa config {key}={observed!r}; expected {expected!r}")
    if config.get("conv_kernel_size") not in (None, 0):
        raise ValueError("Convolutional DeBERTa checkpoints are not supported")
    if config.get("embedding_size", config["hidden_size"]) != config["hidden_size"]:
        raise ValueError("Projected embedding sizes are not supported")
    if config.get("layer_norm_eps") != 1e-7:
        raise ValueError("Only the pinned checkpoint's layer norm epsilon is supported")
    labels = config.get("id2label", {})
    if labels and {str(k): str(v) for k, v in labels.items()} != {
        "0": "entailment", "1": "not_entailment"
    }:
        raise ValueError("Checkpoint classifier labels do not match the expected NLI head")


def _relative_positions(length: int, buckets: int, max_position: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror transformers.models.deberta_v2.make_log_bucket_position."""
    positions = np.arange(length, dtype=np.int64)
    relative = positions[:, None] - positions[None, :]
    mid = buckets // 2
    sign = np.sign(relative)
    absolute = np.where((relative < mid) & (relative > -mid), mid - 1, np.abs(relative))
    safe_absolute = np.maximum(absolute, 1)
    log_position = np.ceil(
        np.log(safe_absolute / mid) / np.log((max_position - 1) / mid) * (mid - 1)
    ) + mid
    bucketed = np.where(absolute <= mid, relative, log_position * sign).astype(np.int64)
    span = buckets
    c2p = np.clip(bucketed + span, 0, span * 2 - 1).astype(np.int32)
    # The p2c table is indexed by key position then query position before
    # the transposed contribution is added to the normal query/key matrix.
    p2c = np.clip(-bucketed + span, 0, span * 2 - 1).astype(np.int32)
    return c2p, p2c


class MLXDeberta:
    """Run the pinned 2-label DeBERTa NLI classifier using MLX fp16 weights."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir).expanduser().resolve()
        config_path = self.model_dir / "config.json"
        weights_path = self.model_dir / WEIGHTS_NAME
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(f"Expected local config.json and {WEIGHTS_NAME} in {self.model_dir}")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        _check_config(self.config)
        self.num_labels = 2
        self.parameter_count = 0
        self.weights: dict[str, Any] = {}
        with safe_open(str(weights_path), framework="np") as checkpoint:
            for name in checkpoint.keys():
                value = checkpoint.get_tensor(name)
                if value.dtype != np.float16:
                    raise ValueError(f"Expected fp16 checkpoint tensor {name}, observed {value.dtype}")
                self.parameter_count += int(value.size)
                self.weights[name] = mx.array(value)

        self.hidden_size = int(self.config["hidden_size"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.head_size = self.hidden_size // self.num_heads
        self.layer_count = int(self.config["num_hidden_layers"])
        self.layer_norm_eps = float(self.config["layer_norm_eps"])
        self.relative_span = int(self.config["position_buckets"])
        self.max_relative_positions = int(self.config["max_position_embeddings"])
        self._relative_key: list[Any] = []
        self._relative_query: list[Any] = []
        self._prepare_relative_projections()
        self._compiled_forward = mx.compile(self._forward)

    def _linear(self, x, prefix: str):
        weight = self.weights[prefix + ".weight"]
        result = mx.matmul(x, mx.swapaxes(weight, -1, -2))
        bias = self.weights.get(prefix + ".bias")
        return result if bias is None else result + bias

    def _layer_norm(self, x, prefix: str):
        return mx.fast.layer_norm(x, self.weights[prefix + ".weight"],
                                  self.weights[prefix + ".bias"], self.layer_norm_eps)

    @staticmethod
    def _gelu(x):
        # Exact erf GELU, matching transformers' ACT2FN["gelu"].
        return 0.5 * x * (1.0 + mx.erf(x / 1.4142135623730951))

    def _prepare_relative_projections(self) -> None:
        rel = self.weights["deberta.encoder.rel_embeddings.weight"]
        rel = self._layer_norm(rel, "deberta.encoder.LayerNorm")
        for index in range(self.layer_count):
            prefix = f"deberta.encoder.layer.{index}.attention.self"
            key = self._linear(rel, prefix + ".key_proj")
            query = self._linear(rel, prefix + ".query_proj")
            self._relative_key.append(
                mx.transpose(mx.reshape(key, (2 * self.relative_span, self.num_heads, self.head_size)), (1, 0, 2))
            )
            self._relative_query.append(
                mx.transpose(mx.reshape(query, (2 * self.relative_span, self.num_heads, self.head_size)), (1, 0, 2))
            )
        # Force the batch-invariant position projections once, not once per batch.
        mx.eval(*self._relative_key, *self._relative_query)

    def _attention(self, hidden, mask, index: int, c2p_index, p2c_index):
        batch, length, _ = hidden.shape
        prefix = f"deberta.encoder.layer.{index}.attention.self"

        def heads(x):
            return mx.transpose(
                mx.reshape(x, (batch, length, self.num_heads, self.head_size)), (0, 2, 1, 3)
            )

        query = heads(self._linear(hidden, prefix + ".query_proj"))
        key = heads(self._linear(hidden, prefix + ".key_proj"))
        value = heads(self._linear(hidden, prefix + ".value_proj"))
        scale = (self.head_size * 3.0) ** 0.5  # content + c2p + p2c
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / scale

        c2p_logits = mx.matmul(query, mx.transpose(self._relative_key[index], (0, 2, 1)))
        c2p_idx = mx.broadcast_to(c2p_index[None, None, :, :], (batch, self.num_heads, length, length))
        c2p_scores = mx.take_along_axis(c2p_logits, c2p_idx, axis=-1)
        scores = scores + c2p_scores / scale

        p2c_logits = mx.matmul(key, mx.transpose(self._relative_query[index], (0, 2, 1)))
        p2c_idx = mx.broadcast_to(p2c_index[None, None, :, :], (batch, self.num_heads, length, length))
        p2c_scores = mx.take_along_axis(p2c_logits, p2c_idx, axis=-1)
        scores = scores + mx.transpose(p2c_scores, (0, 1, 3, 2)) / scale

        pair_mask = mx.expand_dims(mask, 1) & mx.expand_dims(mask, 2)
        pair_mask = mx.expand_dims(pair_mask, 1)
        scores = mx.where(pair_mask, scores, -65504.0)
        probabilities = mx.softmax(scores, axis=-1)
        context = mx.matmul(probabilities, value)
        context = mx.transpose(context, (0, 2, 1, 3)).reshape((batch, length, self.hidden_size))

        output_prefix = f"deberta.encoder.layer.{index}.attention.output"
        projected = self._linear(context, output_prefix + ".dense")
        attended = self._layer_norm(projected + hidden, output_prefix + ".LayerNorm")
        layer_prefix = f"deberta.encoder.layer.{index}"
        intermediate = self._gelu(self._linear(attended, layer_prefix + ".intermediate.dense"))
        output = self._linear(intermediate, layer_prefix + ".output.dense")
        return self._layer_norm(output + attended, layer_prefix + ".output.LayerNorm")

    def __call__(self, input_ids, attention_mask):
        """Return classifier logits for NumPy-compatible token ID/mask batches."""
        ids = mx.array(np.asarray(input_ids, dtype=np.int32))
        mask = mx.array(np.asarray(attention_mask, dtype=np.bool_))
        if len(ids.shape) != 2 or ids.shape != mask.shape:
            raise ValueError("input_ids and attention_mask must be matching [batch, sequence] arrays")
        if ids.shape[1] > self.max_relative_positions:
            raise ValueError("input sequence exceeds the pinned 512-token maximum")
        batch, length = ids.shape
        c2p_np, p2c_np = _relative_positions(length, self.relative_span, self.max_relative_positions)
        c2p_index = mx.array(c2p_np)
        p2c_index = mx.array(p2c_np)
        return self._compiled_forward(ids, mask, c2p_index, p2c_index)

    def _forward(self, ids, mask, c2p_index, p2c_index):
        hidden = mx.take(self.weights["deberta.embeddings.word_embeddings.weight"], ids, axis=0)
        hidden = self._layer_norm(hidden, "deberta.embeddings.LayerNorm")
        hidden = hidden * mx.expand_dims(mask, -1).astype(hidden.dtype)
        for index in range(self.layer_count):
            hidden = self._attention(hidden, mask, index, c2p_index, p2c_index)

        pooled = hidden[:, 0, :]
        pooled = self._gelu(self._linear(pooled, "pooler.dense"))
        return self._linear(pooled, "classifier")
