"""A tiny randomly-initialised Qwen2 model plus a pre-tokenised dataset, so
tests and offline demos run without downloading anything."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen2

TINY_CONFIG = {
    "model_type": "qwen2",
    "hidden_size": 64,
    "num_hidden_layers": 6,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 256,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}


def make_tiny_model(path: str | Path, seed: int = 0, **overrides) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    cfg = TINY_CONFIG | overrides
    mx.random.seed(seed)
    model = qwen2.Model(qwen2.ModelArgs.from_dict(cfg))
    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    return path


def make_tiny_dataset(path: str | Path, n: int = 64, vocab: int = 256, min_len: int = 12,
                      max_len: int = 32, seed: int = 0) -> Path:
    """Learnable sequences: each row is an arithmetic progression mod vocab."""
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(n):
        length = int(rng.integers(min_len, max_len + 1))
        start, stride = int(rng.integers(0, vocab)), int(rng.integers(1, 5))
        toks = [(start + i * stride) % vocab for i in range(length)]
        lines.append(json.dumps({"tokens": toks}))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n")
    return path
