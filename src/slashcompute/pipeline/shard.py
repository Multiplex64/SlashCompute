"""Load only the layers a stage owns.

``mlx_lm.utils.load_model(lazy=True)`` builds the full module tree backed by
lazily-loaded safetensors arrays. We keep references to the modules this stage
needs, drop the rest, and evaluate, so weights for other stages are never read
into memory.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model

from slashcompute.pipeline.model_profile import resolve_model_path


def layer_key(idx: int) -> str:
    # Non-numeric so tree_unflatten keeps a dict rather than building a list.
    return f"L{idx}"


def layer_index(key: str) -> int:
    return int(key[1:])


class ShardModule(nn.Module):
    def __init__(self, layer_start: int, layer_end: int, num_layers: int) -> None:
        super().__init__()
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.num_layers = num_layers
        self.is_first = layer_start == 0
        self.is_last = layer_end == num_layers
        self.layers: dict[str, nn.Module] = {}
        self.embed_tokens: Optional[nn.Embedding] = None
        self.norm: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self.tied_head = False

    def ordered_layers(self):
        return [self.layers[layer_key(i)] for i in range(self.layer_start, self.layer_end)]

    def __call__(self, x: mx.array) -> mx.array:
        """Stage 0 takes token ids; others take hidden states. The last stage
        returns logits; others return hidden states."""
        h = self.embed_tokens(x) if self.is_first else x
        mask = "causal" if h.shape[1] > 1 else None
        for layer in self.ordered_layers():
            h = layer(h, mask, None)
        if not self.is_last:
            return h
        h = self.norm(h)
        if self.tied_head:
            return self.embed_tokens.as_linear(h)
        return self.lm_head(h)


def load_shard(model: str | Path, layer_start: int, layer_end: int,
               num_layers: Optional[int] = None) -> tuple[ShardModule, dict]:
    path = resolve_model_path(str(model))
    full, config = load_model(path, lazy=True)
    inner = full.model
    n = num_layers or len(inner.layers)
    if not (0 <= layer_start < layer_end <= n):
        raise ValueError(f"bad layer range [{layer_start}, {layer_end}) for {n} layers")

    shard = ShardModule(layer_start, layer_end, n)
    shard.layers = {layer_key(i): inner.layers[i] for i in range(layer_start, layer_end)}
    tied = bool(getattr(full.args, "tie_word_embeddings", False)) or not hasattr(full, "lm_head")
    if shard.is_first or (shard.is_last and tied):
        shard.embed_tokens = inner.embed_tokens
    if shard.is_last:
        shard.norm = inner.norm
        shard.tied_head = tied
        if not tied:
            shard.lm_head = full.lm_head

    del full, inner
    gc.collect()
    mx.eval(shard.parameters())
    shard.eval()
    return shard, config
