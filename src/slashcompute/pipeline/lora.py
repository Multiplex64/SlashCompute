"""LoRA adapters on a shard, with per-layer deterministic init so any
partitioning of the same job starts from identical adapters."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear

from slashcompute.pipeline.shard import ShardModule, layer_index


def _resolve(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    return module, parts[-1]


def apply_lora(shard: ShardModule, targets: list[str], rank: int, scale: float, seed: int) -> None:
    shard.freeze()
    for key, layer in shard.layers.items():
        mx.random.seed(seed * 100_003 + layer_index(key))
        for target in targets:
            parent, attr = _resolve(layer, target)
            base = getattr(parent, attr)
            setattr(parent, attr, LoRALinear.from_base(base, r=rank, scale=scale))
    mx.eval(shard.trainable_parameters())


def adapter_weights(shard: ShardModule) -> dict[str, mx.array]:
    return dict(tree_flatten(shard.trainable_parameters()))


def _in_range(key: str, start: int, end: int) -> bool:
    # keys look like "layers.L12.self_attn.q_proj.lora_a"
    parts = key.split(".")
    if len(parts) < 2 or parts[0] != "layers" or not parts[1].startswith("L"):
        return False
    return start <= int(parts[1][1:]) < end


def filter_layers(weights: dict[str, mx.array], start: int, end: int) -> dict[str, mx.array]:
    return {k: v for k, v in weights.items() if _in_range(k, start, end)}


def load_adapter_weights(shard: ShardModule, weights: dict[str, mx.array]) -> None:
    mine = filter_layers(weights, shard.layer_start, shard.layer_end)
    expected = set(adapter_weights(shard))
    missing = expected - set(mine)
    if missing:
        raise ValueError(f"checkpoint missing {len(missing)} adapter tensors, e.g. {sorted(missing)[:3]}")
    shard.update(tree_unflatten(list(mine.items())))
    mx.eval(shard.trainable_parameters())
