"""Canary and sampled-replay work the agent runs on coordinator request."""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from slashcompute.common.canary import run_canary_mlx
from slashcompute.common.protocol import VerifyRequest
from slashcompute.pipeline.lora import apply_lora, load_adapter_weights
from slashcompute.pipeline.shard import load_shard
from slashcompute.pipeline.stage import token_losses


def run_canary(seed: int, size: int) -> dict[str, float]:
    return run_canary_mlx(seed, size)


def run_replay(req: VerifyRequest, bundle: Path, dest: Path) -> dict[str, float]:
    """Recompute a stage forward from a saved bundle. Returns wall/busy stats."""
    tensors = mx.load(str(bundle))
    shard, _ = load_shard(req.model, req.layer_start, req.layer_end, req.num_layers)
    apply_lora(shard, req.lora_targets or [], req.lora_rank or 8, req.lora_scale or 20.0, seed=0)
    adapters = {k.removeprefix("adapter/"): v for k, v in tensors.items() if k.startswith("adapter/")}
    load_adapter_weights(shard, adapters)
    x = tensors["x_in"]
    t0 = time.perf_counter()
    if "targets" in tensors:
        out = token_losses(shard(x), tensors["targets"], tensors["mask"])
    else:
        out = shard(x)
    mx.eval(out)
    busy = time.perf_counter() - t0
    dest.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dest), {"out": out})
    return {"wall_s": busy, "busy_s": busy}
