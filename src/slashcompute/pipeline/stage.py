"""Synchronous MLX compute for one pipeline stage.

Backward recomputes the stage forward (activation recomputation): only the
stage *input* is kept per microbatch, never the intermediate activations.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from slashcompute.pipeline.lora import adapter_weights, filter_layers, load_adapter_weights
from slashcompute.pipeline.shard import ShardModule


def token_losses(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    ce = nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="none")
    return ce * mask


@dataclass
class RingEntry:
    """What a stage consumed and produced for microbatch 0 of a step, plus the
    adapters it used. Kept so a sampled step can be replayed elsewhere."""

    x_in: mx.array
    out: mx.array  # hidden states, or per-token losses on the last stage
    adapters: dict[str, mx.array]
    targets: Optional[mx.array] = None
    mask: Optional[mx.array] = None


class StageCompute:
    def __init__(self, shard: ShardModule, learning_rate: float, ring_size: int = 8) -> None:
        self.shard = shard
        self.optimizer = optim.Adam(learning_rate=learning_rate)
        self.optimizer.init(shard.trainable_parameters())
        self._grads = None
        self.ring: OrderedDict[int, RingEntry] = OrderedDict()
        self.ring_size = ring_size

    @property
    def is_first(self) -> bool:
        return self.shard.is_first

    @property
    def is_last(self) -> bool:
        return self.shard.is_last

    # ------------------------------------------------------------ compute

    def _apply(self, params, x):
        self.shard.update(params)
        return self.shard(x)

    def forward(self, x: mx.array) -> mx.array:
        h = self.shard(x)
        mx.eval(h)
        return h

    def backward(self, x: mx.array, grad_out: mx.array) -> Optional[mx.array]:
        """Accumulate param grads for one microbatch; return grad wrt ``x``
        (None on stage 0, whose input is token ids)."""
        params = self.shard.trainable_parameters()

        def surrogate(p, xin):
            return (self._apply(p, xin).astype(mx.float32) * grad_out.astype(mx.float32)).sum()

        if self.is_first:
            _, gp = mx.value_and_grad(surrogate, argnums=0)(params, x)
            gx = None
        else:
            _, (gp, gx) = mx.value_and_grad(surrogate, argnums=(0, 1))(params, x)
        self.shard.update(params)
        self._accumulate(gp)
        mx.eval(self._grads, *( [gx] if gx is not None else []))
        return gx

    def forward_backward_loss(self, x: mx.array, targets: mx.array, mask: mx.array,
                              ntoks_total: int) -> tuple[float, mx.array, Optional[mx.array]]:
        """Last stage: loss (normalised by tokens in the whole batch), param
        grads accumulated, and grad wrt ``x``. Returns (loss, per-token
        losses, grad_x)."""
        params = self.shard.trainable_parameters()
        denom = max(ntoks_total, 1)

        def loss_fn(p, xin):
            tl = token_losses(self._apply(p, xin), targets, mask)
            return tl.sum() / denom, tl

        if self.is_first:
            (loss, tl), gp = mx.value_and_grad(loss_fn, argnums=0)(params, x)
            gx = None
        else:
            (loss, tl), (gp, gx) = mx.value_and_grad(loss_fn, argnums=(0, 1))(params, x)
        self.shard.update(params)
        self._accumulate(gp)
        mx.eval(loss, tl, self._grads, *([gx] if gx is not None else []))
        return loss.item(), tl, gx

    def _accumulate(self, gp) -> None:
        self._grads = gp if self._grads is None else tree_map(mx.add, self._grads, gp)

    def apply_update(self) -> None:
        if self._grads is None:
            return
        self.optimizer.update(self.shard, self._grads)
        mx.eval(self.shard.trainable_parameters(), self.optimizer.state)
        self._grads = None

    def discard_grads(self) -> None:
        self._grads = None

    # ------------------------------------------------------------ verification ring

    def remember(self, step: int, entry: RingEntry) -> None:
        self.ring[step] = entry
        while len(self.ring) > self.ring_size:
            self.ring.popitem(last=False)

    def current_adapters(self) -> dict[str, mx.array]:
        # mx arrays are immutable; the optimizer swaps in new arrays, so these
        # references keep the pre-update values alive.
        return adapter_weights(self.shard)

    def save_bundle(self, step: int, path: Path) -> bool:
        entry = self.ring.get(step)
        if entry is None:
            return False
        tensors = {"x_in": entry.x_in, "out": entry.out}
        tensors |= {f"adapter/{k}": v for k, v in entry.adapters.items()}
        if entry.targets is not None:
            tensors |= {"targets": entry.targets, "mask": entry.mask}
        mx.save_safetensors(str(path), tensors, metadata={"step": str(step)})
        return True

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: Path, step: int) -> None:
        tensors = {f"adapter/{k}": v for k, v in adapter_weights(self.shard).items()}
        for k, v in tree_flatten(self.optimizer.state):
            tensors[f"opt/{k}"] = v
        meta = {"step": str(step), "layer_start": str(self.shard.layer_start),
                "layer_end": str(self.shard.layer_end)}
        mx.save_safetensors(str(path), tensors, metadata=meta)

    def load_checkpoint(self, path: Path) -> int:
        tensors, meta = mx.load(str(path), return_metadata=True)
        adapters = {k.removeprefix("adapter/"): v for k, v in tensors.items() if k.startswith("adapter/")}
        load_adapter_weights(self.shard, adapters)
        opt = {k.removeprefix("opt/"): v for k, v in tensors.items() if k.startswith("opt/")}
        mine = filter_layers(opt, self.shard.layer_start, self.shard.layer_end)
        scalars = {k: v for k, v in opt.items() if "." not in k}
        self.optimizer.state = tree_unflatten(list((mine | scalars).items()))
        self.optimizer.init(self.shard.trainable_parameters())
        mx.eval(self.optimizer.state)
        return int(meta.get("step", 0))


def merge_checkpoints(paths: list[Path], out: Path) -> None:
    """Merge per-stage checkpoint files into one (keys are global per layer)."""
    merged: dict[str, mx.array] = {}
    meta: dict[str, str] = {}
    for p in paths:
        tensors, m = mx.load(str(p), return_metadata=True)
        for k, v in tensors.items():
            if k.startswith("opt/") and "." not in k.removeprefix("opt/") and k in merged:
                continue  # scalar optimizer state (step, lr) is identical across stages
            merged[k] = v
        meta["step"] = m.get("step", "0")
    mx.save_safetensors(str(out), merged, metadata=meta)
