"""Analytical FLOP estimates for a stage's share of a training step.

Counts are for the logical (unquantized) model, so the same work earns the
same FLOPs regardless of quantization or chip.

Per token, one transformer layer's forward costs about
``2 * layer_params`` (matmuls) + ``2 * seq_len * hidden`` (causal attention
scores and weighted sum). The output head adds ``2 * hidden * vocab``.

With LoRA the base weights are frozen, so backward only computes input grads
(~1x forward); LoRA weight grads are negligible. Non-last stages also
recompute their forward during backward. So:

* non-last stage: forward + recompute + backward = 3x forward
* last stage: forward + backward (single value_and_grad pass) = 2x forward
"""

from __future__ import annotations

from slashcompute.pipeline.model_profile import ModelProfile


def forward_flops(profile: ModelProfile, n_layers: int, tokens: int, seq_len: int,
                  include_head: bool) -> float:
    per_token = n_layers * (2 * profile.layer_params + 2 * seq_len * profile.hidden_size)
    if include_head:
        per_token += 2 * profile.head_params
    return float(per_token) * tokens


def stage_step_flops(profile: ModelProfile, layer_start: int, layer_end: int, tokens: int,
                     seq_len: int, is_last: bool) -> float:
    fwd = forward_flops(profile, layer_end - layer_start, tokens, seq_len, include_head=is_last)
    return fwd * (2.0 if is_last else 3.0)


def replay_flops(profile: ModelProfile, layer_start: int, layer_end: int, tokens: int,
                 seq_len: int, is_last: bool) -> float:
    """A verification replay runs one forward pass."""
    return forward_flops(profile, layer_end - layer_start, tokens, seq_len, include_head=is_last)
