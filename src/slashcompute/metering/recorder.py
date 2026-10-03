"""Turns a stage's raw step stats into a ``UsageSample`` for the ledger.

These are raw measurements only. The credit formula (FLOPs combined with
memory) is decided later and computed from the stored samples.
"""

from __future__ import annotations

from slashcompute.common.protocol import UsageSample
from slashcompute.metering.flops import stage_step_flops
from slashcompute.pipeline.model_profile import ModelProfile
from slashcompute.pipeline.schedule import StepStats


class UsageRecorder:
    def __init__(self, profile: ModelProfile, layer_start: int, layer_end: int) -> None:
        self.profile = profile
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.is_last = layer_end == profile.num_layers
        self.total_flops = 0.0
        self.total_mem_byte_seconds = 0.0
        self.steps = 0

    def sample(self, s: StepStats) -> UsageSample:
        flops = stage_step_flops(self.profile, self.layer_start, self.layer_end,
                                 s.tokens_processed, s.seq_len, self.is_last)
        mem_bs = float(s.resident_mem_bytes) * s.wall_s
        self.total_flops += flops
        self.total_mem_byte_seconds += mem_bs
        self.steps += 1
        return UsageSample(
            flops=flops, tokens=s.tokens_processed, peak_mem_bytes=s.peak_mem_bytes,
            resident_mem_bytes=s.resident_mem_bytes, mem_byte_seconds=mem_bs,
            wall_s=s.wall_s, busy_s=s.busy_s,
        )
