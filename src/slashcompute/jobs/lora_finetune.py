"""Spec for the ``lora_finetune`` job type."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from slashcompute.common.config import DEV_MODEL


class LoraFinetuneSpec(BaseModel):
    kind: Literal["lora_finetune"] = "lora_finetune"
    model: str = DEV_MODEL
    # Path on the coordinator; copied into the coordinator's job store at submit.
    dataset_path: str
    steps: int = Field(100, ge=1)
    learning_rate: float = 1e-4
    lora_rank: int = Field(8, ge=1)
    lora_scale: float = 20.0
    lora_targets: list[str] = ["self_attn.q_proj", "self_attn.v_proj"]
    batch_size: int = Field(4, ge=1)
    microbatches: int = Field(2, ge=1)
    max_seq_len: int = Field(512, ge=8)
    seed: int = 0
    min_stages: int = Field(1, ge=1)
    max_stages: Optional[int] = None
    checkpoint_every: Optional[int] = None

    @model_validator(mode="after")
    def _check(self):
        if self.batch_size % self.microbatches:
            raise ValueError("batch_size must be divisible by microbatches")
        if self.max_stages is not None and self.max_stages < self.min_stages:
            raise ValueError("max_stages must be >= min_stages")
        return self

    @property
    def microbatch_size(self) -> int:
        return self.batch_size // self.microbatches
