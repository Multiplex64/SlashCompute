"""Persistent coordinator state (SQLite via SQLModel)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine


def now() -> float:
    return time.time()


class Node(SQLModel, table=True):
    __tablename__ = "nodes"
    id: str = Field(primary_key=True)
    name: str
    chip: str
    memory_contrib_bytes: int
    matmul_tflops: float
    mem_bandwidth_gbps: float
    first_seen: float = Field(default_factory=now)
    last_seen: float = Field(default_factory=now)
    online: bool = True
    canary_passed: Optional[bool] = None


class Job(SQLModel, table=True):
    __tablename__ = "jobs"
    id: str = Field(primary_key=True)
    kind: str
    spec_json: str
    status: str = "queued"  # queued|starting|running|recovering|completed|failed|cancelled
    submitted_at: float = Field(default_factory=now)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    epoch: int = 0
    recoveries: int = 0
    last_checkpoint_step: int = 0
    progress_step: int = 0
    last_loss: Optional[float] = None
    error: Optional[str] = None


class StageRun(SQLModel, table=True):
    __tablename__ = "stage_runs"
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    epoch: int
    stage_idx: int
    node_id: str = Field(index=True)
    layer_start: int
    layer_end: int
    started_at: float = Field(default_factory=now)
    ended_at: Optional[float] = None
    end_reason: Optional[str] = None


class UsageRecord(SQLModel, table=True):
    __tablename__ = "usage_records"
    id: Optional[int] = Field(default=None, primary_key=True)
    kind: str = "train"  # train | verify
    job_id: Optional[str] = Field(default=None, index=True)
    epoch: Optional[int] = None
    stage_idx: Optional[int] = None
    node_id: str = Field(index=True)
    step: Optional[int] = None
    flops: float
    tokens: int = 0
    peak_mem_bytes: int = 0
    resident_mem_bytes: int = 0
    mem_byte_seconds: float = 0.0
    wall_s: float = 0.0
    busy_s: float = 0.0
    loss: Optional[float] = None
    in_digest: Optional[str] = None
    out_digest: Optional[str] = None
    disputed: bool = False
    created_at: float = Field(default_factory=now)


class Verification(SQLModel, table=True):
    __tablename__ = "verifications"
    id: str = Field(primary_key=True)
    kind: str  # replay | canary | chain
    job_id: Optional[str] = Field(default=None, index=True)
    epoch: Optional[int] = None
    step: Optional[int] = None
    stage_idx: Optional[int] = None
    target_node_id: str
    verifier_node_id: Optional[str] = None
    # fetching -> queued -> running -> passed|failed ; or error
    status: str = "fetching"
    rel_error: Optional[float] = None
    detail: Optional[str] = None
    created_at: float = Field(default_factory=now)
    finished_at: Optional[float] = None


class Checkpoint(SQLModel, table=True):
    __tablename__ = "checkpoints"
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: str = Field(index=True)
    step: int
    path: str
    created_at: float = Field(default_factory=now)


class Database:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
        SQLModel.metadata.create_all(self.engine)

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def add(self, *rows) -> None:
        with self.session() as s:
            for r in rows:
                s.add(r)
            s.commit()

    def get(self, model, key):
        with self.session() as s:
            return s.get(model, key)

    def save(self, row) -> None:
        with self.session() as s:
            s.merge(row)
            s.commit()
