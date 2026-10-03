"""Dataset path and upload guards. No login required for anonymous Take."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from slashcompute.common.config import EngineConfig
from slashcompute.coordinator.app import create_app
from slashcompute.coordinator.core import MAX_DATASET_BYTES, safe_dataset_source
from slashcompute.jobs import LoraFinetuneSpec


@pytest.fixture
def env(tmp_path, tiny_model, tiny_dataset):
    cfg = EngineConfig(home=tmp_path / "home", scheduler_tick_s=0.05, verify_rate=0.0)
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app.state.core, tiny_model, tiny_dataset


def _spec(tiny_model, tiny_dataset, **kw):
    base = dict(model=str(tiny_model), dataset_path=str(tiny_dataset), steps=2,
                batch_size=2, microbatches=1, lora_rank=4, min_stages=1)
    return LoraFinetuneSpec(**(base | kw))


def test_anonymous_jsonl_submit_still_works(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    r = client.post("/jobs", json=spec)
    assert r.status_code == 200, r.text
    assert r.json()["id"] in core.jobs


def test_submit_rejects_non_jsonl_path(env):
    client, core, tiny_model, tiny_dataset = env
    secret = tiny_dataset.parent / "notes.txt"
    secret.write_text("not a dataset")
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["dataset_path"] = str(secret)
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert core.jobs == {}


def test_submit_rejects_parent_segments(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["dataset_path"] = str(tiny_dataset.parent / ".." / tiny_dataset.name)
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert core.jobs == {}


def test_submit_rejects_symlink(env, tmp_path):
    client, core, tiny_model, tiny_dataset = env
    link = tmp_path / "alias.jsonl"
    link.symlink_to(tiny_dataset)
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["dataset_path"] = str(link)
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert core.jobs == {}
    with pytest.raises(ValueError, match="symlink"):
        safe_dataset_source(str(link))


def test_submit_rejects_foreign_model(env):
    client, core, tiny_model, tiny_dataset = env
    spec = json.loads(_spec(tiny_model, tiny_dataset).model_dump_json())
    spec["model"] = "evil/malware-repo"
    r = client.post("/jobs", json=spec)
    assert r.status_code == 400
    assert "not allowed" in r.json()["detail"]
    assert core.jobs == {}
    upload = client.post(
        "/jobs/upload",
        files={"dataset": ("train.jsonl", tiny_dataset.read_bytes(), "application/jsonl")},
        data={"model": "evil/malware-repo", "steps": 2, "min_stages": 1,
              "batch_size": 2, "microbatches": 1},
    )
    assert upload.status_code == 400
    assert "not allowed" in upload.json()["detail"]
    assert core.jobs == {}


def test_upload_rejects_oversize(env):
    client, core, tiny_model, _ = env
    too_big = b"x" * (MAX_DATASET_BYTES + 1)
    r = client.post(
        "/jobs/upload",
        files={"dataset": ("train.jsonl", too_big, "application/jsonl")},
        data={"model": str(tiny_model), "steps": 2, "min_stages": 1,
              "batch_size": 2, "microbatches": 1},
    )
    assert r.status_code == 400
    assert "too large" in r.json()["detail"]
    assert core.jobs == {}
