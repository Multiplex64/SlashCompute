"""HTTP + WebSocket front end for the coordinator."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import ValidationError
from sqlmodel import select

from slashcompute.common.config import EngineConfig
from slashcompute.common.protocol import Register, dump, parse_agent_message
from slashcompute.coordinator.core import Coordinator
from slashcompute.coordinator.db import Verification
from slashcompute.jobs import parse_spec

log = logging.getLogger(__name__)


def create_app(cfg: EngineConfig, advertise: bool = False) -> FastAPI:
    core = Coordinator(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        core.start()
        adv = None
        if advertise:
            try:
                from slashcompute.common.discovery import Advertiser

                adv = Advertiser(cfg.coordinator_port)
            except Exception as e:
                log.warning("mDNS advertising unavailable: %s", e)
        yield
        if adv:
            adv.close()
        await core.stop()

    app = FastAPI(title="/compute coordinator", lifespan=lifespan)
    app.state.core = core

    # ------------------------------------------------------------ agents

    @app.websocket("/ws/agent")
    async def agent_ws(ws: WebSocket):
        await ws.accept()
        send_lock = asyncio.Lock()

        async def send(msg):
            async with send_lock:
                await ws.send_text(dump(msg))

        node_id: Optional[str] = None
        try:
            first = parse_agent_message(await ws.receive_text())
            if not isinstance(first, Register):
                await ws.close(code=4000, reason="first message must be register")
                return
            node_id = first.node_id
            await core.on_register(first, send)
            while True:
                raw = await ws.receive_text()
                try:
                    msg = parse_agent_message(raw)
                except ValidationError as e:
                    log.warning("bad message from %s: %s", node_id[:8], e)
                    continue
                await core.handle(node_id, msg)
        except WebSocketDisconnect:
            pass
        except Exception:
            log.exception("agent session error")
        finally:
            node = core.registry.get(node_id) if node_id else None
            if node is not None and node.send is send:
                await core.on_disconnect(node_id)

    # ------------------------------------------------------------ jobs

    def _job(job_id: str):
        job = core.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.post("/jobs")
    async def submit_job(body: dict):
        try:
            spec = parse_spec(body)
            job = core.submit(spec)
        except (ValidationError, ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        return core.job_view(job)

    @app.get("/jobs")
    async def list_jobs():
        return [core.job_view(j) for j in sorted(core.jobs.values(), key=lambda j: j.row.submitted_at)]

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        return core.job_view(_job(job_id))

    @app.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = _job(job_id)
        await core.cancel_job(job)
        return core.job_view(job)

    @app.get("/jobs/{job_id}/usage")
    async def job_usage(job_id: str):
        _job(job_id)
        return [r.model_dump() for r in core.ledger.job_records(job_id)]

    @app.get("/jobs/{job_id}/dataset")
    async def get_dataset(job_id: str):
        _job(job_id)
        return FileResponse(core.dataset_path(job_id))

    @app.get("/jobs/{job_id}/checkpoints/{step}")
    async def get_checkpoint(job_id: str, step: int):
        path = core.checkpoints.merged_path(job_id, step)
        if not path.exists():
            raise HTTPException(404, "checkpoint not found")
        return FileResponse(path)

    @app.post("/jobs/{job_id}/checkpoints/{step}")
    async def put_checkpoint(job_id: str, step: int, epoch: int, stage: int, request: Request):
        data = await request.body()
        try:
            core.on_checkpoint_upload(job_id, epoch, stage, step, data)
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"ok": True}

    @app.get("/jobs/{job_id}/adapter/{name}")
    async def get_adapter(job_id: str, name: str):
        if name not in ("adapters.safetensors", "adapter_config.json"):
            raise HTTPException(404)
        path = core.checkpoints.job_dir(job_id) / "adapter" / name
        if not path.exists():
            raise HTTPException(404, "adapter not ready")
        return FileResponse(path)

    # ------------------------------------------------------------ nodes, ledger, verification

    @app.get("/nodes")
    async def nodes():
        return core.node_view()

    @app.get("/ledger")
    async def ledger():
        return core.ledger.summary()

    @app.get("/verifications")
    async def verifications():
        with core.db.session() as s:
            rows = s.exec(select(Verification).order_by(Verification.created_at)).all()
        return [r.model_dump() for r in rows]

    @app.post("/verify/{vid}/bundle")
    async def put_bundle(vid: str, request: Request):
        try:
            core.verification.store_bundle(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a bundle")
        return {"ok": True}

    @app.get("/verify/{vid}/bundle")
    async def get_bundle(vid: str):
        if core.db.get(Verification, vid) is None:
            raise HTTPException(404)
        path = core.verification.bundle_path(vid)
        if not path.exists():
            raise HTTPException(404)
        return FileResponse(path)

    @app.post("/verify/{vid}/result")
    async def put_result(vid: str, request: Request):
        try:
            core.verification.store_result(vid, await request.body())
        except KeyError:
            raise HTTPException(404, "no verification awaiting a result")
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True, "nodes": len(core.registry.nodes), "jobs": len(core.jobs)}

    return app
