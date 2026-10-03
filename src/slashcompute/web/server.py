"""Local shell on :8766 — UI, launcher control, coordinator proxy."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from slashcompute.common.config import DEMO_MODEL_CANDIDATES, DEV_MODEL
from slashcompute.launcher.controller import (
    FINISHES, Launcher, LauncherError, LauncherSettings,
)
from slashcompute.launcher.dashboard import PoolData, node_flops, overview, split_credits
from slashcompute.web import sample_grants

STATIC = Path(__file__).resolve().parent / "static"
SHELL_HOST = os.environ.get("SLASHCOMPUTE_SHELL_HOST", "127.0.0.1")
SHELL_PORT = int(os.environ.get("SLASHCOMPUTE_SHELL_PORT", "8766"))
SHELL_GENERATION = 3
MODELS = [DEV_MODEL, *DEMO_MODEL_CANDIDATES]
_PROXY_BLOCK = {"verify"}


def settings_from_body(body: dict) -> LauncherSettings:
    return LauncherSettings(
        mode=body.get("mode", "host"),
        url=body.get("url", ""),
        gpu_percent=body.get("gpu_percent", 50),
        contribute=body.get("contribute", True),
        finish=body.get("finish", "carbon"),
        session_token=body.get("session_token", ""),
        grant_split=body.get("grant_split", 0),
    ).clamp()


def create_shell(launcher: Optional[Launcher] = None) -> FastAPI:
    launch = launcher or Launcher()
    app = FastAPI(title="/compute")
    app.state.launcher = launch
    app.state.grants = sample_grants.sample_book()
    grants_lock = threading.Lock()

    @app.get("/")
    def index():
        path = STATIC / "index.html"
        if not path.is_file():
            raise HTTPException(500, "UI missing")
        return FileResponse(path)

    @app.get("/api/shell")
    def shell_info():
        return {"ok": True, "generation": SHELL_GENERATION, "proxy": "coord"}

    @app.get("/api/settings")
    def get_settings():
        return asdict(launch.load_settings())

    @app.post("/api/settings")
    def post_settings(body: dict):
        s = settings_from_body(body)
        launch.save_settings(s)
        return asdict(s)

    @app.get("/api/status")
    def status():
        snap = launch.snapshot()
        return {**asdict(snap), **asdict(launch.load_settings()),
                "coordinator_url": launch.coordinator_url(launch.load_settings()),
                "finishes": list(FINISHES)}

    @app.post("/api/start")
    def start(body: dict):
        s = settings_from_body(body)
        try:
            snap = launch.start(s)
        except LauncherError as e:
            raise HTTPException(400, str(e)) from e
        return {**asdict(snap), **asdict(launch.load_settings())}

    @app.post("/api/stop")
    def stop():
        return asdict(launch.stop())

    @app.post("/api/stop-agent")
    def stop_agent():
        return asdict(launch.stop_agent())

    @app.get("/api/overview")
    def get_overview():
        """Status, pool, this Mac and the leaderboard in one poll."""
        s = launch.load_settings()
        snap = launch.snapshot(s)
        pool = launch.fetch_pool(launch.proxy_url(s)) if snap.coordinator_up else PoolData()
        status = {**asdict(snap), **asdict(s), "coordinator_url": launch.coordinator_url(s),
                  "models": MODELS}
        return overview(status, pool, launch.my_node_id(), s.grant_split)

    # ------------------------------------------------------------ sample grants

    def grant_share() -> float:
        """This Mac's FLOPs pledged to grants by its credit split."""
        s = launch.load_settings()
        pool = launch.fetch_pool(launch.proxy_url(s))
        return split_credits(node_flops(pool.ledger, launch.my_node_id()),
                             s.grant_split)["to_grants"]

    def grant_board(sort: str = "top") -> dict:
        mode = sort if sort in sample_grants.SORTS else "top"
        return sample_grants.board(app.state.grants, mode, grant_share())

    def change_grants(fn) -> None:
        with grants_lock:
            try:
                app.state.grants = fn(app.state.grants)
            except sample_grants.GrantError as e:
                raise HTTPException(400, str(e)) from e

    def amount(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, "Enter a number.") from None

    @app.get("/api/grants")
    def list_grants(sort: str = "top"):
        return grant_board(sort)

    @app.post("/api/grants")
    def request_grant(body: dict):
        goal = amount(body.get("goal"))
        change_grants(lambda b: sample_grants.request(
            b, str(body.get("title", "")), str(body.get("summary", "")), goal))
        return grant_board(str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/fund")
    def fund_grant(grant_id: str, body: dict):
        flops, share = amount(body.get("amount")), grant_share()
        change_grants(lambda b: sample_grants.fund(
            b, grant_id, flops, sample_grants.available(b, share)))
        return grant_board(str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/review")
    def review_grant(grant_id: str, body: dict):
        change_grants(lambda b: sample_grants.review(b, grant_id, bool(body.get("approve"))))
        return grant_board(str(body.get("sort", "top")))

    @app.post("/api/discover")
    def discover():
        found = launch.find_on_lan()
        if not found:
            raise HTTPException(404, "No coordinator found on the LAN.")
        s = launch.load_settings()
        s.url = found
        launch.save_settings(s)
        return {"url": found}

    @app.api_route("/api/coord/{path:path}", methods=["GET", "POST", "PATCH"])
    async def proxy(path: str, request: Request):
        root = path.split("/", 1)[0]
        if root in _PROXY_BLOCK or not root:
            raise HTTPException(404, "not proxied")
        base = launch.proxy_url()
        if not base:
            raise HTTPException(503, "No coordinator URL. Host or enter one, then Start.")
        url = f"{base.rstrip('/')}/{path}"
        headers = {}
        if request.headers.get("authorization"):
            headers["authorization"] = request.headers["authorization"]
        if request.headers.get("cookie"):
            headers["cookie"] = request.headers["cookie"]
        try:
            if request.method == "GET":
                r = launch._http.get(url, params=dict(request.query_params),
                                     headers=headers or None, timeout=30.0)
            else:
                ct = request.headers.get("content-type", "")
                if request.method == "POST" and ct.startswith("multipart/"):
                    form = await request.form()
                    data, files = {}, {}
                    for key, val in form.multi_items():
                        if hasattr(val, "read"):
                            files[key] = (val.filename, await val.read(),
                                          val.content_type or "application/octet-stream")
                        else:
                            data[key] = val
                    r = launch._http.post(url, data=data, files=files or None,
                                          headers=headers or None, timeout=60.0)
                else:
                    fwd = dict(headers)
                    fwd["content-type"] = ct or "application/json"
                    body = await request.body()
                    if request.method == "PATCH":
                        r = launch._http.patch(url, content=body, headers=fwd, timeout=60.0)
                    else:
                        r = launch._http.post(url, content=body, headers=fwd, timeout=60.0)
        except httpx.RequestError as e:
            raise HTTPException(502, f"Coordinator unreachable at {base}: {e}") from e
        return Response(content=r.content, status_code=r.status_code,
                        media_type=r.headers.get("content-type", "application/json"))

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
