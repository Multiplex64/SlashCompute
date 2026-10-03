"""Local shell on :8766 — UI, launcher control, coordinator proxy."""

from __future__ import annotations

import json
import os
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
from slashcompute.launcher.dashboard import PoolData, overview

STATIC = Path(__file__).resolve().parent / "static"
SHELL_HOST = os.environ.get("SLASHCOMPUTE_SHELL_HOST", "127.0.0.1")
SHELL_PORT = int(os.environ.get("SLASHCOMPUTE_SHELL_PORT", "8766"))
SHELL_GENERATION = 5
MODELS = [DEV_MODEL, *DEMO_MODEL_CANDIDATES]
_PROXY_BLOCK = {"verify"}
_SORTS = ("top", "trending", "least")


def _proxy_blocked(path: str) -> bool:
    parts = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return True
        parts.append(part)
    return not parts or parts[0].lower() in _PROXY_BLOCK


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


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {}
    if request.headers.get("authorization"):
        headers["authorization"] = request.headers["authorization"]
    if request.headers.get("cookie"):
        headers["cookie"] = request.headers["cookie"]
    return headers


def _set_cookies(headers) -> list[str]:
    if headers is None:
        return []
    get_list = getattr(headers, "get_list", None)
    if callable(get_list):
        return [c for c in (get_list("set-cookie") or []) if c]
    raw = headers.get("set-cookie")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [c for c in raw if c]
    return [raw]


def _cookie_response(r) -> Response:
    media = "application/json"
    if getattr(r, "headers", None) is not None:
        media = r.headers.get("content-type", media) or media
    resp = Response(content=r.content, status_code=r.status_code, media_type=media)
    for cookie in _set_cookies(getattr(r, "headers", None)):
        resp.headers.append("set-cookie", cookie)
    return resp


def _coord_detail(r) -> str:
    try:
        data = r.json()
    except Exception:
        text = getattr(r, "text", None) or ""
        return text or "Coordinator error."
    detail = data.get("detail", data) if isinstance(data, dict) else data
    return detail if isinstance(detail, str) else json.dumps(detail)


def _grant_card(row: dict) -> dict:
    goal = float(row.get("goal_flops") or row.get("goal") or 0)
    raised = float(row.get("received_flops") or row.get("raised") or 0)
    progress = float(row.get("progress") or 0)
    if not progress and goal:
        progress = raised / goal
    return {
        "id": row.get("id"),
        "title": row.get("title", ""),
        "author": row.get("author", ""),
        "summary": row.get("body") or row.get("summary") or "",
        "goal": goal,
        "raised": raised,
        "progress": progress,
        "remaining": max(0.0, goal - raised),
        "backers": int(row.get("backers") or 0),
        "tag": str(row.get("status") or "approved"),
        "status": row.get("status", "approved"),
    }


def _empty_board() -> dict:
    return {
        "sample": False, "online": False, "grants": [], "pending": [],
        "pledged": 0.0, "starter": 0.0, "share": 0.0, "available": 0.0,
        "leaders": [],
    }


def create_shell(launcher: Optional[Launcher] = None) -> FastAPI:
    launch = launcher or Launcher()
    app = FastAPI(title="/compute")
    app.state.launcher = launch

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

    # ------------------------------------------------------------ live grants

    def amount(value) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, "Enter a number.") from None

    def coord_call(request: Request, method: str, path: str, *,
                   params: Optional[dict] = None, payload: Optional[dict] = None,
                   required: bool = True):
        base = launch.proxy_url()
        if not base:
            if required:
                raise HTTPException(503, "No coordinator URL. Host or enter one, then Start.")
            return None
        url = f"{base.rstrip('/')}/{path.lstrip('/')}"
        headers = _forward_headers(request)
        try:
            if method == "GET":
                r = launch._http.get(url, params=params or None,
                                     headers=headers or None, timeout=30.0)
            else:
                body = json.dumps(payload or {}).encode()
                fwd = dict(headers)
                fwd["content-type"] = "application/json"
                r = launch._http.post(url, content=body, headers=fwd, timeout=30.0)
        except (httpx.RequestError, ConnectionError, OSError) as e:
            if required:
                raise HTTPException(502, f"Coordinator unreachable at {base}: {e}") from e
            return None
        if r.status_code >= 400:
            if required:
                raise HTTPException(r.status_code, _coord_detail(r))
            return None
        try:
            return r.json()
        except Exception:
            return None

    def grant_board(request: Request, sort: str = "top") -> dict:
        mode = sort if sort in _SORTS else "top"
        rows = coord_call(request, "GET", "/grants", params={"sort": mode}, required=False)
        if rows is None:
            return _empty_board()
        if not isinstance(rows, list):
            rows = []
        cards = [_grant_card(g) for g in rows if isinstance(g, dict)]
        approved = [g for g in cards if g["status"] == "approved"]
        pending = [g for g in cards if g["status"] == "pending"]
        me = coord_call(request, "GET", "/auth/me", required=False) or {}
        user = me.get("user") if isinstance(me, dict) else None
        credits = me.get("credits") if isinstance(me, dict) else None
        available = float((credits or {}).get("balance") or 0)
        pledged = 0.0
        txns = coord_call(request, "GET", "/credits/transactions",
                          params={"limit": 50}, required=False)
        if isinstance(txns, dict):
            pledged = sum(
                abs(float(t.get("amount") or 0))
                for t in txns.get("items") or []
                if t.get("kind") == "donate"
            )
        raw_leaders = coord_call(request, "GET", "/community/leaderboard", required=False)
        me_id = user.get("id") if isinstance(user, dict) else None
        leaders = []
        if isinstance(raw_leaders, list):
            for i, row in enumerate(raw_leaders, 1):
                if not isinstance(row, dict):
                    continue
                leaders.append({
                    "rank": i,
                    "user_id": row.get("user_id"),
                    "name": row.get("name", ""),
                    "flops": float(row.get("lifetime_earned") or 0),
                    "is_me": bool(me_id and row.get("user_id") == me_id),
                })
        return {
            "sample": False, "online": True,
            "grants": approved, "pending": pending,
            "pledged": pledged, "starter": 0.0,
            "share": float((user or {}).get("grant_split") or 0),
            "available": available, "leaders": leaders,
        }

    @app.get("/api/grants")
    def list_grants(request: Request, sort: str = "top"):
        return grant_board(request, sort)

    @app.post("/api/grants")
    def request_grant(body: dict, request: Request):
        goal = amount(body.get("goal"))
        coord_call(request, "POST", "/grants", payload={
            "title": str(body.get("title", "")),
            "body": str(body.get("summary", "") or body.get("body", "")),
            "goal_flops": goal,
        })
        return grant_board(request, str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/fund")
    def fund_grant(grant_id: str, body: dict, request: Request):
        coord_call(request, "POST", f"/grants/{grant_id}/donate", payload={
            "flops": amount(body.get("amount") if body.get("amount") is not None
                            else body.get("flops")),
        })
        return grant_board(request, str(body.get("sort", "top")))

    @app.post("/api/grants/{grant_id}/review")
    def review_grant(grant_id: str, body: dict, request: Request):
        coord_call(request, "POST", f"/admin/grants/{grant_id}/review", payload={
            "approve": bool(body.get("approve")),
            "note": body.get("note"),
        })
        return grant_board(request, str(body.get("sort", "top")))

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
        if _proxy_blocked(path):
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
        return _cookie_response(r)

    if STATIC.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
