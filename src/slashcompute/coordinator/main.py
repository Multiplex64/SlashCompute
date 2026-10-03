"""``slashcompute-coordinator`` CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
import typer

from slashcompute.common.config import EngineConfig
from slashcompute.common.logging import setup_logging

app = typer.Typer(no_args_is_help=True, help="/compute coordinator")


def _url(url: Optional[str]) -> str:
    return url or f"http://127.0.0.1:{EngineConfig.from_env().coordinator_port}"


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default 0.0.0.0)"),
    port: int = typer.Option(None, help="Port (default 8765)"),
    home: Optional[Path] = typer.Option(None, help="State directory (default ~/.slashcompute)"),
    mdns: bool = typer.Option(True, help="Advertise on the LAN via mDNS"),
    inference_transport: Optional[str] = typer.Option(
        None, help="LLM inference: direct (LAN, default) or relay (internet: RPC through the coordinator)"),
):
    """Run the coordinator."""
    import uvicorn

    from slashcompute.coordinator.app import create_app
    from slashcompute.inference.config import TRANSPORTS, InferenceSettings

    setup_logging("coordinator")
    cfg = EngineConfig.from_env(coordinator_host=host, coordinator_port=port, home=home)
    inference = InferenceSettings.from_env(DB_PATH=str(cfg.home / "inference.sqlite3"),
                                           MODELS_DIR=str(cfg.home / "models"))
    if inference_transport:
        if inference_transport not in TRANSPORTS:
            raise typer.BadParameter(f"--inference-transport must be one of {', '.join(TRANSPORTS)}")
        inference = inference.replace(TRANSPORT=inference_transport)
    uvicorn.run(create_app(cfg, advertise=mdns, inference=inference), host=cfg.coordinator_host,
                port=cfg.coordinator_port, log_level="warning", ws_ping_interval=20,
                ws_max_size=64 * 1024 * 1024)


@app.command()
def submit(spec: Path = typer.Argument(..., help="JSON job spec"), url: Optional[str] = None):
    """Submit a job spec (JSON file)."""
    body = json.loads(spec.read_text())
    if "dataset_path" in body:
        body["dataset_path"] = str((spec.parent / body["dataset_path"]).resolve()) \
            if not Path(body["dataset_path"]).is_absolute() else body["dataset_path"]
    r = httpx.post(f"{_url(url)}/jobs", json=body, timeout=60)
    if r.status_code >= 400:
        typer.echo(r.text, err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps(r.json(), indent=2))


def _get(path: str, url: Optional[str]):
    r = httpx.get(f"{_url(url)}{path}", timeout=30)
    r.raise_for_status()
    typer.echo(json.dumps(r.json(), indent=2))


@app.command()
def jobs(job_id: Optional[str] = typer.Argument(None), url: Optional[str] = None):
    """List jobs, or show one."""
    _get(f"/jobs/{job_id}" if job_id else "/jobs", url)


@app.command()
def nodes(url: Optional[str] = None):
    """List connected nodes."""
    _get("/nodes", url)


@app.command()
def ledger(url: Optional[str] = None):
    """Per-node usage totals (raw FLOPs, memory, time)."""
    _get("/ledger", url)


@app.command()
def verifications(url: Optional[str] = None):
    """List verification results."""
    _get("/verifications", url)


@app.command()
def cancel(job_id: str, url: Optional[str] = None):
    """Cancel a job."""
    r = httpx.post(f"{_url(url)}/jobs/{job_id}/cancel", timeout=30)
    r.raise_for_status()
    typer.echo(json.dumps(r.json(), indent=2))


if __name__ == "__main__":
    app()
