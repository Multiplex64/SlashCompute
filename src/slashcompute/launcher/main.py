"""``slashcompute`` — native window around the local shell."""

from __future__ import annotations

import os
import socket
import threading
import time

import httpx
import uvicorn

from slashcompute.web.server import SHELL_GENERATION, SHELL_HOST, SHELL_PORT, create_shell

WINDOW_W = 1280
WINDOW_H = 820


def shell_url() -> str:
    return f"http://127.0.0.1:{SHELL_PORT}"


def _port_open(host: str, port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _shell_generation(url: str, timeout: float = 0.6) -> int:
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/shell", timeout=timeout)
    except httpx.RequestError:
        return 0
    if r.status_code != 200:
        return 0
    try:
        return int(r.json().get("generation", 0))
    except (TypeError, ValueError, AttributeError):
        return 0


def _stop_listener(port: int) -> None:
    """Quit a leftover /compute shell so this launch can bind :8766."""
    import psutil

    for c in psutil.net_connections(kind="tcp"):
        if not c.laddr or c.laddr.port != port or c.status != "LISTEN" or not c.pid:
            continue
        try:
            proc = psutil.Process(c.pid)
        except (psutil.Error, OSError):
            continue
        if proc.pid == os.getpid() or proc.ppid() == os.getpid():
            continue
        try:
            cmd = " ".join(proc.cmdline())
        except (psutil.Error, OSError):
            cmd = ""
        if "slashcompute" not in cmd and "uvicorn" not in cmd:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except psutil.TimeoutExpired:
            proc.kill()


def ui_ready(url: str, timeout: float = 0.6) -> bool:
    try:
        r = httpx.get(url, timeout=timeout)
    except httpx.RequestError:
        return False
    if r.status_code != 200:
        return False
    text = r.text
    return "COMPUTE" in text or "/compute" in text


def _serve() -> None:
    uvicorn.run(create_shell(), host=SHELL_HOST, port=SHELL_PORT, log_level="warning")


def ensure_shell(wait: float = 8.0) -> str:
    """Bind the local UI if needed. Returns the URL the window should load."""
    url = shell_url()
    if ui_ready(url) and _shell_generation(url) >= SHELL_GENERATION:
        return url
    if ui_ready(url) or _port_open("127.0.0.1", SHELL_PORT):
        _stop_listener(SHELL_PORT)
        time.sleep(0.2)
        if ui_ready(url) and _shell_generation(url) >= SHELL_GENERATION:
            return url
        if _port_open("127.0.0.1", SHELL_PORT):
            raise SystemExit(
                f"Port {SHELL_PORT} is in use by an old /compute. Quit that app and open again."
            )
    threading.Thread(target=_serve, daemon=True).start()
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if ui_ready(url):
            return url
        time.sleep(0.1)
    raise SystemExit(f"UI did not start at {url}")


def open_window(url: str) -> None:
    import webview

    webview.create_window("/compute", url, width=WINDOW_W, height=WINDOW_H)
    webview.start()


def main() -> None:
    open_window(ensure_shell())


if __name__ == "__main__":
    main()
