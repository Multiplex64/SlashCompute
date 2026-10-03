import os
import stat
import subprocess
from pathlib import Path

from slashcompute.launcher.main import SHELL_GENERATION, ensure_shell, ui_ready


ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts" / "macos" / "install_app.sh"


def test_ui_ready_accepts_compute_page(monkeypatch):
    class R:
        status_code = 200
        text = "<title>/compute</title>\n<h1>COMPUTE</h1>"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is True


def test_ui_ready_rejects_foreign_port(monkeypatch):
    class R:
        status_code = 200
        text = "ok"

    monkeypatch.setattr("slashcompute.launcher.main.httpx.get", lambda *a, **k: R())
    assert ui_ready("http://127.0.0.1:8766") is False


def test_ensure_shell_attaches_when_already_up(monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.main.ui_ready", lambda url, timeout=0.6: True)
    monkeypatch.setattr("slashcompute.launcher.main._shell_generation", lambda url: SHELL_GENERATION)
    assert ensure_shell() == "http://127.0.0.1:8766"


def test_install_app_writes_plist_and_launcher(tmp_path):
    repo = tmp_path / "repo"
    py = repo / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    dest = tmp_path / "out" / "compute.app"
    dest.parent.mkdir()
    subprocess.run(
        ["bash", str(INSTALL), "--repo", str(repo), "--dest", str(dest)],
        check=True,
        env={**os.environ, "HOME": str(tmp_path)},
    )
    plist = (dest / "Contents" / "Info.plist").read_text()
    assert "com.slashcompute.app" in plist
    assert "<string>/compute</string>" in plist
    assert "<string>compute</string>" in plist
    launch = (dest / "Contents" / "MacOS" / "compute").read_text()
    assert str(repo) in launch
    assert "-m slashcompute.launcher.main" in launch
    assert os.access(dest / "Contents" / "MacOS" / "compute", os.X_OK)
    icon = dest / "Contents" / "Resources" / "icon.png"
    assert icon.is_file()
    assert icon.read_bytes() == (ROOT / "scripts" / "macos" / "icon.png").read_bytes()
