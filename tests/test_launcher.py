import json
import signal
from pathlib import Path

import pytest

from slashcompute.launcher.controller import (
    Launcher, LauncherError, LauncherSettings, normalize_url,
)


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health
        self.urls: list[str] = []

    def get(self, url: str, timeout: float = 1.0):
        self.urls.append(url)
        if self.health is None:
            raise ConnectionError("down")
        return FakeResponse(self.health)


def _launcher(tmp_path: Path, **kw) -> Launcher:
    spawned: list[FakeProc] = []
    next_pid = {"n": 4000}

    def popen(argv, **_kw):
        next_pid["n"] += 1
        proc = FakeProc(next_pid["n"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: None),
        lan_ip_fn=kw.pop("lan_ip_fn", lambda: "192.168.1.20"),
    )
    launcher._spawned = spawned  # type: ignore[attr-defined]
    return launcher


def test_normalize_url():
    assert normalize_url("") == ""
    assert normalize_url("192.168.1.10") == "http://192.168.1.10:8765"
    assert normalize_url("192.168.1.10:9000") == "http://192.168.1.10:9000"
    assert normalize_url("http://10.0.0.2:8765/") == "http://10.0.0.2:8765"


def test_settings_persist(tmp_path):
    launcher = _launcher(tmp_path)
    s = LauncherSettings(mode="join", url="http://10.0.0.1:8765", gpu_percent=75,
                         contribute=False)
    launcher.save_settings(s)
    raw = json.loads((tmp_path / "launcher.json").read_text())
    assert raw["mode"] == "join"
    assert raw["gpu_percent"] == 75
    loaded = launcher.load_settings()
    assert loaded == s.clamp()


def test_settings_corrupt_and_clamp(tmp_path):
    launcher = _launcher(tmp_path)
    (tmp_path / "launcher.json").write_text("not-json")
    assert launcher.load_settings() == LauncherSettings()
    launcher.save_settings(LauncherSettings(mode="nope", gpu_percent=999, contribute=1))
    s = launcher.load_settings()
    assert s.mode == "host"
    assert s.gpu_percent == 100
    assert s.contribute is True


def test_coordinator_and_agent_argv(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.coordinator_argv() == [
        "/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve",
        "--home", str(tmp_path),
    ]
    assert launcher.agent_argv("http://192.168.1.20:8765", 40) == [
        "/opt/venv/bin/python", "-m", "slashcompute.agent.main", "start",
        "--url", "http://192.168.1.20:8765", "--gpu-percent", "40",
        "--no-sandbox", "--home", str(tmp_path),
    ]
    assert "--session-token" in launcher.agent_argv(
        "http://192.168.1.20:8765", 40, session_token="tok",
    )


def test_host_url_uses_lan_ip(tmp_path):
    launcher = _launcher(tmp_path, lan_ip_fn=lambda: "10.1.2.3")
    assert launcher.coordinator_url(LauncherSettings(mode="host")) == "http://10.1.2.3:8765"
    assert launcher.coordinator_url(LauncherSettings(mode="join", url="10.1.2.9")) == (
        "http://10.1.2.9:8765"
    )


def test_start_host_spawns_coordinator_and_agent(tmp_path, monkeypatch):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)

    def health_after_spawn(url: str):
        if launcher._spawned:  # type: ignore[attr-defined]
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    def alive(pid: int) -> bool:
        return any(p.pid == pid for p in launcher._spawned)  # type: ignore[attr-defined]

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    launcher.poll_health = health_after_spawn  # type: ignore[method-assign]
    snap = launcher.start(LauncherSettings(mode="host", gpu_percent=50, contribute=True))
    argv_lists = [p.argv for p in launcher._spawned]  # type: ignore[attr-defined]
    assert argv_lists[0][:4] == ["/opt/venv/bin/python", "-m", "slashcompute.coordinator.main", "serve"]
    assert argv_lists[1][2:4] == ["slashcompute.agent.main", "start"]
    assert "--url" in argv_lists[1] and "127.0.0.1" in argv_lists[1][argv_lists[1].index("--url") + 1]
    assert "--no-sandbox" in argv_lists[1]
    assert (tmp_path / "coordinator.pid").read_text().strip() == str(launcher._spawned[0].pid)
    assert snap.last_error == ""


def test_start_host_without_contribute_skips_agent(tmp_path):
    http = FakeHTTP()
    launcher = _launcher(tmp_path, http=http)
    launcher.poll_health = lambda url: {"ok": True} if launcher._spawned else None  # type: ignore
    launcher.start(LauncherSettings(mode="host", contribute=False))
    kinds = [p.argv[2] for p in launcher._spawned]  # type: ignore[attr-defined]
    assert kinds == ["slashcompute.coordinator.main"]


def test_start_rebinds_agent_when_session_changes(tmp_path, monkeypatch):
    http = FakeHTTP({"ok": True, "nodes": 0, "jobs": 0})
    launcher = _launcher(tmp_path, http=http)
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    (tmp_path / "agent.session").write_text("old\n")
    running = {77: True}

    def alive(pid: int) -> bool:
        return running.get(pid, False)

    def stop(paths):
        running[77] = False
        paths.clear_pid()
        return True

    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", alive)
    monkeypatch.setattr("slashcompute.launcher.controller.request_stop", stop)
    launcher.start(LauncherSettings(mode="host", contribute=True, session_token="newtok"))
    spawned = launcher._spawned  # type: ignore[attr-defined]
    assert len(spawned) == 1
    assert spawned[0].argv[-2:] == ["--session-token", "newtok"]
    assert (tmp_path / "agent.session").read_text() == "newtok"


def test_start_when_already_up_is_noop(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path, http=FakeHTTP({"ok": True, "nodes": 1, "jobs": 0}))
    (tmp_path / "agent" / "agent.pid").write_text("77\n")
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: True)
    launcher.start(LauncherSettings(mode="host", contribute=True))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_join_requires_url(tmp_path):
    launcher = _launcher(tmp_path)
    with pytest.raises(LauncherError, match="coordinator URL"):
        launcher.start(LauncherSettings(mode="join", url=""))
    assert launcher._spawned == []  # type: ignore[attr-defined]


def test_start_join_without_health_fails(tmp_path):
    launcher = _launcher(tmp_path, http=FakeHTTP(None))
    with pytest.raises(LauncherError, match="No coordinator"):
        launcher.start(LauncherSettings(mode="join", url="http://10.0.0.8:8765"))


def test_find_on_lan(tmp_path):
    launcher = _launcher(
        tmp_path,
        discover_fn=lambda timeout=5.0: "http://192.168.0.4:8765/",
    )
    assert launcher.find_on_lan() == "http://192.168.0.4:8765"


def test_stop_signals_agent_and_coordinator(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []

    def fake_kill(pid, sig):
        kills.append((pid, sig))

    monkeypatch.setattr("os.kill", fake_kill)
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop()
    assert (111, signal.SIGTERM) in kills
    assert (222, signal.SIGTERM) in kills


def test_snapshot_reads_health_and_agent_status(tmp_path, monkeypatch):
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive", lambda pid: pid == 9)
    launcher = _launcher(
        tmp_path,
        http=FakeHTTP({"ok": True, "nodes": 2, "jobs": 1}),
        lan_ip_fn=lambda: "192.168.9.9",
    )
    launcher.paths.write_status(status="idle", job_id=None)
    launcher.paths.pid_file.write_text("9\n")
    snap = launcher.snapshot(LauncherSettings(mode="host"))
    assert snap.coordinator_up is True
    assert snap.nodes == 2
    assert snap.jobs == 1
    assert snap.agent_running is True
    assert snap.agent_status == "idle"
    assert snap.lan_ip == "192.168.9.9"


def test_stop_agent_leaves_coordinator_running(tmp_path, monkeypatch):
    kills: list[tuple[int, int]] = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    launcher = _launcher(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    (tmp_path / "agent" / "agent.pid").write_text("222\n")
    launcher.stop_agent()
    assert (222, signal.SIGTERM) in kills
    assert (111, signal.SIGTERM) not in kills  # only liveness probes (signal 0)


def test_my_node_id_never_creates_one(tmp_path):
    launcher = _launcher(tmp_path)
    assert launcher.my_node_id() is None
    assert not launcher.paths.node_id_file.exists()
    launcher.paths.node_id_file.write_text("abc123\n")
    assert launcher.my_node_id() == "abc123"


def test_fetch_pool_down_and_partial(tmp_path):
    assert _launcher(tmp_path, http=FakeHTTP(None)).fetch_pool("http://x:8765").online is False
    # Health answers but list endpoints return a dict: tolerated as empty lists.
    pool = _launcher(tmp_path, http=FakeHTTP({"ok": True})).fetch_pool("http://x:8765")
    assert pool.online is True and (pool.nodes, pool.jobs, pool.ledger) == ([], [], [])
