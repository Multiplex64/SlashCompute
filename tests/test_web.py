from fastapi.testclient import TestClient

from slashcompute.launcher.controller import Launcher, LauncherSettings
from slashcompute.web.server import create_shell


class FakeProc:
    def __init__(self, pid: int, argv: list[str]) -> None:
        self.pid = pid
        self.argv = argv


class FakeHTTP:
    def __init__(self, health=None) -> None:
        self.health = health

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        if self.health is None:
            raise ConnectionError("down")
        class R:
            status_code = 200
            content = b'{"ok":true,"nodes":1,"jobs":0}'
            headers = {"content-type": "application/json"}
            def json(self_inner):
                return {"ok": True, "nodes": 1, "jobs": 0}
        return R()


def _shell(tmp_path, **kw):
    spawned = []
    n = {"p": 5000}

    def popen(argv, **_):
        n["p"] += 1
        proc = FakeProc(n["p"], argv)
        spawned.append(proc)
        return proc

    launcher = Launcher(
        home=tmp_path,
        python="/opt/venv/bin/python",
        popen=popen,
        http=kw.pop("http", FakeHTTP()),
        discover_fn=kw.pop("discover_fn", lambda timeout=5.0: "http://10.0.0.9:8765"),
        lan_ip_fn=lambda: "192.168.1.20",
    )
    app = create_shell(launcher)
    return app, launcher, spawned


def test_index_and_css(tmp_path):
    app, _, _ = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert b"COMPUTE" in r.content
        for tab in (b"contributions", b"usage", b"grants", b"pool"):
            assert b'data-tab="' + tab + b'"' in r.content
        assert b'data-ink="signal"' in r.content
        assert b'data-file="community"' not in r.content
        assert b">COM<" not in r.content
        js = c.get("/static/app.js")
        assert js.status_code == 200
        assert b"AUTH.sc" not in js.content
        assert b"/api/overview" in js.content
        assert b"GOOGLE" not in js.content
        assert c.get("/static/app.css").status_code == 200
        assert c.get("/static/fonts/archivo-black.woff2").status_code == 200


def test_settings_and_status(tmp_path):
    app, launcher, _ = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/settings", json={"mode": "join", "url": "10.1.2.3",
                                          "gpu_percent": 40, "finish": "signal"})
        assert r.status_code == 200
        body = r.json()
        assert body["mode"] == "join" and body["finish"] == "signal"
        assert body["url"] == "10.1.2.3"
        assert body["session_token"] == ""
        assert body["grant_split"] == 0
        got = c.get("/api/settings").json()
        assert got["gpu_percent"] == 40
        st = c.get("/api/status").json()
        assert st["lan_ip"] == "192.168.1.20"
        assert "carbon" in st["finishes"]
    assert launcher.load_settings().finish == "signal"


def test_start_stop_and_discover(tmp_path, monkeypatch):
    http = FakeHTTP()
    app, launcher, spawned = _shell(tmp_path, http=http)
    monkeypatch.setattr("slashcompute.launcher.controller.process_alive",
                        lambda pid: any(p.pid == pid for p in spawned))

    def health_after(url):
        if spawned:
            http.health = {"ok": True, "nodes": 0, "jobs": 0}
            return {"ok": True, "nodes": 0, "jobs": 0}
        return None

    launcher.poll_health = health_after
    with TestClient(app) as c:
        r = c.post("/api/start", json={"mode": "host", "gpu_percent": 50, "contribute": True})
        assert r.status_code == 200, r.text
        assert len(spawned) == 2
        found = c.post("/api/discover").json()
        assert found["url"] == "http://10.0.0.9:8765"
        c.post("/api/stop")


def test_join_start_requires_url(tmp_path):
    app, _, spawned = _shell(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/start", json={"mode": "join", "url": ""})
        assert r.status_code == 400
        assert spawned == []


def test_proxy_allows_health_and_blocks_other(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=FakeHTTP({"ok": True}))
    launcher.save_settings(LauncherSettings(mode="host"))
    with TestClient(app) as c:
        assert c.get("/api/coord/health").status_code == 200
        assert c.get("/api/coord/auth/me").status_code == 200
        assert c.get("/api/coord/verify/secret").status_code == 404
        assert c.get("/api/shell").json()["generation"] >= 2


class RoutedHTTP:
    """Answers GETs by path. ``routes`` maps path -> JSON payload."""

    def __init__(self, routes: dict) -> None:
        self.routes = routes

    def get(self, url: str, timeout: float = 1.0, params=None, headers=None):
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        if path not in self.routes:
            raise ConnectionError("down")
        payload = self.routes[path]

        class R:
            status_code = 200

            def json(self_inner):
                return payload
        return R()


POOL = {
    "/health": {"ok": True, "nodes": 2, "jobs": 2},
    "/nodes": [
        {"node_id": "me", "name": "Air", "matmul_tflops": 2.0, "memory_contrib_bytes": 8 << 30,
         "gpu_percent": 50},
        {"node_id": "b", "name": "Studio", "matmul_tflops": 20.0,
         "memory_contrib_bytes": 64 << 30, "gpu_percent": 80},
    ],
    "/jobs": [
        {"id": "old", "status": "completed", "steps": 10, "progress_step": 10, "submitted_at": 1},
        {"id": "new", "status": "running", "steps": 10, "progress_step": 4, "submitted_at": 2},
    ],
    "/ledger": [
        {"node_id": "me", "kind": "train", "flops": 4e12, "disputed_flops": 0.0},
        {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
    ],
}


def test_overview_offline(tmp_path):
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is False
    assert ov["pool"]["jobs"] == [] and ov["leaderboard"] == []
    assert ov["me"]["flops"] == 0 and ov["me"]["rank"] is None
    assert ov["status"]["lan_ip"] == "192.168.1.20"
    assert ov["status"]["models"][0].endswith("0.5B-Instruct-4bit")


def test_overview_online_ranks_this_mac(tmp_path):
    app, launcher, _ = _shell(tmp_path, http=RoutedHTTP(POOL))
    launcher.save_settings(LauncherSettings(mode="host", grant_split=25))
    launcher.paths.node_id_file.write_text("me\n")
    with TestClient(app) as c:
        ov = c.get("/api/overview").json()
    assert ov["pool"]["online"] is True
    assert [j["id"] for j in ov["pool"]["jobs"]] == ["new", "old"]
    assert ov["pool"]["jobs"][0]["progress"] == 0.4 and ov["pool"]["jobs"][0]["can_cancel"]
    assert ov["pool"]["capacity"]["macs"] == 2 and ov["pool"]["capacity"]["running"] == 1
    assert ov["me"]["flops"] == 4e12
    assert ov["me"]["credits"] == {"earned": 4e12, "kept": 3e12, "to_grants": 1e12}
    assert (ov["me"]["rank"], ov["me"]["of"]) == (2, 2)
    assert [n["is_me"] for n in ov["pool"]["nodes"]] == [True, False]


def test_stop_agent_endpoint_leaves_coordinator(tmp_path, monkeypatch):
    kills = []
    monkeypatch.setattr("os.kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr("slashcompute.agent.daemon._alive", lambda pid: True)
    app, launcher, _ = _shell(tmp_path)
    (tmp_path / "coordinator.pid").write_text("111\n")
    launcher.paths.pid_file.write_text("222\n")
    with TestClient(app) as c:
        assert c.post("/api/stop-agent").status_code == 200
    assert (222, 15) in kills and (111, 15) not in kills


def test_sample_grants_flow(tmp_path):
    T = 1e12
    app, _, _ = _shell(tmp_path, http=RoutedHTTP({}))
    with TestClient(app) as c:
        board = c.get("/api/grants?sort=least").json()
        assert board["sample"] is True and board["pledged"] == 0
        progress = [g["progress"] for g in board["grants"]]
        assert progress == sorted(progress)
        target = board["grants"][0]

        r = c.post(f"/api/grants/{target['id']}/fund", json={"amount": 10 * T})
        assert r.status_code == 200, r.text
        assert r.json()["pledged"] == 10 * T

        too_much = c.post(f"/api/grants/{target['id']}/fund", json={"amount": 10_000 * T})
        assert too_much.status_code == 400

        made = c.post("/api/grants", json={"title": "Parser", "summary": "For my thesis",
                                            "goal": 50 * T}).json()
        new = made["pending"][-1]
        assert new["title"] == "Parser"
        assert c.post("/api/grants", json={"title": "", "summary": "x", "goal": 1}).status_code == 400

        approved = c.post(f"/api/grants/{new['id']}/review", json={"approve": True}).json()
        assert any(g["id"] == new["id"] for g in approved["grants"])
        assert c.post(f"/api/grants/{new['id']}/review", json={"approve": True}).status_code == 400
