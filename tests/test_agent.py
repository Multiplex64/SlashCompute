import asyncio

import pytest

from slashcompute.agent.paths import resolve_data_host
from slashcompute.agent.throttle import sleep_s
from slashcompute.transport import Frame, LinkServer, TcpLink


def test_sleep_s_duty_cycle():
    assert sleep_s(2.0, 100) == 0.0
    assert sleep_s(2.0, 50) == pytest.approx(2.0)
    assert sleep_s(1.0, 25) == pytest.approx(3.0)
    assert sleep_s(1.0, 0) == 3600.0


def test_resolve_data_host_localhost():
    assert resolve_data_host(True) == "127.0.0.1"
    assert resolve_data_host(False) != ""


def test_benchmark_profile_shape():
    from slashcompute.agent.benchmark import benchmark
    from slashcompute.common.protocol import DeviceProfile

    d = benchmark(max_memory_bytes=512 * 1024 * 1024)
    assert isinstance(d, DeviceProfile)
    assert d.memory_contrib_bytes <= 512 * 1024 * 1024
    assert d.matmul_tflops > 0 and d.mem_bandwidth_gbps > 0
    assert d.chip


async def test_hello_reject_does_not_leak_expect():
    server = await LinkServer("127.0.0.1", 0, {"job_id": "secret", "epoch": 9}).start()
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    link = TcpLink(reader, writer)
    await link.send(Frame("hello", {"job_id": "nope", "epoch": 0}))
    ack = await link.recv(5)
    assert ack.kind == "hello_reject"
    assert "expect" not in ack.meta and "secret" not in str(ack.meta)
    await link.close()
    await server.close()


def test_agent_options_localhost_host(tmp_path):
    from slashcompute.agent.daemon import AgentOptions

    opt = AgentOptions(url="http://127.0.0.1:8765", home=tmp_path, localhost=True)
    assert opt.data_host == "127.0.0.1"
    assert opt.node_id == AgentOptions(url="http://127.0.0.1:8765", home=tmp_path).node_id


# ------------------------------------------------------------ coordinator outages

def _fake_profile(_max_bytes=None):
    from slashcompute.common.protocol import DeviceProfile

    return DeviceProfile(chip="fake", memory_total_bytes=16 << 30, memory_available_bytes=8 << 30,
                         working_set_bytes=12 << 30, memory_contrib_bytes=8 << 30, matmul_tflops=1.0,
                         mem_bandwidth_gbps=100.0)


async def _fake_coordinator(on_register):
    """A /ws/agent endpoint that calls ``on_register(ws, n)`` for the n-th registration."""
    import websockets

    count = {"n": 0}

    async def handler(ws):
        await ws.recv()                                    # Register
        count["n"] += 1
        await on_register(ws, count["n"])

    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}", count


async def test_agent_reconnects_after_the_coordinator_restarts(tmp_path, monkeypatch):
    """A coordinator restart (close 1012) used to end the agent with a traceback; it must rejoin."""
    from slashcompute.agent.daemon import AgentOptions, Daemon
    from slashcompute.common.protocol import Welcome, dump

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)
    rejoined = asyncio.Event()

    async def on_register(ws, n):
        await ws.send(dump(Welcome(node_id="x", heartbeat_interval_s=30)))
        if n == 1:
            await ws.close(code=1012, reason="service restart")
            return
        rejoined.set()
        await ws.wait_closed()

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    running = asyncio.create_task(daemon.run())
    try:
        await asyncio.wait_for(rejoined.wait(), 10)
        assert count["n"] == 2 and not running.done()
    finally:
        await daemon.shutdown()
        await asyncio.wait_for(running, 5)
        server.close()
    assert daemon.status == "stopped"


async def test_agent_stops_when_the_coordinator_refuses_it(tmp_path, monkeypatch):
    from slashcompute.agent.daemon import AgentOptions, Daemon

    monkeypatch.setattr("slashcompute.agent.daemon.benchmark", _fake_profile)

    async def on_register(ws, n):
        await ws.close(code=4003, reason="banned")

    server, url, count = await _fake_coordinator(on_register)
    daemon = Daemon(AgentOptions(url=url, home=tmp_path, localhost=True))
    try:
        with pytest.raises(SystemExit, match="banned"):
            await asyncio.wait_for(daemon.run(), 10)
    finally:
        server.close()
    assert count["n"] == 1                                 # no retry loop against a refusal


def test_chosen_memory_is_lent_even_above_what_is_free_but_not_past_the_working_set(monkeypatch):
    from slashcompute.agent import benchmark as bm

    monkeypatch.setattr(bm, "_memory", lambda: (16 << 30, 5 << 30))   # 16 GB Mac, 5 GB free
    monkeypatch.setattr(bm, "_matmul_tflops", lambda: 1.0)
    monkeypatch.setattr(bm, "_mem_bandwidth_gbps", lambda: 100.0)
    assert bm.benchmark().memory_contrib_bytes == 5 << 30            # automatic: what is free
    assert bm.benchmark(10 << 30).memory_contrib_bytes == 10 << 30   # the owner's choice
    assert bm.benchmark(3 << 30).memory_contrib_bytes == 3 << 30
    assert bm.benchmark(15 << 30).memory_contrib_bytes == 12 << 30   # capped at 75% of RAM
