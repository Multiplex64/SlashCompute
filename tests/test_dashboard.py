import pytest

from slashcompute.launcher.dashboard import (
    PoolData, capacity, format_flops, job_card, leaderboard, node_flops, overview,
    split_credits, with_unit,
)

LEDGER = [
    {"node_id": "a", "kind": "train", "flops": 5e12, "disputed_flops": 1e12},
    {"node_id": "a", "kind": "verify", "flops": 1e12, "disputed_flops": 0.0},
    {"node_id": "b", "kind": "train", "flops": 9e12, "disputed_flops": 0.0},
]
NODES = [
    {"node_id": "a", "name": "Air", "matmul_tflops": 2.5, "memory_contrib_bytes": 8 << 30},
    {"node_id": "c", "name": "Studio", "matmul_tflops": 20.0, "memory_contrib_bytes": 64 << 30},
]


@pytest.mark.parametrize("value, text", [
    (0, "0"), (512, "512"), (1_500, "1.50 K"), (2.5e9, "2.50 G"), (1.24e12, "1.24 T"),
    (36.5e15, "36.5 P"), (123e12, "123 T"),
])
def test_format_flops(value, text):
    assert format_flops(value) == text


def test_with_unit_spacing():
    assert with_unit(1.24e12) == "1.24 TFLOPs"
    assert with_unit(512, "FLOP/s") == "512 FLOP/s"


def test_node_flops_subtracts_disputed_work():
    assert node_flops(LEDGER, "a") == pytest.approx(5e12)
    assert node_flops(LEDGER, None) == 0.0


def test_split_credits_is_one_to_one_with_flops():
    assert split_credits(10e12, 30) == pytest.approx(
        {"earned": 10e12, "kept": 7e12, "to_grants": 3e12})
    assert split_credits(-5, 200)["to_grants"] == 0.0


def test_leaderboard_ranks_by_net_flops_and_marks_me():
    rows = leaderboard(LEDGER, NODES, my_id="a")
    assert [(r["rank"], r["node_id"]) for r in rows] == [(1, "b"), (2, "a"), (3, "c")]
    assert [r["name"] for r in rows] == ["b", "Air", "Studio"]  # unknown names fall back to id
    assert [r["is_me"] for r in rows] == [False, True, False]


def test_capacity_and_job_card():
    jobs = [{"status": "running"}, {"status": "queued"}, {"status": "recovering"},
            {"status": "completed"}]
    assert capacity(NODES, jobs) == {"macs": 2, "tflops": 22.5, "memory_bytes": 72 << 30,
                                     "running": 1, "waiting": 2}
    assert job_card({"status": "running", "steps": 50, "progress_step": 10})["progress"] == 0.2
    done = job_card({"status": "completed", "steps": 50, "progress_step": 49})
    assert done["progress"] == 1.0 and done["can_cancel"] is False


def test_overview_without_a_registered_mac():
    ov = overview({"lan_ip": "x"}, PoolData(online=True, nodes=NODES, ledger=LEDGER), None, 50)
    assert ov["me"]["rank"] is None and ov["me"]["node"] is None
    assert ov["me"]["of"] == 3
    assert not any(n["is_me"] for n in ov["pool"]["nodes"])


def test_no_rank_until_this_mac_has_contributed():
    nodes = [{"node_id": "me", "name": "Air"}, {"node_id": "b", "name": "Bee"}]
    ov = overview({}, PoolData(online=True, nodes=nodes), "me", 0)
    assert ov["me"]["rank"] is None and ov["me"]["of"] == 2
