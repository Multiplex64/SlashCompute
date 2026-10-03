# /compute

Pooling engine: pipeline-parallel LoRA fine-tunes across Apple Silicon Macs on a LAN. Contributors run a headless agent; one coordinator schedules stages, records usage, and checks work.

Linux, VMs, credits, and the web app are later. This repo is the Mac engine.

## Install

Apple Silicon Mac, Python 3.11+, [uv](https://docs.astral.sh/uv/):

```bash
cd slashcompute
uv sync --group dev
```

## Quick start

Open the desktop window. Pick **Host pool** on one Mac and **Join pool** on the others.

```bash
uv run slashcompute
```

Host shows this Mac’s LAN IP. Join can type that address or press **Find on LAN**. Start/Stop spawn the same coordinator and agent the CLI uses. Closing the window leaves them running.

The rest of this README is the terminal equivalent.

## Same Mac (two fake nodes)

No second machine needed. Starts a coordinator and two agents on localhost:

```bash
uv run python scripts/run_local_cluster.py --agents 2
```

In another terminal, submit a job (dataset path is on the coordinator machine):

```json
{
  "kind": "lora_finetune",
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "dataset_path": "./train.jsonl",
  "steps": 50,
  "min_stages": 2
}
```

```bash
uv run slashcompute-coordinator submit job.json --url http://127.0.0.1:8765
uv run slashcompute-coordinator jobs --url http://127.0.0.1:8765
uv run slashcompute-coordinator nodes --url http://127.0.0.1:8765
uv run slashcompute-coordinator ledger --url http://127.0.0.1:8765
```

JSONL rows can be `{"text": "..."}`, `{"prompt": "...", "completion": "..."}`, or `{"tokens": [1,2,3]}`.

## Two Macs on a LAN

Prefer Ethernet or Thunderbolt. Wi-Fi works but the pipeline will be slower.

**Mac A — coordinator** (and optionally also an agent):

```bash
uv run slashcompute-coordinator serve
```

It listens on port 8765 and advertises via mDNS. State lives in `~/.slashcompute`.

**Each Mac — agent:**

```bash
uv run slashcompute-agent start --gpu-percent 50
```

The agent finds the coordinator on the LAN. To pin it:

```bash
uv run slashcompute-agent start --url http://192.168.1.10:8765 --gpu-percent 50
```

Allow incoming TCP on each agent's data port (default **9700**) in System Settings → Network → Firewall, or turn the firewall off on a trusted LAN.

Then submit a job from any machine that can reach the coordinator, with `"min_stages": 2` so it will not run on a single Mac.

```bash
uv run slashcompute-agent status
uv run slashcompute-agent stop
```

`--max-memory-gb` caps how much unified memory the node lends. `--no-sandbox` disables `sandbox-exec` around the worker (useful while debugging).

## Tests

```bash
uv run pytest -q
```

The two-process cluster test is marked `integration` and is included in a normal run. To run only that:

```bash
uv run pytest -m integration -q
```

## How a job runs

1. Agents register, pass a short GPU canary, and heartbeat.
2. The coordinator splits the model by layer, sized to each Mac's contributed memory.
3. Neighbouring stages open a TCP link and run a GPipe LoRA step: activations forward, gradients back.
4. Each step is metered (FLOPs, memory, time). Credits are derived later.
5. `stop` drains after the current step; a crash resumes from the last complete checkpoint.
