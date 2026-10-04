# /compute

Pooling engine: pipeline-parallel LoRA fine-tunes across Apple Silicon Macs on a LAN. Contributors run a headless agent; one coordinator schedules stages, records usage, and checks work.

Accounts, credits, and grants are later. This build is the Mac engine plus a local app.

## Install

Apple Silicon Mac, Python 3.11+, [uv](https://docs.astral.sh/uv/):

```bash
cd slashcompute
uv sync --group dev
```

## Quick start

Install a double-clickable Mac app (once per machine):

```bash
uv sync --group dev
./scripts/macos/install_app.sh
open ~/Applications/compute.app
```

That writes `~/Applications/compute.app`. It opens a `/compute` window (not Safari, not Terminal). Closing the window does not stop the pool.

From this repo you can also run:

```bash
uv run slashcompute
```

The window has four sections in the sidebar:

- **Pool**: on the host Mac pick **Host pool** → **Start hosting** and copy the LAN address. On the others pick **Join pool**, paste that address or press **Find on LAN**, then **Connect**.
- **Contributions**: set the GPU share and the credit split (kept vs. given to grants), then **Start contributing**. Shows FLOPs given, estimated credits (1:1 with FLOPs) and your rank.
- **Usage**: submit a LoRA fine-tune (the JSONL is uploaded from this Mac) and follow, or cancel, jobs live.
- **Grants**: browse, fund and request community grants, with an admin review queue and a live contributor leaderboard. Grants are sample data until accounts return.

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

## LLMs (chat with a model split across Macs)

The **05 LLMs** tab chats with a GGUF model whose layers are split across the pool by llama.cpp RPC: `rpc-server` on workers, `llama-server` on one head Mac. It runs next to fine-tuning. When a Mac starts training, its LLM work drains and moves elsewhere.

Each Mac that serves LLMs needs llama.cpp built with RPC:

```bash
brew install cmake
./scripts/build_llama.sh
```

Then use the **Serve LLMs on this Mac** card (memory, models folder, head or worker) and press Start serving. Add models with **Upload GGUF**. The file is pushed to every head-capable Mac, or you can drop `.gguf` files into a head's models folder (default `~/models`). `./scripts/fetch_smoke_model.sh` grabs a tiny one.

From the terminal:

```bash
uv run python -m slashcompute.inference.node start --url http://192.168.1.10:8765
curl http://192.168.1.10:8765/v1/models
```

The coordinator also serves an OpenAI-compatible `POST /v1/chat/completions` (SSE streaming).

- **Transport:** `direct` (default) has heads reach workers' RPC servers on the LAN. `relay` (`--inference-transport relay`, or the toggle on the host) tunnels RPC through the coordinator over WebSockets, so Macs anywhere can join with no open ports. Set `SLASHCOMPUTE_INF_TOKEN` on the coordinator and nodes when exposing it to the internet.
- **Credits:** same FLOP book as training. Prompt tokens count at `2 × params` (plus attention) per token. Generated tokens are memory-bound, so they are weighted by prompt speed ÷ generation speed (clamped 1–50). An hour of serving then earns about what an hour of training does. Hosts are paid only what a signed-in chatter's reservation covers. Anonymous chats are free on LAN pools and still logged in the ledger. Public pools require an account session, accepted terms, and sufficient credits; the shared inference secret does not replace account authentication.
- **Tunables:** every setting can be overridden with `SLASHCOMPUTE_INF_<FIELD>` (see `src/slashcompute/inference/config.py`).

## Public-pool access

With `SLASHCOMPUTE_PUBLIC_POOL=1`, only a job's owner or an administrator can cancel it or download its output adapter. Dataset and checkpoint downloads also allow authenticated contributors currently assigned to that job. Checkpoint uploads and verification transfers require the matching assignment. Agents use `--session-token` (or `SLASHCOMPUTE_SESSION`) for both registration and HTTP transfers, including sandboxed workers.

Public users submit training datasets through `POST /jobs/upload` (the app's Usage form). Submitting coordinator-local dataset paths through `POST /jobs` is restricted to administrators. Temporary uploads are removed after submission, and rejected reservations discard their dataset copies.

Changing the pool address, GPU share, or session restarts the training agent after it drains. If it is still stopping, the app reports that settings have not yet been applied; start again after the current work finishes. Existing agents without saved argument metadata rejoin once after upgrading.

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
4. Each step is metered (FLOPs, memory, time).
5. `stop` drains after the current step; a crash resumes from the last complete checkpoint.
