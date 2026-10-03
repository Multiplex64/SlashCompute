"""Find the llama.cpp binaries: ``$SLASHCOMPUTE_LLAMA_DIR``, then ``vendor/llama.cpp`` (scripts/build_llama.sh), then PATH."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]


def search_dirs() -> list[Path]:
    dirs = []
    if env := os.environ.get("SLASHCOMPUTE_LLAMA_DIR"):
        dirs.append(Path(env).expanduser())
    dirs += [REPO / "vendor/llama.cpp/build/bin", Path("~/.slashcompute/llama.cpp/build/bin").expanduser()]
    return dirs


def find_binary(*names: str) -> str:
    for d in search_dirs():
        for n in names:
            if (d / n).exists():
                return str(d / n)
    for n in names:
        if found := shutil.which(n):
            return found
    return names[0]


def find_llama_server() -> str:
    return find_binary("llama-server")


def find_rpc_server() -> str:
    return find_binary("ggml-rpc-server", "rpc-server")
