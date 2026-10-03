#!/usr/bin/env bash
# Build the pinned llama.cpp (llama-server + the RPC server) with RPC enabled, plus Metal or CUDA.
# Every Mac in one pipeline must run the same build: RPC breaks across versions. The inference node
# finds the binaries here (or set SLASHCOMPUTE_LLAMA_DIR, or put them on PATH).
#
#   ./scripts/build_llama.sh                 # ggml-org/llama.cpp @ b11160 into vendor/llama.cpp
#   LLAMA_REF=b11160 LLAMA_REPO=https://github.com/ggml-org/llama.cpp ./scripts/build_llama.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${LLAMA_REPO:-https://github.com/ggml-org/llama.cpp}"
REF="${LLAMA_REF:-b11160}"
DIR="${LLAMA_DIR:-$ROOT/vendor/llama.cpp}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required. macOS: brew install cmake   Ubuntu: sudo apt install cmake build-essential" >&2
  exit 1
fi

# Blobless clone: full commit history (llama.cpp's build number is the commit count) without
# downloading every old file version. A --depth 1 clone would report "build 1".
if [ ! -d "$DIR/.git" ]; then
  git clone --filter=blob:none --branch "$REF" "$REPO" "$DIR"
else
  if [ "$(git -C "$DIR" rev-parse --is-shallow-repository)" = "true" ]; then
    git -C "$DIR" fetch --unshallow --filter=blob:none origin
  fi
  git -C "$DIR" fetch --filter=blob:none origin "refs/tags/$REF:refs/tags/$REF"
  git -C "$DIR" checkout --quiet "$REF"
fi

FLAGS=(-DGGML_RPC=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF)
case "$(uname -s)" in
  Darwin) FLAGS+=(-DGGML_METAL=ON) ;;
  *) if command -v nvidia-smi >/dev/null 2>&1; then FLAGS+=(-DGGML_CUDA=ON); fi ;;
esac

cmake -S "$DIR" -B "$DIR/build" "${FLAGS[@]}"
# The RPC server target is `ggml-rpc-server` in b11160 (older builds: `rpc-server`).
cmake --build "$DIR/build" --config Release -j --target llama-server ggml-rpc-server 2>/dev/null \
  || cmake --build "$DIR/build" --config Release -j --target llama-server rpc-server

BIN="$DIR/build/bin"
RPC="$BIN/ggml-rpc-server"; [ -x "$RPC" ] || RPC="$BIN/rpc-server"
echo
echo "llama-server: $BIN/llama-server"
echo "rpc server:   $RPC"
"$BIN/llama-server" --version 2>&1 | tail -2
BUILD=$("$BIN/llama-server" --version 2>&1 | sed -nE 's/.*build ([0-9]+), commit ([0-9a-f]+).*/b\1-\2/p' | head -1)
echo
echo "Pinned build string: ${BUILD:-unknown}"
echo "Optional: pin it on the coordinator so mismatched Macs are refused:"
echo "  SLASHCOMPUTE_INF_PINNED_LLAMA_BUILD=${BUILD%%-*} uv run slashcompute-coordinator serve"
