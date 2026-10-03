#!/usr/bin/env bash
# Download a tiny GGUF for the inference smoke test (and optionally the join-benchmark basket model)
# into ~/models, the inference node's default models folder.
# Repos/files can be overridden if the defaults move:
#   SMOKE_REPO=unsloth/Qwen3.5-0.8B-GGUF SMOKE_FILE=Qwen3.5-0.8B-Q4_K_M.gguf ./scripts/fetch_smoke_model.sh
set -euo pipefail

DEST="${MODELS_DIR:-$HOME/models}"
mkdir -p "$DEST"

fetch() {
  local repo="$1" file="$2"
  if [ -f "$DEST/$file" ]; then echo "already have $DEST/$file"; return; fi
  echo "downloading $repo/$file -> $DEST/$file"
  curl -fL --progress-bar "https://huggingface.co/$repo/resolve/main/$file" -o "$DEST/$file.part" \
    || { rm -f "$DEST/$file.part"; echo "failed: check the repo/file names (set SMOKE_REPO/SMOKE_FILE)" >&2; return 1; }
  mv "$DEST/$file.part" "$DEST/$file"
}

fetch "${SMOKE_REPO:-unsloth/Qwen3.5-0.8B-GGUF}" "${SMOKE_FILE:-Qwen3.5-0.8B-Q4_K_M.gguf}"
if [ "${WITH_BASKET:-0}" = "1" ]; then
  fetch "${BASKET_REPO:-unsloth/Qwen3.5-4B-GGUF}" "${BASKET_FILE:-Qwen3.5-4B-Q4_K_M.gguf}"
fi
