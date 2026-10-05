#!/usr/bin/env bash
# Download every model the grid names into the Hugging Face cache (safetensors and configs; no GGUF, no original
# checkpoints) at the commit configs/models/revisions.yaml pins, and point the cache's refs/main at it: every engine
# resolves `main` offline (HF_HUB_OFFLINE=1), and a download by commit writes no ref of its own, so without this a box
# with an older download would load that instead. A snapshot that is missing afterwards stops the script. About 125 GB
# for Qwen3 0.6B-32B plus 25 GB for the Llama check.
#
#   scripts/fetch_models.sh [repo ...]        default: every model in the grid (GRID, configs/grid/v1.yaml)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}  # hf_transfer is retired; Xet is the fast path
HUB=${HF_HOME:-$HOME/.cache/huggingface}/hub
LOG=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}/${SESSION:-adhoc}/host/model-revisions.txt
mkdir -p "$(dirname "$LOG")"
models=("$@")
[ ${#models[@]} -eq 0 ] && mapfile -t models < <(.venv/bin/python -m anatomy.grid "${GRID:-configs/grid/v1.yaml}" --get models)
for m in "${models[@]}"; do
  rev=$(.venv/bin/python -c 'import sys; from servebench.env import pinned; print(pinned(sys.argv[1]) or "")' "$m")
  [ -n "$rev" ] || { echo "fetch: $m has no pin in configs/models/revisions.yaml" >&2; exit 2; }
  echo "fetch: $m at $rev" >&2
  # one --exclude per pattern: the CLI takes one value each, and a second pattern would be read as a file name to
  # download, leaving the snapshot incomplete (AIPerf's offline tokenizer loader then refuses it)
  .venv/bin/hf download "$m" --revision "$rev" --exclude "*.gguf" --exclude "original/*" --exclude "*.pth" --quiet \
    >/dev/null
  dir=$HUB/models--${m//\//--}
  [ -d "$dir/snapshots/$rev" ] || { echo "fetch: $m: no snapshot $rev after the download" >&2; exit 1; }
  mkdir -p "$dir/refs" && printf '%s' "$rev" > "$dir/refs/main"
  echo "$m $rev"
done | tee -a "$LOG"
