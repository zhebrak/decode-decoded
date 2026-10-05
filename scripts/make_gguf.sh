#!/usr/bin/env bash
# bf16 GGUFs for llama.cpp, converted from the snapshot configs/models/revisions.yaml pins, which every other engine
# loads too, with the converter shipped in the pinned llama.cpp image (/app/convert_hf_to_gguf.py --outtype bf16), so
# llama.cpp moves the same bytes per step as everyone else (bf16 runs on llama.cpp's CUDA backend from Ampere).
# Output: ~/models/gguf/llamacpp-<version>/<name>-<commit8>-bf16.gguf (anatomy.grid.gguf), its sha256 beside it.
# A GGUF is reused only where its name matches the pin and the release and its checksum holds; a new pin or a new
# release is a new file. Each is written as .part and renamed when done, so an interrupted conversion is redone.
# Pushing the GGUFs to an HF repo saves the conversion in later sessions (GGUF_REPO=<user>/<repo>
# scripts/make_gguf.sh --push), once that repo exists; a downloaded one is checked against its sha256 as well.
#
#   scripts/make_gguf.sh [--push] [repo ...]     default: the models llama.cpp runs in the grid (GRID,
#                                                configs/grid/v1.yaml); its release and image come from the grid too
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
push=0; [ "${1:-}" = --push ] && { push=1; shift; }
grid=${GRID:-configs/grid/v1.yaml}
IFS=$'\t' read -r _ VERSION _ IMAGE < <(.venv/bin/python -m anatomy.grid "$grid" --get images | awk -F'\t' '$1 == "llamacpp"')
[ -n "${VERSION:-}" ] && [ -n "${IMAGE:-}" ] || { echo "gguf: $grid names no llama.cpp release" >&2; exit 2; }
read -r -a DOCKER <<< "${DA_DOCKER:-docker}"   # "sudo -n docker" where the user is not in the docker group
HF=${HF_HOME:-$HOME/.cache/huggingface}
OUT=$HOME/models/gguf/llamacpp-$VERSION
mkdir -p "$OUT"
models=("$@")
[ ${#models[@]} -eq 0 ] && mapfile -t models < <(.venv/bin/python -m anatomy.grid "$grid" --get models --engines llamacpp)
checked() { [ -f "$1" ] && [ -f "$1.sha256" ] && (cd "$(dirname "$1")" && sha256sum -c --quiet "$(basename "$1").sha256"); }
for m in "${models[@]}"; do
  rev=$(.venv/bin/python -c 'import sys; from servebench.env import pinned; print(pinned(sys.argv[1]) or "")' "$m")
  [ -n "$rev" ] || { echo "gguf: $m has no pin in configs/models/revisions.yaml" >&2; exit 2; }
  file=${m##*/}-${rev:0:8}-bf16.gguf; dest=$OUT/$file
  if [ -n "${GGUF_REPO:-}" ] && ! checked "$dest"; then
    .venv/bin/hf download "$GGUF_REPO" "llamacpp-$VERSION/$file" "llamacpp-$VERSION/$file.sha256" \
      --local-dir "$(dirname "$OUT")" --quiet >/dev/null 2>&1 || true
  fi
  if ! checked "$dest"; then
    [ -d "$HF/hub/models--${m//\//--}/snapshots/$rev" ] || { echo "gguf: $m $rev not fetched (fetch_models.sh)" >&2; exit 1; }
    echo "gguf: converting $m ($rev)"
    rm -f "$dest" "$dest.sha256"
    "${DOCKER[@]}" run --rm -v "$HF:/hf:ro" -v "$OUT:/out" --entrypoint python3 "$IMAGE" /app/convert_hf_to_gguf.py \
      "/hf/hub/models--${m//\//--}/snapshots/$rev" --outtype bf16 --outfile "/out/$file.part"
    mv "$dest.part" "$dest"
    (cd "$OUT" && sha256sum "$file" > "$file.sha256")
  fi
  [ "$push" = 1 ] && for f in "$dest" "$dest.sha256"; do
    .venv/bin/hf upload "${GGUF_REPO:?set GGUF_REPO}" "$f" "llamacpp-$VERSION/$(basename "$f")" --quiet
  done
  echo "gguf: $dest ($(du -h "$dest" | cut -f1))"
done
