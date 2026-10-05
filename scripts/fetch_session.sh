#!/usr/bin/env bash
# A session back from the HF dataset into ~/decode-anatomy-runs/SESSION on this machine: a lost box's recovery (what it
# finished is on the Hub), or the data on another laptop. Then, for a grid session, which models it finished, and the
# --models argument that finishes the rest on a new host under a new session id.
# Needs the dataset extra (uv sync --extra dataset) and a Hugging Face login that can read the dataset.
#
#   DATASET=zhebrak/decode-anatomy scripts/fetch_session.sh S2-20261014 [S2]
#   (GRID=configs/grid/v1.yaml; VERSIONS defaults to the grid's version set)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
session=${1:?session id, e.g. S2-20261014}; block=${2:-}
: "${DATASET:?set DATASET, e.g. zhebrak/decode-anatomy}"
grid=${GRID:-configs/grid/v1.yaml}
VERSIONS=${VERSIONS:-$(.venv/bin/python -m anatomy.grid "$grid" --get versions)}
RAW=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
.venv/bin/hf download "$DATASET" --repo-type dataset --include "$VERSIONS/$session/*" --local-dir "$tmp" --quiet
[ -d "$tmp/$VERSIONS/$session" ] || { echo "fetch: nothing under $VERSIONS/$session in $DATASET" >&2; exit 1; }
mkdir -p "$RAW/$session"
rsync -a "$tmp/$VERSIONS/$session/" "$RAW/$session/"
echo "fetch: $DATASET/$VERSIONS/$session -> $RAW/$session"
if [ -n "$block" ]; then
  .venv/bin/python -m anatomy.grid "$grid" --session "$block" --session-id "$session" --progress
fi
