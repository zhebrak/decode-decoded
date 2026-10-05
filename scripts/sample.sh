#!/usr/bin/env bash
# One engine and model's cells rebuilt from the raw data on a CPU laptop: its timing and trace units from the HF
# dataset with the session's ruler (a few tens of MB, .nsys-rep never uploaded), tidied into cells and drawn as the
# post's figures. No GPU, no rented box: what a reader runs to check a number on the page.
#
#   DATASET=zhebrak/decode-anatomy VERSIONS=2026-09 SESSION=S2-<date> scripts/sample.sh [engine-version] [model]
#   (defaults: vllm-0.30.0 Qwen3-0.6B; the folders are the cell explorer's "raw files" links)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${DATASET:?set DATASET, the HF dataset}" "${VERSIONS:?set VERSIONS, its version-set folder}"
: "${SESSION:?set SESSION, a study session in it}"
unit=${1:-vllm-0.30.0}; model=${2:-Qwen3-0.6B}
out=${SAMPLE:-$HOME/decode-anatomy-sample}
cells=$VERSIONS/$SESSION/$unit/defaults/*/$model
.venv/bin/hf download "$DATASET" --repo-type dataset --local-dir "$out" --quiet \
  --include "$VERSIONS/$SESSION/ruler/*" --include "$cells/timing/*" --include "$cells/served/*" \
  --include "$cells/graph/*" --include "$cells/node/*"   # one --include per pattern (the CLI takes one value each)
.venv/bin/python -m anatomy.tidy "$out/$VERSIONS/$SESSION" --out "$out/results"
.venv/bin/python -m anatomy.report.static --cells "$out/results/cells.parquet" --out "$out/figures"
echo "sample: $out/results/cells.csv and $out/figures/"
