#!/usr/bin/env bash
# A session's raw artifacts to the Hugging Face dataset, under <version set>/<session>/, with results/README.md (the
# dataset card) as its README: the copy that outlives the rental. The grid runs it after every unit when DA_SYNC=1, and session.sh runs
# it once more over the whole session when DATASET is set, so terminating the instance never loses data. VERSIONS is
# the grid's version set (anatomy.grid and session.sh pass it); there is no default, so a session never lands under
# another version set's directory. The dataset repo: DATASET=<org>/<name>.
#
#   SESSION=S2-20261014 VERSIONS=2026-09 scripts/sync_raw.sh [unit-subpath]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${SESSION:?set SESSION}" "${DATASET:?set DATASET, e.g. zhebrak/decode-anatomy}"
: "${VERSIONS:?set VERSIONS to the version set of the grid (its versions key)}"
SRC=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}/$SESSION/${1:-}
# Nsight exports hold the traced processes' environment (a cloud API key, SSH addresses): blanked here before every
# upload, and nothing goes up while anything secret-looking is left (an environment entry, the same name as a JSON or
# YAML key, a token's shape; .gz read decompressed). .nsys-rep files cannot be scrubbed: never uploaded.
.venv/bin/python -m servebench.nsys.scrub "$SRC" || { echo "sync: refused, secret-looking strings remain in $SRC" >&2; exit 4; }
# only the artifact types a session writes (results/README.md), so a file nothing expected never leaves the box;
# one --include per pattern: the CLI takes one value per flag and refuses a second pattern as an extra argument
.venv/bin/hf upload "$DATASET" "$SRC" "$VERSIONS/$SESSION/${1:-}" --repo-type dataset --quiet \
  --include "*.sqlite" --include "*.jsonl" --include "*.json" --include "*.json.gz" --include "*.csv" \
  --include "*.log" --include "*.txt" --include "*.yaml" --exclude "*.nsys-rep" \
  --commit-message "raw: $SESSION ${1:-(whole session)}"
[ -n "${1:-}" ] || .venv/bin/hf upload "$DATASET" results/README.md README.md --repo-type dataset --quiet \
  --commit-message "dataset card"
echo "sync: $SRC -> hf://datasets/$DATASET/$VERSIONS/$SESSION/${1:-}"
