#!/usr/bin/env bash
# The ruler on the GPU box: ruler/run.py (launch latency, bandwidth, the DRAM counters, the synthetic busy workload
# traced and untraced, then ruler.validate), with the box's settings from provision*.sh.
#
# The version set and the SM clock come from the grid (GRID, configs/grid/v1.yaml by default), as the cells' do.
#
#   SESSION=S1-20261007 [GRID=configs/grid/v1.yaml] scripts/ruler.sh [--dry-run]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[ -f "$HOME/.decode-anatomy.env" ] && source "$HOME/.decode-anatomy.env"   # DA_NSYS (and DA_RUNTIME without Docker)
export GRID=${GRID:-configs/grid/v1.yaml}   # ruler.run reads its version set and clock with anatomy.grid.load
exec .venv/bin/python -m ruler.run "$@"
