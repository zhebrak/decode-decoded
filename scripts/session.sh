#!/usr/bin/env bash
# One grid session on the box, what `box.sh grid` runs in tmux:
# 1. the full ruler when the session has no report yet (S1's runs before this, by box.sh ruler): bandwidth, launch
#    latency and the parser against its truth, so every session's cells are read against its own card, never another
#    rental's (anatomy.calibration.calibrations); then the session's provenance (host/, ruler/) to the HF dataset at once, so
#    a box lost mid-session still leaves them;
# 2. the grid;
# 3. quality control on the box, for a study grid: tidy's reruns-only pass, and one --redo round of the units it
#    flags, here on the same host (a rerun on another host would mix two hosts inside one cell). The round's list is
#    written once: a session resumed after an interruption reuses it, so no unit gets a second chance to pass. A
#    validation grid (the smoke grid, the slot probes) has no redo round: its units are never study data;
# 4. the whole session to the HF dataset once more (per-unit syncs can fail on a flaky network), job.log with it, under
#    the grid's own version set, bounded in time and tried twice. A final sync that fails exits 3, distinct from a
#    failed unit's 1: the box then holds the only full copy, so it must not be terminated before `box.sh results`.
#
#   GRID=configs/grid/v1.yaml SESSION=S2-20261014 DATASET=zhebrak/decode-anatomy scripts/session.sh S2 [grid args]
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
block=${1:?S1..S3}; shift
grid=${GRID:-configs/grid/v1.yaml}
: "${SESSION:?set SESSION}"
VERSIONS=$(.venv/bin/python -m anatomy.grid "$grid" --get versions) || exit 2
validation=$(.venv/bin/python -m anatomy.grid "$grid" --get validation) || exit 2
GRID=$grid
RAW=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}/$SESSION
export VERSIONS SESSION GRID
uploads=0; [ -n "${DATASET:-}" ] && [ "${DA_SYNC:-1}" != 0 ] && uploads=1

if [ ! -f "$RAW/ruler/ruler_report.json" ]; then
  # a ruler that does not pass leaves its fingerprint, and tidy then borrows S1's calibration only if this card
  # matches it (bandwidth and empty graph node within card_match, the same GPU and Nsight), and says so per cell
  scripts/ruler.sh || scripts/ruler.sh || echo "session: the ruler did not pass twice; the grid goes on" >&2
  # without even the fingerprint no cell of this session could be read against its own card: stop here, before the
  # grid spends the rental on cells that would be validation data at best
  if [ ! -f "$RAW/ruler/bandwidth.json" ] || [ ! -f "$RAW/ruler/launch_latency.json" ]; then
    echo "session: the ruler left no bandwidth or launch latency for this card; not running the grid" >&2
    exit 2
  fi
fi
if [ "$uploads" = 1 ]; then
  for part in host ruler; do
    [ -d "$RAW/$part" ] && { timeout 30m scripts/sync_raw.sh "$part" || echo "session: $part not uploaded yet" >&2; }
  done
fi

.venv/bin/python -m anatomy.grid "$grid" --session "$block" --session-id "$SESSION" --run "$@"
code=$?

# quality control here, on this host: the units tidy flags (a repeat or differential range above range_ok) run once
# more now, never on another box later, and never for a validation grid
if [ "$validation" = true ]; then
  echo "session: a validation grid: no quality-control redo round"
elif [ "$code" -ge 2 ]; then
  # the grid ended early (exit 2: the box broken, or a signal): a list written now would cover half a session, and a
  # resume would reuse it; the resumed session, run to its end, builds the list over every unit
  echo "session: the grid ended early (exit $code): no quality-control list now; the resumed session writes it" >&2
elif [ -f "$RAW/qc/reruns.csv" ]; then
  echo "session: reusing this session's reruns list ($RAW/qc/reruns.csv): one redo round per session"
elif ! .venv/bin/python -m anatomy.tidy "$RAW" --reruns-only --out "$RAW/qc"; then
  echo "session: the reruns check failed; the session's units stand as they are" >&2
fi
if [ "$validation" != true ] && [ "$code" -lt 2 ] && [ -f "$RAW/qc/reruns.csv" ]; then
  flagged=$(tail -n +2 "$RAW/qc/reruns.csv" | wc -l)   # tidy lists the grid's own units only, never noise or sentinels
  if [ "$flagged" -gt 0 ]; then
    echo "session: $flagged units flagged; one --redo round on this host"
    .venv/bin/python -m anatomy.grid "$grid" --session "$block" --session-id "$SESSION" --run \
      --redo "$RAW/qc/reruns.csv" "$@" || code=1
  fi
fi

if [ "$uploads" = 1 ]; then
  cp job.log "$RAW/job.log" 2>/dev/null || true   # the session's own log goes with its data
  if ! { timeout 3h scripts/sync_raw.sh || timeout 3h scripts/sync_raw.sh; }; then
    echo "session: FINAL SYNC FAILED. The box holds the only full copy of $SESSION: run box.sh results $SESSION" \
         "before terminating it" >&2
    code=3
  fi
fi
exit $code
