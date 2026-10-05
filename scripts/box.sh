#!/usr/bin/env bash
# The rented GPU box, driven from the laptop: a box only runs committed code (git pull --ff-only; code is never edited or rsynced there), one job at a time
# (~/decode-anatomy/busy holds the job's pid), long jobs in tmux so a dropped ssh never kills a session, and raw data
# outside the clone (~/decode-anatomy-runs) so a run never dirties the tree it runs from.
#
#   BOX=<ssh destination> scripts/box.sh <command> [args]
#     deploy                     the current branch's commits to the box as a git bundle (no remote needed); a fresh
#                                clone on first use, a fast-forward after
#     pull                       git pull --ff-only on the box from its own origin; prints the commit
#     test                       the test suite on the box (pytest, the same suite as on the laptop)
#     provision SESSION          scripts/provision.sh: card gate, provenance, nsys, images, models, GGUFs   (tmux)
#     provision-native S [steps] scripts/provision_native.sh, for a box without Docker (validation only; MODELS=...)
#     ruler SESSION              scripts/ruler.sh: bandwidth, launch latency, the parser vs ground truth     (tmux)
#     grid SESSION S2 [args]     scripts/session.sh: the full ruler if the session has no report, the grid's
#                                session S2, then the whole session to the HF dataset when DATASET is set (tmux;
#                                DA_SYNC=1; GRID defaults to configs/grid/v1.yaml, configs/grid/refresh.yaml for a
#                                refresh)
#     status                     the running job's log tail, and units done per session
#     wait                       until the running job ends; exits with its exit code (provision ... && wait && grid)
#     results SESSION            rsync the session's raw data to ~/decode-anatomy-runs/SESSION here
#
# A tmux job's exit status lands in ~/decode-anatomy/exit.code; `status` shows it, `wait` returns it. With DA_NOTIFY
# set to an ntfy.sh topic of your own, the job posts "decode-anatomy <job>: exit <code>" there when it ends (nothing
# else: no host, no session). provision and grid refuse to run without DATASET (the HF dataset the raw data goes to)
# unless DA_SYNC=0 says so: a session without uploads lives only on a box that can be terminated. Stopping the
# instance is the provider's console's job, after the session's final sync (session.sh) or `results`.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
USAGE="usage: BOX=<ssh destination> $0 deploy | pull | test | provision S | provision-native S [steps] | ruler S | grid S BLOCKSESSION [args] | status | wait | results S"
[ -n "${BOX:-}" ] || { echo "$USAGE" >&2; exit 2; }
SSH="ssh -o ConnectTimeout=15"
REPO='~/decode-anatomy'

read -r -d '' PULL <<'REMOTE' || true
set -euo pipefail
cd ~/decode-anatomy
if [ -n "$(git status --short --untracked-files=no)" ]; then
  echo "box.sh: tracked files changed on the box; code is never edited there. git checkout -- . after pushing the fix" >&2
  git status --short --untracked-files=no >&2; exit 1
fi
holder=$(cat busy 2>/dev/null || true)
if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then echo "box.sh: a job (pid $holder) is running; not pulling" >&2; exit 1; fi
rm -f busy
git pull --ff-only --prune --quiet
echo "box.sh: ~/decode-anatomy at $(git describe --always --tags) ($(git log -1 --format=%s | cut -c1-60))"
REMOTE

need_dataset() {  # uploads on, or turned off on purpose
  [ -n "${DATASET:-}" ] || [ "${DA_SYNC:-1}" = 0 ] || {
    echo "box.sh: set DATASET (the HF dataset, e.g. DATASET=zhebrak/decode-anatomy), or DA_SYNC=0 to run with no" \
         "uploads at all" >&2; exit 2; }
}

job() {  # job <name> <command...>: claim, pull, run in tmux with the env file sourced, report the start
  local name=$1; shift
  printf -v cmd '%q ' "$@"
  local notify=${DA_NOTIFY:-}
  notify=${notify//[^A-Za-z0-9_-]/}   # an ntfy.sh topic: letters, digits, - and _ only
  $SSH "$BOX" bash -s <<<"$PULL
cat > ~/decode-anatomy-job.sh <<JOB
cd ~/decode-anatomy
echo \\\$\\\$ > busy
trap 'rm -f busy' EXIT
[ -f ~/.decode-anatomy.env ] && source ~/.decode-anatomy.env
export PATH=\\\$HOME/.local/bin:\\\$PATH
$cmd > job.log 2>&1; echo \\\$? > exit.code
echo \"exit \\\$(cat exit.code)\" >> job.log
[ -n \"$notify\" ] && curl -fsS -m 10 -d \"decode-anatomy $name: exit \\\$(cat exit.code)\" \"https://ntfy.sh/$notify\" >/dev/null 2>&1 || true
JOB
rm -f exit.code
tmux new -d -s da-$name 'bash ~/decode-anatomy-job.sh'
echo \"box.sh: $name started in tmux da-$name; BOX=$BOX scripts/box.sh status\""
}

case "${1:-}" in
  deploy)
    branch=$(git rev-parse --abbrev-ref HEAD)
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "box.sh: commit first; only commits deploy" >&2; exit 1; }
    bundle=$(mktemp --suffix=.bundle); git bundle create -q "$bundle" "$branch"
    scp -q -o ConnectTimeout=15 "$bundle" "$BOX:decode-anatomy.bundle" && rm -f "$bundle"
    $SSH "$BOX" bash -s <<<"set -euo pipefail
holder=\$(cat ~/decode-anatomy/busy 2>/dev/null || true)
if [ -n \"\$holder\" ] && kill -0 \"\$holder\" 2>/dev/null; then echo 'box.sh: a job is running; not deploying under it' >&2; exit 1; fi
if [ -d ~/decode-anatomy/.git ]; then git -C ~/decode-anatomy pull -q --ff-only ~/decode-anatomy.bundle $branch
else git clone -q -b $branch ~/decode-anatomy.bundle ~/decode-anatomy; fi
echo \"box.sh: ~/decode-anatomy at \$(git -C ~/decode-anatomy describe --always --tags)\"" ;;
  pull) $SSH "$BOX" bash -s <<<"$PULL" ;;
  test) $SSH "$BOX" bash -s <<<"$PULL
cd ~/decode-anatomy && .venv/bin/python -m pytest -q -p no:cacheprovider" ;;
  provision)  # GRID: the card gate, the clock and the models to fetch are the grid's (a validation grid names its card)
    need_dataset
    job provision env SESSION="${2:?session}" DATASET="${DATASET:-}" GRID="${GRID:-configs/grid/v1.yaml}" \
      scripts/provision.sh ;;
  provision-native)  # a box without Docker (a pod): validation only, never the study's runtime
    session=${2:?session}; shift 2
    job provision env SESSION="$session" MODELS="${MODELS:-Qwen/Qwen3-0.6B}" DA_JOBS="${DA_JOBS:-32}" \
      scripts/provision_native.sh "$@" ;;
  ruler) job ruler env SESSION="${2:?session}" GRID="${GRID:-configs/grid/v1.yaml}" scripts/ruler.sh ;;
  grid)
    session=${2:?session id}; block_session=${3:?S1..S3}; shift 3
    need_dataset
    job grid env DA_SYNC="${DA_SYNC:-1}" DATASET="${DATASET:-}" GRID="${GRID:-configs/grid/v1.yaml}" SESSION="$session" \
      scripts/session.sh "$block_session" "$@" ;;
  status) $SSH "$BOX" bash -s <<'REMOTE'
cd ~/decode-anatomy
holder=$(cat busy 2>/dev/null || true)
if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then echo "running (pid $holder)"; else echo "idle (last exit: $(cat exit.code 2>/dev/null || echo none))"; fi
tail -n 15 job.log 2>/dev/null || true
for s in ~/decode-anatomy-runs/*/; do
  echo "$(basename "$s"): $(grep -rl '"ok": true' "$s" --include=status.json 2>/dev/null | wc -l) units done, $(grep -rl '"ok": false' "$s" --include=status.json 2>/dev/null | wc -l) failed"
done
REMOTE
  ;;
  wait)  # polls from here, one short ssh a minute, so a long job survives dropped connections
    misses=0
    while :; do
      state=$($SSH -o ServerAliveInterval=30 "$BOX" 'cd ~/decode-anatomy 2>/dev/null || exit 0
if [ -f exit.code ]; then echo "done $(cat exit.code)"
elif h=$(cat busy 2>/dev/null) && [ -n "$h" ] && kill -0 "$h" 2>/dev/null; then echo running
else echo none; fi' 2>/dev/null || echo unreachable)
      case $state in
        done*) $SSH "$BOX" 'tail -n 5 ~/decode-anatomy/job.log' 2>/dev/null || true; exit "${state#done }" ;;
        running) misses=0 ;;
        none) misses=$((misses + 1))   # a job just started has not claimed busy yet: a few polls' grace
              [ "$misses" -lt 5 ] || { echo "box.sh: no job is running on $BOX and none has finished" >&2; exit 1; } ;;
        *) echo "box.sh: $BOX unreachable; trying again" >&2 ;;
      esac
      sleep "${DA_WAIT_S:-60}"
    done ;;
  results)
    session=${2:?session}
    mkdir -p "$HOME/decode-anatomy-runs/$session"
    rsync -az --info=stats1 -e "$SSH" "$BOX:decode-anatomy-runs/$session/" "$HOME/decode-anatomy-runs/$session/" ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
