#!/usr/bin/env bash
# Session start, on the GPU box, from ~/decode-anatomy (scripts/box.sh provision runs it). Idempotent: a second run
# skips what is done. Writes the host's provenance to $DA_RAW_ROOT/<session>/host/ and stops on any gate it fails.
#
#   SESSION=S1-20261007 NSYS_VERSION=2026.5.1 scripts/provision.sh
#
# 1. The card gate: nvidia-smi must name an H100 SXM5 ("NVIDIA H100 80GB HBM3"); PCIe and NVL cards are refused. The
#    driver must be 580 or newer (the vLLM and SGLang images are CUDA 13.0 with no compat libraries), and Docker must
#    answer: both checked before anything is downloaded. With DATASET set, the Hugging Face dataset must exist private
#    and take an upload (created private when it does not exist): a session's exports are scrubbed, but nothing
#    goes public by accident; making the dataset public is a deliberate step after the session.
# 2. Provenance: nvidia-smi -q, lscpu, governor, driver, docker; what root allows (clock lock, perf counters).
# 3. Tools: the pinned Nsight Systems CLI (mounted into every container), the host venvs (uv), AIPerf.
# 4. Images by digest (configs/engines/*/*/image.txt), so every unit runs the bytes the provenance names. Docker 29
#    keeps its images under containerd's root (/var/lib/containerd), not the daemon's data-root: on a box with a small
#    root disk, put both on the data disk (a symlink for containerd's root works) beside the caches and the runs.
# 5. Models (scripts/fetch_models.sh) and bf16 GGUFs (scripts/make_gguf.sh).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${SESSION:?set SESSION, e.g. S1-$(date +%Y%m%d)}"
NSYS_VERSION=${NSYS_VERSION:-2026.5.1}
RAW=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}/$SESSION
mkdir -p "$RAW/host"

# the card and the SM clock the grid names (configs/grid/v1.yaml's gpu and clock_mhz), read with awk because the gate
# runs before uv and the venv exist on a fresh box; a grid that extends another must be given its base here
GRID=${GRID:-configs/grid/v1.yaml}
grid_value() { awk -v k="$1" -F': *' '$1 == k { sub(/ *#.*$/, "", $2); print $2; exit }' "$GRID"; }
CARD=$(grid_value gpu); CLOCK=$(grid_value clock_mhz)
[ -n "$CARD" ] || { echo "provision: $GRID names no gpu (an extending grid? GRID=its base)" >&2; exit 2; }
name=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
if [ "$name" != "$CARD" ]; then
  echo "provision: this box has '$name'; the grid runs on '$CARD' only" >&2; exit 1
fi
echo "provision: card ok ($name)"
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)
if [ "${driver%%.*}" -lt 580 ]; then
  echo "provision: driver $driver; the vLLM and SGLang images need 580 or newer (CUDA 13.0): choose another image" >&2
  exit 1
fi
read -r -a DOCKER <<< "${DA_DOCKER:-docker}"   # "sudo -n docker" where the user is not in the docker group
if ! "${DOCKER[@]}" info >/dev/null 2>&1; then
  echo "provision: '${DOCKER[*]} info' fails for $USER. Either sudo usermod -aG docker $USER, then tmux kill-server and" \
       "reconnect, or put export DA_DOCKER='sudo -n docker' in ~/.decode-anatomy.env" >&2
  exit 1
fi
echo "provision: driver $driver, docker ok"

nvidia-smi -q > "$RAW/host/nvidia-smi-q.txt"
nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$RAW/host/nvidia-smi-clocks.txt"
lscpu > "$RAW/host/lscpu.txt"
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$RAW/host/governor.txt" 2>/dev/null || echo n/a > "$RAW/host/governor.txt"
{ uname -a; "${DOCKER[@]}" version --format 'docker {{.Server.Version}}'; nproc; free -g; df -h "$HOME"; } \
  > "$RAW/host/system.txt" 2>&1
grep -h RmProfilingAdminOnly /proc/driver/nvidia/params > "$RAW/host/perf-counters.txt" 2>/dev/null || true

# What root allows: a clock lock that sticks (at the grid's clock_mhz), and perf counters for GPU-metrics sampling.
# Either may be refused by the provider; the study records which and goes on (they are cross-checks, not the
# measurement). A grid with no clock (clock_mhz null: a validation grid) asks for no lock.
if [ -z "$CLOCK" ] || [ "$CLOCK" = null ]; then
  echo "not required" > "$RAW/host/clock-lock.txt"
elif sudo -n nvidia-smi -i 0 -lgc "$CLOCK,$CLOCK" >/dev/null 2>&1; then
  # a lock the driver takes is not always the clock the card holds (an H100 PCIe VM held 1620 MHz of 1755 asked, idle
  # and busy alike): the ruler's own clock log reads what it holds, and the cells are read against that
  echo locked > "$RAW/host/clock-lock.txt"; sudo -n nvidia-smi -i 0 -rgc >/dev/null
else
  echo refused > "$RAW/host/clock-lock.txt"
fi
echo "provision: clock lock $(cat "$RAW/host/clock-lock.txt"); perf counters: $(cat "$RAW/host/perf-counters.txt" || true)"

# Nsight Systems, one pinned version for every engine, from NVIDIA's devtools repository (the CUDA apt repository
# stops at older releases): the .deb is unpacked under ~/opt, no root and no apt source needed. Checked 2026-10-01:
# nsight-systems-2026.5.1_2026.5.1.161-1_amd64.deb, 461 MB.
NSYS_DEB=${NSYS_DEB:-nsight-systems-${NSYS_VERSION}_2026.5.1.161-1_amd64.deb}
NSYS_DIR=$HOME/opt/nsight-systems/$NSYS_VERSION
if [ ! -x "$NSYS_DIR/bin/nsys" ]; then
  mkdir -p "$HOME/opt" && cd "$HOME/opt"
  curl -sfL -o nsys.deb "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/$NSYS_DEB"
  dpkg -x nsys.deb nsys-root && rm nsys.deb
  mkdir -p nsight-systems && ln -sfn "$HOME/opt/nsys-root/opt/nvidia/nsight-systems/$NSYS_VERSION" "$NSYS_DIR"
  cd - >/dev/null
fi
"$NSYS_DIR/bin/nsys" --version | tee "$RAW/host/nsys-version.txt"
touch "$HOME/.decode-anatomy.env"   # the box's settings: only DA_NSYS is ours to replace (DA_DOCKER and others stay)
{ grep -v '^export DA_NSYS=' "$HOME/.decode-anatomy.env" || true; echo "export DA_NSYS=$NSYS_DIR"; } \
  > "$HOME/.decode-anatomy.env.new" && mv "$HOME/.decode-anatomy.env.new" "$HOME/.decode-anatomy.env"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv sync --quiet --extra report --extra dataset
if [ -n "${DATASET:-}" ]; then  # every unit's sync would otherwise fail quietly, or create the dataset public
  .venv/bin/hf auth whoami >/dev/null 2>&1 || {
    echo "provision: DATASET is set but Hugging Face is not logged in here: .venv/bin/hf auth login" >&2; exit 1; }
  # the dataset private (created so when missing), and a write that works: a read-only token passes whoami
  .venv/bin/python - "$DATASET" "$SESSION" <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
repo, session = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
if not api.repo_info(repo, repo_type="dataset").private:
    sys.exit(f"provision: {repo} is public; the raw data goes to a private dataset: make it private first")
marker = f".provision/{session}"
api.upload_file(path_or_fileobj=b"ok\n", path_in_repo=marker, repo_id=repo, repo_type="dataset",
                commit_message=f"provision: write check ({session})")
api.delete_file(marker, repo_id=repo, repo_type="dataset", commit_message=f"provision: write check done ({session})")
print(f"provision: {repo} private and writable")
PY
fi
[ -x .venv-aiperf/bin/aiperf ] || { uv venv -q --python 3.12 .venv-aiperf && uv pip install -q -p .venv-aiperf "aiperf==0.13.0"; }

# the smallest image first, and a GPU container from it at once (with the pinned nsys mounted), so a box whose --gpus
# or nsys fails inside a container is refused in minutes, not after the other pulls, ~150 GB of models and the GGUFs
llama=$(sed -n 2p configs/engines/llamacpp/*/image.txt)
echo "provision: pulling $llama"; "${DOCKER[@]}" pull -q "$llama" >/dev/null
if ! "${DOCKER[@]}" run --rm --gpus device=0 -v "$NSYS_DIR:/opt/nsys:ro" --entrypoint /opt/nsys/bin/nsys "$llama" \
     --version >/dev/null 2>&1 || ! "${DOCKER[@]}" run --rm --gpus device=0 --entrypoint nvidia-smi "$llama" -L \
     >/dev/null 2>&1; then
  echo "provision: a GPU container fails here (--gpus, the NVIDIA container toolkit, or nsys inside it)" >&2
  exit 1
fi
echo "provision: a GPU container runs, nsys inside it"
for f in configs/engines/*/*/image.txt; do
  ref=$(sed -n 2p "$f"); echo "provision: pulling $ref"; "${DOCKER[@]}" pull -q "$ref" >/dev/null
done
"${DOCKER[@]}" images --digests > "$RAW/host/images.txt"
# A real image-owned SQLite artifact through the host uploader's scrub path. Default container users may be root;
# a host-only write test would miss the ownership failure until the first trace upload.
scrub_dir=$(mktemp -d "$RAW/host/.scrub-check.XXXXXX")
vllm=$(sed -n 2p configs/engines/vllm/*/image.txt)
"${DOCKER[@]}" run --rm --network none -v "$scrub_dir:/out" --entrypoint python3 "$vllm" -c '
import sqlite3
with sqlite3.connect("/out/check.sqlite") as db:
    db.execute("create table StringIds (id integer primary key, value text)")
    db.execute("insert into StringIds values (1, ?)", ("SSH_CONNECTION=provision-canary",))
'
.venv/bin/python -m servebench.nsys.scrub "$scrub_dir"
rm "$scrub_dir/check.sqlite"
rmdir "$scrub_dir"
echo "provision: image-owned export scrubbed by the host"
# with a GPU, every image's real --help (TRT-LLM's included), into the session's provenance, not the tracked tree
CAPTURE_ROOT="$RAW/host/help" scripts/capture_help.sh >/dev/null 2>&1 || true

scripts/fetch_models.sh
# bf16 GGUFs only for a grid that runs llama.cpp (the four-engine validation grid and the probes; the study does not,
# since 2026-10-03): 117 GB written and 13 minutes on a fresh box otherwise
if .venv/bin/python -c '
import sys
from anatomy.grid import expand, load
sys.exit(0 if any(u.engine == "llamacpp" for u in expand(load(sys.argv[1]))) else 1)' "$GRID"; then
  scripts/make_gguf.sh
else
  echo "provision: no llama.cpp unit in $GRID: no GGUFs"
fi
echo "provision: done; session $SESSION, provenance in $RAW/host"
