#!/usr/bin/env bash
# Provision a box that has no Docker (a container-hosted GPU, such as a cloud pod): each engine in its own uv venv at
# the pinned version, llama.cpp built from its tag, the pinned Nsight Systems unpacked, and every environment frozen
# into the session's host/ directory, which stands in for the image digest. DA_RUNTIME=native runs from
# these. Idempotent: a step whose result exists is skipped.
#
#   SESSION=N1-20261001 scripts/provision_native.sh [host vllm sglang trtllm llamacpp aiperf nsys models cuda-links]
#
# Engines and versions are the ones configs/engines pins. The environments live in $DA_ENVS (default ~/da-envs).
# DA_JOBS caps the llama.cpp build's parallelism (default 32, at nice 19): the box may be shared with CPU-timed work.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
: "${SESSION:?set SESSION}"
ENVS=${DA_ENVS:-$HOME/da-envs}
RAW=${DA_RAW_ROOT:-$HOME/decode-anatomy-runs}/$SESSION
HF=${HF_HOME:-$HOME/.cache/huggingface}
NSYS_VERSION=${NSYS_VERSION:-2026.5.1}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
mkdir -p "$ENVS" "$RAW/host"
steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(host nsys aiperf vllm sglang trtllm llamacpp models)
log() { echo "[provision_native $(date -u +%H:%M:%S)] $*"; }

link_bundled_cuda() {  # the CUDA toolkit an engine's wheels bring (nvidia/cu13) ships libfoo.so.13 but not the
  # libfoo.so a linker asks for, nor lib64: kernels compiled at first use fail with "cannot find -lcudart" without them
  local cuda
  for cuda in "$1"/lib/python3*/site-packages/nvidia/cu1*; do
    [ -d "$cuda/lib" ] || continue
    [ -e "$cuda/lib64" ] || ln -s lib "$cuda/lib64"
    for so in "$cuda"/lib/lib*.so.[0-9]*; do
      base=${so%%.so.*}.so
      [ -e "$base" ] || ln -s "$(basename "$so")" "$base"
    done
  done
}

venv() {  # venv <name> <pip args...>: a Python 3.12 venv with the given packages, frozen to host/
  local dir=$ENVS/$1; shift
  if [ ! -f "$dir/.done" ]; then
    log "venv $(basename "$dir"): $*"
    uv venv -q --python 3.12 "$dir"
    uv pip install -q -p "$dir" "$@"
    touch "$dir/.done"
  fi
  link_bundled_cuda "$dir"
  uv pip freeze -p "$dir" > "$RAW/host/freeze-$(basename "$dir").txt"
}

for step in "${steps[@]}"; do
  case $step in
    nsys)
      dir=$ENVS/nsight-systems-$NSYS_VERSION
      if [ ! -x "$dir/bin/nsys" ]; then
        log "nsys $NSYS_VERSION"
        deb=nsight-systems-${NSYS_VERSION}_2026.5.1.161-1_amd64.deb
        tmp=$(mktemp -d)
        curl -sfL -o "$tmp/nsys.deb" "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/$deb"
        dpkg -x "$tmp/nsys.deb" "$tmp/root" && rm "$tmp/nsys.deb"
        mv "$tmp/root/opt/nvidia/nsight-systems/$NSYS_VERSION" "$dir" && rm -rf "$tmp"
      fi
      "$dir/bin/nsys" --version | tee "$RAW/host/nsys-version.txt" ;;
    aiperf) venv aiperf-0.13.0 "aiperf==0.13.0"
            ln -sfn "$ENVS/aiperf-0.13.0" .venv-aiperf ;;  # where anatomy.grid looks for it (--aiperf), as provision.sh puts it
    vllm) venv vllm-0.30.0 "vllm==0.30.0" ;;
    sglang) venv sglang-0.5.20 --prerelease=allow "sglang==0.5.20" ;;   # it pins cuda-tile==1.6.0rc5
    trtllm)
      # MPI for its executor; the wheel is on NVIDIA's index
      command -v mpirun >/dev/null || { log "apt: openmpi"; apt-get -qq update && apt-get -qq install -y libopenmpi-dev openmpi-bin >/dev/null; }
      # its wheel is built against CUDA 13 (libcublasLt.so.13): torch 2.9.1 and its torchvision from PyTorch's cu130
      # index first, since PyPI's are the CUDA 12.8 builds, then tensorrt_llm on top (==2.9.1 is met by 2.9.1+cu130)
      dir=$ENVS/trtllm-1.2.1
      if [ ! -f "$dir/.done" ]; then
        log "venv trtllm-1.2.1: torch 2.9.1+cu130, tensorrt_llm 1.2.1"
        uv venv -q --python 3.12 "$dir"
        uv pip install -q -p "$dir" --index-url https://download.pytorch.org/whl/cu130 "torch==2.9.1" "torchvision==0.24.1"
        uv pip install -q -p "$dir" --extra-index-url https://pypi.nvidia.com "tensorrt_llm==1.2.1"
        touch "$dir/.done"
      fi
      uv pip freeze -p "$dir" > "$RAW/host/freeze-trtllm-1.2.1.txt"
      "$dir/bin/python" -c "import tensorrt_llm" ;;   # an import that fails here fails every unit later
    llamacpp)
      dir=$ENVS/llamacpp-v0.5.0
      if [ ! -x "$dir/bin/llama-server" ]; then
        log "llama.cpp v0.5.0, CUDA sm_90, with $($CUDA_HOME/bin/nvcc --version | tail -1)"
        command -v cmake >/dev/null || { apt-get -qq update && apt-get -qq install -y cmake build-essential >/dev/null; }
        rm -rf "$dir/src" && git clone -q --depth 1 --branch v0.5.0 https://github.com/ggml-org/llama.cpp "$dir/src"
        cmake -S "$dir/src" -B "$dir/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=90 \
          -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
        nice -n 19 cmake --build "$dir/build" -j "${DA_JOBS:-32}" --target llama-server llama-batched-bench >/dev/null
        mkdir -p "$dir/bin" && cp "$dir/build/bin/llama-server" "$dir/build/bin/llama-batched-bench" "$dir/bin/"
        uv venv -q --python 3.12 "$dir/py" && uv pip install -q -p "$dir/py" -r "$dir/src/requirements/requirements-convert_hf_to_gguf.txt"
      fi
      git -C "$dir/src" describe --tags --always > "$RAW/host/llamacpp-commit.txt"
      "$dir/bin/llama-server" --version > "$RAW/host/llamacpp-version.txt" 2>&1 || true ;;
    models)
      for m in ${MODELS:-Qwen/Qwen3-0.6B}; do
        log "model $m"
        # at its pin, refs/main pointed at it (scripts/fetch_models.sh says why)
        snap=$(awk -v m="$m:" '$1 == m {print $2}' configs/models/revisions.yaml)
        [ -n "$snap" ] || { echo "no pin for $m in configs/models/revisions.yaml" >&2; exit 2; }
        HF_HOME=$HF HF_XET_HIGH_PERFORMANCE=1 "$ENVS/vllm-0.30.0/bin/hf" download "$m" --revision "$snap" --quiet >/dev/null
        mkdir -p "$HF/hub/models--${m//\//--}/refs" && printf '%s' "$snap" > "$HF/hub/models--${m//\//--}/refs/main"
        name=${m##*/}; gguf=${DA_MODELS:-$HOME/models}/gguf/llamacpp-v0.5.0/$name-${snap:0:8}-bf16.gguf
        if [ ! -f "$gguf" ] && [ -d "$ENVS/llamacpp-v0.5.0/py" ]; then
          mkdir -p "$(dirname "$gguf")"
          # to .part and renamed on success: an interrupted conversion is never taken for a finished one
          "$ENVS/llamacpp-v0.5.0/py/bin/python" "$ENVS/llamacpp-v0.5.0/src/convert_hf_to_gguf.py" \
            "$HF/hub/models--${m//\//--}/snapshots/$snap" --outtype bf16 --outfile "$gguf.part" >/dev/null
          mv "$gguf.part" "$gguf"
          (cd "$(dirname "$gguf")" && sha256sum "$(basename "$gguf")" > "$(basename "$gguf").sha256")
        fi
      done ;;
    cuda-links) for d in "$ENVS"/*/; do link_bundled_cuda "$d"; done ;;
    host) log "host venv"; uv sync --quiet --extra report ;;  # the grid, tidy and session.sh run from it
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
nvidia-smi -q > "$RAW/host/nvidia-smi-q.txt"; lscpu > "$RAW/host/lscpu.txt"
echo "export DA_NSYS=$ENVS/nsight-systems-$NSYS_VERSION DA_ENVS=$ENVS DA_RUNTIME=native" > "$HOME/.decode-anatomy.env"
log "done"
