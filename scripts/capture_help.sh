#!/usr/bin/env bash
# Pull every engine image the grid's version set names, record its digest and save its --help output as provenance:
#   configs/engines/<engine>/<version>/image.txt   repo:tag and the RepoDigest the runs pin
#   configs/engines/<engine>/<version>/help*.txt   what the pinned release accepts (flags were read from source at the
#                                                  tag; this is the release's own word, and what maintainers can check)
# The releases, images and versions come from the grid (GRID, configs/grid/v1.yaml: anatomy.grid --get images), so
# a new release needs only its configs and a version set. No GPU needed: --help only parses arguments (TRT-LLM
# excepted: see below). CAPTURE_ROOT redirects the output (provision.sh on the GPU box writes into the session's host/
# directory, never into the tracked tree, and runs the pinned digest, never a moved tag).
#
#   scripts/capture_help.sh [engine...]     default: every engine of the version set
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
grid=${GRID:-configs/grid/v1.yaml}
GPU=$(command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 && echo yes || echo no)
read -r -a DOCKER <<< "${DA_DOCKER:-docker}"

run() {  # run <image> <entrypoint> [args...] -> stdout+stderr, never fails the script
  local image=$1 entry=$2; shift 2
  local gpus=(); [ "$GPU" = yes ] && gpus=(--gpus device=0)
  "${DOCKER[@]}" run --rm "${gpus[@]}" ${RUN_ENV:+-e "$RUN_ENV"} --entrypoint "$entry" "$image" "$@" 2>&1 || echo "[exit $?]"
}

capture() {  # capture <engine> <version> <image tag> <pinned digest>
  local engine=$1 version=$2 image=$3 pinned=$4 dir=${CAPTURE_ROOT:-configs/engines}/$1/$2
  mkdir -p "$dir"
  if [ -n "${CAPTURE_ROOT:-}" ]; then  # on the GPU box: the digest provision.sh pulled
    image=$pinned
    cp "configs/engines/$engine/$version/image.txt" "$dir/image.txt"
  else
    echo "== $engine: pulling $image"
    "${DOCKER[@]}" pull -q "$image" >/dev/null || { echo "pull failed: $image"; return 1; }
    printf '%s\n%s\n' "$image" "$("${DOCKER[@]}" image inspect --format '{{index .RepoDigests 0}}' "$image")" \
      > "$dir/image.txt"
  fi
  case $engine in
    vllm)  # without a GPU, vLLM cannot infer a device for serve's config defaults: the CPU platform parses the same flags
      RUN_ENV=$([ "$GPU" = yes ] || echo VLLM_TARGET_DEVICE=cpu) run "$image" vllm serve --help=all > "$dir/help.txt" ;;
    sglang)
      run "$image" python3 -m sglang.launch_server --help > "$dir/help.txt" ;;
    trtllm)  # tensorrt_llm initialises NVML at import: --help needs a GPU, so without one record the source instead
      if [ "$GPU" = yes ]; then
        run "$image" trtllm-serve serve --help > "$dir/help.txt"
      else
        run "$image" bash -c 'P=/usr/local/lib/python3.12/dist-packages/tensorrt_llm
          echo "# no GPU: source listing in place of --help (trtllm-serve options, then LLM-API fields)"
          grep -n -E "@click.option\(\"--[a-z_]+" $P/commands/serve.py
          grep -n -E "^class |^    [a-z_][a-z_0-9]*: " $P/llmapi/llm_args.py' > "$dir/help.txt"
      fi ;;
    llamacpp)  # the server, and the L0 runner the grid also drives
      run "$image" /app/llama-server --help > "$dir/help.txt"
      run "$image" /app/llama-batched-bench --help > "$dir/help-batched-bench.txt" ;;
  esac
  echo "== $engine: $(tail -1 "$dir/image.txt")"
}

wanted=" ${*:-} "
while IFS=$'\t' read -r engine version image pinned; do
  [ -n "${*:-}" ] && [[ $wanted != *" $engine "* ]] && continue
  capture "$engine" "$version" "$image" "$pinned"
done < <(.venv/bin/python -m anatomy.grid "$grid" --get images)
