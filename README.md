# Decode decoded

Where a decode step's time goes in vLLM 0.30.0, SGLang 0.5.20 and TensorRT-LLM 1.2.1 on one H100 SXM: Qwen3 0.6B
to 32B, batch 1 to 256, each engine's scheduler switches turned off one at a time, the OpenAI server against the
engine API. Step time comes from unprofiled runs; two Nsight Systems traces per cell of the engine-API grid split it into a
bandwidth floor, copies, kernel excess, launch cost and host stall; every difference is judged against measured
run-to-run noise.

- The write-up: [zhebrak.io/posts/decode-decoded](https://zhebrak.io/posts/decode-decoded/)
- The table: `results/` (312 cells, 306 accepted; `patterns.json` lists every difference beyond noise with the
  cells it rests on)

## What it found

- The GPU is not waiting on the host at the engines' defaults: at least 96% busy in every cell (97% at the engine
  API on Qwen3), batch 1 on 0.6B included. The host-bound regime appears only when an optimisation is switched off:
  with CUDA graphs off the step is 7 to 8.5× longer and the GPU 11 to 13% busy at 0.6B batch 1; with the overlap or
  async scheduler off the GPU is 62 to 90% busy.
- Busy is not fast. A 0.6B step at batch 1 runs at a quarter of what its bytes need at the card's measured
  bandwidth. The rest is kernel excess (small matmuls reach 49% of bandwidth at 0.6B, 95% at 32B) and launch cost,
  which CUDA graphs move onto the GPU rather than remove: about 1 µs per graph node, 337 to 382 nodes a step at
  0.6B, plus the gaps the trace shows between them.
- The three engines run the same cuBLAS matmul kernels; they differ in attention and norm by a constant per layer.
  TensorRT-LLM trails by 22% at 0.6B and 3% at 32B at batch 1 for that reason, and by 2 to 33% across the 0.6B batches.
- Graphs matter most; the overlap or async scheduler is worth 1.1 to 1.6× at batch 1 (1.5 to 1.6× at 0.6B);
  vLLM's Model Runner V2 moves the step by under 1% (the noise could hide 11%). The server sits within 0.7% of the
  engine API in every cell measured at both (the noise could hide 5% for vLLM, 1% for SGLang and 16% for
  TensorRT-LLM at batch 1), except that TensorRT-LLM's frontend falls behind the batch between about 5,500 and
  7,700 tokens per second on an 8-core host, at 32 to 64 streams.

## How it measures

**Defaults.** Each release's defaults with the batch cap (256), the token budget (8,192), the memory fraction
(0.9) and prefix caching (off) equalised, and TensorRT-LLM's decode graphs padded and captured every 8 up to 256 as
vLLM's and SGLang's are (`configs/engines/<engine>/<version>/arms.yaml`, every flag beside the image's `--help`).

**Three depths.** L0 is the forward loop alone (SGLang's `one_batch`), L1 the engine's offline API with its
scheduler (`AsyncLLM`, `sgl.Engine`, `tensorrt_llm.LLM`), L2 the OpenAI-compatible server driven by AIPerf. The
headline is L1.

**The workload.** 896 random token ids per sequence, unique per sequence and generation; 288 tokens generated,
greedy, `ignore_eos`, no detokenisation at L1; prefix caching off; an 8,064-token prompt as a context check. All
sequences of a batch are submitted at once, every output chunk timestamped. After the timed waves each unit generates
32 greedy tokens from eight fixed text prompts, and the ids are compared across engines and arms (`agreement.csv`).

**Step time** (`servebench/cadence.py`) is each sequence's time per token inside the window where every sequence is
decoding, 16 steps trimmed at each end, the median over sequences. Two warm-up generations, then three timed ones
alternated with three of 96 tokens: the differential (T(288) − T(96)) / 192 over each stream's own span must agree
with the cadence within the tolerance set from the first session. A sequence that falls behind the batch is a stall,
and the cell is rejected, never averaged.

**GPU busy** is the share of the step with a kernel, copy or graph launch in flight: the trace's busy time per step
over the unprofiled step, so the profiler's cost stays out. A graph-mode trace sees CUDA graphs as single ranges; a
node-mode trace sees every kernel. A hostload capture repeats the graph trace with a spinning process on each of the
engine's cores: the GPU's busy time per step must not move. It ran on 15 cells, all at 0.6B; under that contention
TensorRT-LLM's batch-256 step grew 95% and vLLM's 31% with busy time held.

**The decomposition** (`anatomy/decompose.py`): `step = floor + copies + kernel excess + launch + host stall`.
The floor is bytes per step (every matmul weight once, LM head included, plus every sequence's KV cache,
`servebench/physics.py`) over the card's measured bandwidth. Copies are memsets and copies outside any kernel.
Launch is kernels per step times the measured cost of a graph node, less what programmatic dependent launches hide
and never below the release the ruler measures, plus the gaps the node trace shows between kernels. Kernel excess is
what remains of kernel time, taken from the node trace less the per-kernel inflation tracing adds (the ruler measures
it), after the floor and the launch cost CUPTI records inside kernel durations. Host stall is (1 − busy) × step. The
split exists for the cells with both traces: the engine-API grid and the graphs-off arm, 186 of the 306 accepted.

**The ruler** (`ruler/`) runs before any engine in every session, at the locked SM clock: a synthetic workload with
a known busy fraction checks the trace parser to within a point; streaming bandwidth, launch latency, an empty graph
node's cost and the PDL chain's release are measured; the DRAM counters are calibrated against a stream of known
bytes.

**Noise** is measured between processes, not within one: the first session runs representative cells as three
separate processes at its start and at its end (`noise.csv`); every other cell borrows its group's spread, never
below 0.2%. Each later session's sentinels widen an engine's noise where the first session's does not hold (in S2,
vLLM's by 5.4× and TensorRT-LLM's by 3.5×). A difference counts beyond three times the combined noise, and `patterns.json` states how many
differences chance would give. Everything compared with a headline cell runs in the same session, back to back, model by model,
with the engines' order reversed on alternate models. Sentinel cells at both ends of every session bound the drift.

## Running it

No GPU is needed for the tests and the dry runs:

```bash
uv sync
uv run pytest
uv run python -m anatomy.grid configs/grid/v1.yaml --session S2 --dry-run   # every command of a session
```

The study needs a root VM with one H100 SXM (the grid refuses other cards), driver 580 or newer, Docker and about
400 GB of disk. Everything runs from the laptop through `scripts/box.sh`, whose header documents each command: the
box runs committed code shipped as a git bundle, one job at a time in tmux, raw data outside the clone. Uploads to a
Hugging Face dataset are optional (`DATASET=<org>/<name>`); otherwise `DA_SYNC=0`, and `box.sh results` fetches the
raw data before the box is stopped.

```bash
export BOX=<ssh alias> DA_SYNC=0
export SMOKE=SMOKE-$(date +%Y%m%d) S1=S1-$(date +%Y%m%d) S2=S2-$(date +%Y%m%d) S3=S3-$(date +%Y%m%d)
scripts/box.sh deploy

# smoke: the full ruler, then the study's grid as one validation session at Qwen3-0.6B
scripts/box.sh provision "$SMOKE" && scripts/box.sh wait && scripts/box.sh ruler "$SMOKE" && scripts/box.sh wait &&
GRID=configs/grid/smoke.yaml scripts/box.sh grid "$SMOKE" SMOKE --timeout 1500 && scripts/box.sh wait
scripts/box.sh results "$SMOKE"
uv run python -m anatomy.gate smoke --session ~/decode-anatomy-runs/"$SMOKE" --grid configs/grid/smoke.yaml

# S1: ruler, the noise block at both ends, bring-up, the scale block, the anchors; then gate G1 and the tolerances
scripts/box.sh provision "$S1" && scripts/box.sh wait && scripts/box.sh ruler "$S1" && scripts/box.sh wait &&
scripts/box.sh grid "$S1" S1 --timeout <3 x the smoke's slowest unit> && scripts/box.sh wait
scripts/box.sh results "$S1"
uv run python -m anatomy.gate g1 --session ~/decode-anatomy-runs/"$S1" --grid configs/grid/v1.yaml
uv run python -m anatomy.noise apply ~/decode-anatomy-runs/"$S1"/gate/tolerances-suggested.yaml
git commit -am "Tolerances set from $S1" && scripts/box.sh deploy

# S2: the main grid and everything compared with it; S3: long context, the second family, the DRAM counters
scripts/box.sh provision "$S2" && scripts/box.sh wait && scripts/box.sh grid "$S2" S2 --timeout <as S1> && scripts/box.sh wait &&
scripts/box.sh provision "$S3" && scripts/box.sh wait && scripts/box.sh grid "$S3" S3 --timeout <as S1> && scripts/box.sh wait
scripts/box.sh results "$S2"; scripts/box.sh results "$S3"
uv run python -m anatomy.gate g2 --session ~/decode-anatomy-runs/"$S2" --grid configs/grid/v1.yaml

make results SESSIONS="$HOME/decode-anatomy-runs/$S2 $HOME/decode-anatomy-runs/$S3" CALIBRATE=$HOME/decode-anatomy-runs/$S1
```

The post's figures are rendered from the table by `anatomy.report.static` (`uv sync --frozen --extra report`, then
`uv run python -m anatomy.report.static --cells results/cells.parquet --out figures`): four figures, light and dark
SVG each, in the page's typeface or another given with `--font` and `--font-dir`.

The gates are code (`anatomy.gate`). The smoke gate wants every planned unit with its evidence, the clock locked,
the DRAM counters calibrated, the anchors holding and the greedy probe recorded. G1 wants the ruler within a point of
its truth, the noise block complete from six processes with no engine above its band, and bring-up, anchors and
scale accepted. G2 wants every feasible cell recorded or its failure explained, the flagged units rerun, and every
sentinel's end within `sentinel_drift` of its start. An engine above a band fails G1 for that engine alone and is
then judged at its own noise.

A session is one host. Re-running a grid on the same box skips finished units whose recorded workload, commands and
settings match. On another instance, `scripts/fetch_session.sh` brings back what a lost box uploaded (when `DATASET`
was set) and prints the `--models` still to run, which run under a new session id, so their cells carry the drift
between sessions. A box whose clock cannot be locked is refused; `DA_ALLOW_UNLOCKED=1` overrides that for every
session alike. A new engine release is a refresh: `configs/grid/refresh.yaml` re-runs S2 for the engines that
changed, with every engine's sentinels (`scripts/new_release.sh` starts its configs).

## Layout

```
servebench/   the harness, free of study code: physics, cadence, results, containers, drivers, nsys parser
  engines/    arms and command rendering per engine release
  drivers/    L1 drivers, run inside the engine's image (standard library only)
anatomy/      the study: grid, served runner, metrics, decomposition, tidy (the cell table), noise, gates
  report/     the page, its charts, the patterns, the post's figures
ruler/        bandwidth, launch latency, the PDL chain, the synthetic-busy check and its validator
configs/      models and their pinned revisions, engine releases (arms, image digests, --help), grids, tolerances
scripts/      box.sh, provisioning, models, the ruler, optional sync to a dataset
tests/        pytest, with golden command lines and a real trace fixture
results/      the study's tables: cells, noise, agreement, sentinels, patterns.json
```

Every flag was read from source at the pinned tags and checked against each image's `--help`, kept under
`configs/engines/<engine>/<version>/`. Every command the grid can run is pinned by a golden file.

## Limitations

- One GPU and one host. The study's VM had eight physical cores; nothing was measured at another core count. The
  server results at 32 to 64 streams depend on it, and under contention the engine-API steps at 0.6B batch 256 did too.
- Dense bf16 Qwen3 only, with Llama 3 as a cross-check on two engines; no MoE, FP8, speculative decoding or tensor
  parallelism. One prompt length in the main grid.
- These three releases. Engine defaults change monthly.
- llama.cpp is not in the study: its v0.5.0 server was validated with the harness and left out (its many-slot cells
  are host-bound on this host). The harness keeps its paths, tested against `configs/grid/validation/four-engines.yaml`.
- Noise is borrowed per group, not measured on every cell; the launch term's range covers two ways of pricing
  overlapped launches, not a confidence interval.
