---
license: cc-by-4.0
pretty_name: Decode-step anatomy, raw measurements
tags: [llm-inference, benchmark, vllm, sglang, tensorrt-llm, nsight-systems, h100]
---
# Decode-step anatomy: the tables, and the raw data's layout

`results/` holds the study's tables, built by `anatomy.tidy` from the sessions' raw data and read by
`anatomy.report`: where vLLM, SGLang and TensorRT-LLM spend a decode step on one H100 SXM, for Qwen3 0.6B to 32B at
batch 1 to 256. This file is also the README of the raw data when it is published as a dataset
(`scripts/sync_raw.sh`). Every number in the tables names the unit directory it came from.

## The tables

- `cells.parquet`, `cells.csv`: one row per cell (engine, arm, level, model, batch): step time, GPU busy share, MBU,
  the five-part decomposition, kernel time by class, every check's outcome, and the cell's sources (nested columns
  are JSON in the CSV);
- `noise.csv`: per session and noise cell, `processes`, `invalid` (readings left out: a stalled or cut-short wave,
  the GPU shared or slowed, a CPU quota, the wrong snapshot) with their `reasons`, `occupied_invalid` and
  `occupied_reasons` (graph readings left out), the mean `step_ms`, `sd` (between processes, relative), `range`,
  `start_to_end`, `within_range_max`, and from the graph traces `occupied`, `occupied_processes` and `occupied_sd`
  (between processes, absolute), and `process_steps` (each process's step, for the A/A check of the comparisons' rule);
- `agreement.csv`: the greedy-token check, one shape for every row: `kind` (identity: an arm against its own defaults;
  pair: two engines' defaults), `session`, `model`, `a` and `b` (engine-version/arm/level), `prompts`, `identical`,
  `median_first_divergence`, `flagged` (an arm that changed tokens; a pair with an engine that parted from every
  other at once) and `odd` (that engine);
- `sentinels.csv`: every sentinel reading, with its `level` (read against the cells and the noise of that level),
  `valid` and its `problem` where it is not (the same rules as a noise reading's), `drift` against its own session's
  opening reading (`drift_ok` is Gate G2; empty on opening and unlocked rows, False on an invalid closing one), `redo`
  (a redo round's sentinels) and `host_drift` (the session's opening against the earliest session's: a host
  difference, never a gate);
- `reruns.csv`: `session`, `unit`, `batches` and `ranges` (why), the units `anatomy.grid --redo` runs again;
- `tolerances-suggested.yaml` (with `--calibrate`): the values the rule gives, `set_from`, the `bands`,
  `overlap_model` where the PDL chain decided it, with `overlap_model_evidence` (the windows, how many prefer the
  corrected model, the smallest gap, the resolution: the timer's step and the truth's spread over the untraced
  passes), `g1_failures` and the `per_engine` statistics behind them;
- `patterns.json` (`anatomy.report.patterns --json`): `about` (what the data are: `dataset_status`, the source commit,
  cards, engines, models, batches, sessions and dates), the principles (each with the `cells` it read), every
  difference beyond noise, the cross-session pairs as `exploratory`, the chart `summaries`, and the count chance
  would give with the A/A rates (`stats.aa`: the noise block's processes, left two out, and the sentinels' end
  against start);
- `strips.json`, `cell-strips.json`: the kernel strips the page's opening figure and cell explorer draw, cut from the
  node traces.

## The raw data's layout

```
<version set>/<session>/<engine>-<version>/<arm>/<level>/<model>[-p<prompt>g<gen>]/<capture>/
    meta.json        the unit, its exact command (docker, or native from a pinned environment), the host fingerprint
                     (CPU, GPU, driver, nsys), the git commit
    status.json      ok, seconds, error; for a runner under a CPU quota, cpu (its whole run's quota periods, and how
                     many throttled it)
    driver.jsonl     one record per generation: tokens, wall_ns (the whole wave), span_ns (each stream's first token
                     to its last, the mean over streams: the differential's input), cadence estimate (step time),
                     profile windows, cpu (under a CPU quota: the quota periods the wave spanned, and how many
                     throttled it), a server's prompt_tokens (its own count of each prompt); the first record holds
                     the waves' settings; a timing unit's last records are its greedy probe (kind "greedy": each
                     prompt's 32 token ids, or the error)
    chunks-*.json.gz per-token arrival times (ns) of every sequence, long and short waves, for re-estimating step time
    aiperf-*/        the L2 client's raw exports, the same arrivals for a served unit, with client.log
    container.log    the engine's own log: resolved configuration, KV capacity
    clocks.csv       SM/memory clock, throttle reasons, temperature, power, once a second
    gpu-processes.csv every compute process on the GPU, once a second
    *.sqlite         Nsight Systems exports (graph mode: GPU busy time; node mode: kernel time), one per capture window
<version set>/<session>/.../<capture>.attempt-<n>/   an attempt set aside before the unit ran again: one that failed,
                     or one redone from reruns.csv (anatomy.grid --redo); provenance, not data
<version set>/<session>/host/      nvidia-smi -q, lscpu, image digests, every image's --help, model revisions (each the
                                   commit configs/models/revisions.yaml pins)
<version set>/<session>/ruler/     achievable bandwidth, launch latency and an empty graph node's cost, the parser's
                                   ground-truth check, the PDL chain (what a programmatic dependent launch leaves
                                   exposed) and the DRAM counters' scale, in ruler_report.json
<version set>/<session>/noise-<start|end>-p<n>/...   the noise block: the same cells as separate processes
```

- **Levels:** L0 is the forward loop alone, L1 the engine's offline API with its scheduler, L2 the OpenAI server.
- **Captures:**
  - `timing` is unprofiled;
  - `graph` and `node` are the two Nsight trace modes;
  - `hostload` is the graph trace again with the engine's CPU cores contended, the check that a slower host leaves
    the GPU's busy time per step where it was;
  - `metrics` is the DRAM counters, the check on the byte model;
  - `served` is the AIPerf client;
  - `runner` is L0.

## The workload every engine gets

- **Models:** Qwen3 0.6B to 32B and Llama 3 (unsloth's copies), each at the commit `configs/models/revisions.yaml`
  pins; bf16 weights everywhere.
- **Prompts:** at the engine API, random token ids, unique per sequence and per wave (no prefix can be reused);
  through a server, AIPerf's synthetic text of the same length, which each server's own count must match
  (`prompt_tokens`). 896 tokens; 8,064 in the long-context block.
- **Generation:** greedy (temperature 0), `ignore_eos`, exactly 288 tokens, no detokenisation at the engine API;
  prefix caching off everywhere.
- **KV cache:** bf16 on every engine (their `auto`).
- **Equivalent work:** every timing unit ends with eight natural-text prompts (`configs/correctness/prompts.json`,
  tokenised once with each family's pinned tokenizer), 32 greedy tokens each, after its last timed wave.
  `agreement.csv` sets them against each other:
  - an arm that changes no arithmetic (async or overlap scheduling off, MRV2 off, GPU sampling) must give its
    defaults' very tokens;
  - engines may part where bf16 lets them; one that parts from every other within two tokens, on most prompts, is
    flagged.
- **Every flag of every arm** is in `configs/engines/<engine>/<version>/arms.yaml`, each image's `--help` beside it,
  and each unit's exact commands are in its `meta.json`.

## Reading it

```bash
# one row per cell (step, GPU busy share, MBU, decomposition), with S1 (the ruler, the noise block, the sentinels) read
# for calibration only; it refuses to run without a measured bandwidth
uv run python -m anatomy.tidy <S2 dir> <S3 dir> --calibrate <S1 dir> --out results
# what stands out beyond the run-to-run noise, also written as patterns.json
uv run python -m anatomy.report.patterns results/cells.parquet --json results/patterns.json
```

The cell table names its sources as `<session>/<unit>[/<file>]` under the version set, so every number resolves to a
directory of the raw data. A few columns say how a row was read:
- `accepted` and `rejection`: whether the cell counts, and why not;
- `attempts`: how many times its units ran (the earlier ones set aside as `.attempt-<n>`);
- `rerun_ratio`: where a rerun superseded the cell's timing, its step over the superseded run's; the cell's noise is
  never taken below how far the two readings sit apart;
- `bandwidth_from`: the session whose bandwidth it was read at: its own, or S1's where it measured none;
- `model_revision`: the snapshot its unit loaded (rejected unless it is the pin in `configs/models/revisions.yaml`),
  `revision_problem` the reason where it is not;
- `dataset_status`: preview, validation (a validation grid, another card, or a calibration lent unchecked) or study;
- `version_set`, `written`: the dataset folder its files are under, and when its unit's provenance was written;
- `ruler_from`, `calibration_lent`: where a session's own ruler did not pass, the session whose calibration it was
  read with (" (unchecked)" where the session measured nothing of its own card), and whether node tracing's inflation
  was lent (the same GPU and Nsight); the session's own empty graph node always prices its launches;
- `eager_launch`: its graph trace launched no graph in whole steps (CUDA graphs off): its kernel time comes from that
  trace, no node-trace inflation is taken off it, and its launch term prices each kernel at the empty graph node's
  cost, the only per-launch cost the ruler measures;
- `copy_ms`, `copy_share`: copies and memsets outside any kernel, the anatomy's fifth part;
- `slots_problem`, `prefill_problem`: empty in the study (llama-server's, from the harness's validation grids);
- `sm_mhz`, `sm_mhz_unthrottled`, `slowdowns`, `clock_problem`: the median SM clock over the unit's samples, the
  median where no slowdown held it (what the lock held: beyond `clock_lock` of the lock the cell is rejected, the
  reason in `clock_problem`, which also names a ruler that priced the launches at another clock), and each slowdown's
  share of the busy samples (a power cap is reported, never a rejection);
- `capture_conditions`: per trace capture (graph, node, hostload, metrics), the conditions its unit ran under, checked
  against the timing's, and its `problem`; `hostload_steps`, `metrics_steps` and their `_row_ratio`: the whole steps
  each window held and whether an anchor was missed;
- `metrics_valid`, `metrics_problem`, `metrics_ruler_from`: whether the DRAM-counter capture can be read (its own
  session's ruler and card, whole steps, the counters present over a measured step) and why not; `overlap_model`: the
  PDL overlap model the launch term was priced with (`configs/tolerances.yaml`);
- `mbu_flag`: set where the step moved more bytes than the card was measured to reach, MBU and SOL then left empty
  (the byte model or the bandwidth is wrong for that cell; its step and busy share stand);
- `finish_spread_steps`, `trace_offset_steps`: how far apart the sequences finished, and how far the trace window was
  moved to sit at the timing window's context; `anchor` and `window_from_end`: what marked the steps in the cell's
  traces (graph launches, a once-a-step kernel or copy) and where the window sat, counted back from the capture's
  last anchor, which the page's strips cut the same way;
- `occupied`: the GPU busy share, the share of the step with GPU work in flight (kernels, copies, graph launches)
  from the graph-mode trace over the unprofiled step; not CUDA's occupancy (warps per multiprocessor) nor
  nvidia-smi's utilisation. The column keeps its first name;
- `occupied_source`: "trace" (the trace's own fraction) or "trace over the untraced step"; the busy share's noise and
  whether a host stall is resolved are computed from the noise tables, not stored;
- `dram_over_model`: empty where the metrics capture is not valid (`metrics_problem`): no `dram_counter_scale` from
  its session's ruler, too few whole steps, no counters over a measured step, or another card or clock than its
  ruler's.

## Caveats

- **Profiled runs and timed runs are separate.** Nsight traces are never used for step time.
- **A cell's uncertainty is the noise between processes** (`noise.csv`), not its repeats' range inside one process.
- **The host CPU is part of what is measured.** Every session records its host fingerprint and re-runs sentinel cells.

## Licence

The data (the tables here and every file of the raw data) are CC BY 4.0: use them with credit to Alexander Zhebrak.
The code that produced them is MIT (the repository's LICENSE); the page's typeface, Instrument Sans, is OFL.
