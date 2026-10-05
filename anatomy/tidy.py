"""Session directories in, one tidy row per cell out (results/cells.parquet and .csv): what the analysis, the page and
the post read. Every row names the unit directories its numbers came from.

    python -m anatomy.tidy ~/decode-anatomy-runs/S2-20261014 ~/decode-anatomy-runs/S3-20261021 --out results

Joins, per cell (engine, version, arm, level, model, prompt_len, gen_len, batch):
- the timing unit (L1 timing / L2 served) for step_ms and its checks (anatomy.metrics.timing_cells);
- graph unit for the GPU's busy time per step (occupancy) and the profiled step, the node unit for kernel ms per step
  and per class (servebench.nsys.parse.summarise). Windows are whole decode steps between step anchors (graph
  launches, or an arm's once-a-step kernel or copy: llama.cpp's copy to the host), counted back from the wave's last
  step and moved to the timing window's context (trace_offset); a trace without anchors gives fractions only, and its
  cell is rejected (anchor_check);
- physics at the measured achievable bandwidth (ruler/bandwidth.json in the cell's own session, else the calibrating
  session's: calibrations) and the decomposition;
- the metrics unit, where there is one: the bytes the DRAM counters saw per step (read + write throughput over the
  trace's own steps, scaled by the ruler's dram_counter_scale) against the byte model's bytes_per_step, as
  dram_over_model. Near 1 means MBU's numerator is right; above it, the step moves bytes the model does not count
  (activations, logits, re-reads that miss L2).
L1 profile windows are numbered in the order the driver opened them (one per batch, nsys --capture-range-end=repeat
writes <kind>.1.sqlite, <kind>.2.sqlite, ...); L2 windows are named <kind>-b<batch>.sqlite.
Every path a row names (sources, duplicate_sources) is <session>/<unit key>[/<file>], relative to the runs root, so
the table carries no machine's paths; resolve() turns one back into a path given the session roots.
"""
import argparse
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from statistics import median

from anatomy.calibration import bandwidth, calibrations
from anatomy.decompose import decompose, exposed_launch_ms
from anatomy.llama_log import llama_prefill, llama_slots
from anatomy.metrics import (
    BEYOND_K,
    OVERHEAD_OK,
    TOLERANCES,
    conditions_problem,
    physics,
    rejection,
    row_status,
    runner_cells,
    timing_cells,
)
from servebench import clocks
from servebench.cgroup import throttled_share
from servebench.engines.base import Release
from servebench.env import pinned
from servebench.nsys.capture import KINDS
from servebench.nsys.parse import FULL_BATCH, numbered_exports, summarise
from servebench.physics import H100_SXM, PEAKS
from servebench.results import (
    contention_for,
    keyed_by_block,
    read_jsonl,
    read_meta,
    read_status,
    resolve,
)
from servebench.runners import batched_bench_steps, one_batch_steps

KEY = ("engine", "version", "arm", "level", "model", "prompt_len", "gen_len")
TIMING = ("timing", "served")
TIMED = (*TIMING, "runner")  # every unit that times a step: the unprofiled sweeps and the L0 runner


def runner_timing(unit_dir, engine):
    """L0 step times: SGLang's one_batch result file, or llama-batched-bench's lines in the container log; under a CPU
    quota, the share of the whole run's periods it throttled (status.json: a runner has no waves to bracket)."""
    if engine == "sglang":
        f = unit_dir / "one_batch.jsonl"
        steps = one_batch_steps(f.read_text()) if f.exists() else {}
    else:
        f = unit_dir / "container.log"
        steps = batched_bench_steps(f.read_text(errors="replace")) if f.exists() else {}
    status = read_status(unit_dir)
    throttled = throttled_share([status]) if status else None
    return {b: {**t, "cpu_throttled": throttled} for b, t in runner_cells(steps).items()}


def units(roots):
    """(session, dir, unit dict, status, GPU name) for every unit directory under the session roots. The GPU is the
    one the unit's provenance recorded (meta.json host.gpu.name): peaks and floors are that card's, never assumed."""
    for root in map(Path, roots):
        for meta in sorted(root.rglob("meta.json")):
            if ".attempt-" in meta.parent.name:  # an attempt set aside by anatomy.grid (failed, or redone): provenance
                continue
            m = read_meta(meta.parent)
            gpu = ((m.get("host") or {}).get("gpu") or {}).get("name")
            yield root.name, meta.parent, m["unit"], read_status(meta.parent), gpu


def raw_waves(d, kind, batch, repeat):
    """A long or short wave's arrivals from its raw data, or None: the L1 driver's chunk file (chunks-b<B>-r<n>, -s<n>
    for a short one) or the L2 client's AIPerf raw export (aiperf-b<B>-r<n>, -s<n>)."""
    import gzip

    from servebench.aiperf import RAW, read_raw

    tag = "r" if kind == "long" else "s"
    chunks, raw = d / f"chunks-b{batch}-{tag}{repeat}.json.gz", d / f"aiperf-b{batch}-{tag}{repeat}" / RAW
    if chunks.exists():
        return [[tuple(c) for c in s] for s in json.loads(gzip.open(chunks, "rt").read())]
    return read_raw(raw)[0] if raw.exists() else None


def reestimated(d, recs):
    """The unit's records read again from their raw arrival times by the current servebench.cadence, so every session
    is read by one estimator, whatever version ran it: every wave's cadence (a long one's tail too), its stream span,
    and the differential rebuilt from the spans where both halves of every pair have raw data. A record without its
    raw data keeps what it was written with (older L1 sessions saved no short-wave chunks)."""
    from servebench.cadence import differential, estimate, stream_span_ns

    info = next((r for r in recs if r.get("kind") == "info"), {})
    argv = info.get("argv") or []  # sessions before the info record carried the waves' settings
    arg = lambda name, default: int(argv[argv.index(name) + 1]) if name in argv else default  # noqa: E731
    margin = info.get("margin") or arg("--margin", 16)
    short = info.get("short_gen_len") or arg("--short-gen-len", 96)
    out, spans = [], {}
    for r in recs:
        if r.get("kind") in ("long", "short"):
            seqs = raw_waves(d, r["kind"], r["batch"], r["repeat"])
            if seqs is not None:
                r = {**r, "cadence": estimate(seqs, margin), "span_ns": stream_span_ns(seqs)}
                if r["kind"] == "long":
                    r["cadence_tail"] = estimate(seqs, margin, after_tokens=short)
                spans[r["kind"], r["batch"], r["repeat"]] = (r["span_ns"], r.get("gen_len"))
        out.append(r)
    for i, r in enumerate(out):
        if r.get("kind") != "differential":
            continue
        pairs = [(spans.get(("long", r["batch"], k)), spans.get(("short", r["batch"], k)))
                 for k in range(len(r.get("step_ns") or []))]
        if pairs and all(lo and sh and all(lo) and all(sh) for lo, sh in pairs):
            out[i] = {**r, "step_ns": [differential(lo[0], sh[0], lo[1], sh[1]) for lo, sh in pairs]}
    return out


COMPAT = re.compile(r"Using CUDA [\d.]+ driver version (\d+(?:\.\d+)*) with kernel driver version (\d+(?:\.\d+)*)")


def cuda_compat(d):
    """The forward-compatible CUDA driver an NGC image switched to, from its banner in container.log ("590.44.01 over
    580.105.08"), None when it ran on the host's own: launches, and so the launch term, go through it."""
    log = Path(d) / "container.log"
    if not log.exists():
        return None
    with open(log, errors="replace") as f:
        m = COMPAT.search(f.read(200_000))  # the banner opens the log
    return f"{m.group(1)} over {m.group(2)}" if m else None


def clock_reference(roots):
    """{session: the SM clock its ruler held under the lock, or None}: what a unit's clock is read against, a reading
    of the card taken whether or not the ruler's report passed (its own clock log, ruler/clocks.csv, else the clock
    its report recorded, card.sm_mhz_held). A card may hold less than the lock asked (the H100 PCIe VM held 1620 MHz
    of 1755, idle and busy alike, no slowdown bit set), and the calibration is taken at what it held, so the lock
    asked for is the reference only where the ruler left no log. The gate's clock item reads the same figure."""
    out = {}
    for r in map(Path, roots):
        log, report = r / "ruler" / "clocks.csv", r / "ruler" / "ruler_report.json"
        held = ((clocks.summarise(log) if log.exists() else None) or {}).get("sm_mhz_median_unthrottled")
        if held is None and report.exists():
            held = (json.loads(report.read_text()).get("card") or {}).get("sm_mhz_held")
        out[r.name] = held
    return out


def unit_conditions(d, own=1, reference=None):
    """What the GPU was doing while a unit ran (servebench.clocks): its median SM clock and the median where no slowdown
    held it (what the lock held: a cell whose clock sat beyond clock_lock of the reference where nothing slowed it is
    rejected, clock_problem; the reference is the clock the session's ruler held under the same lock, else the lock
    asked for; a power cap is reported, never a rejection), the share of busy samples under each slowdown, whether
    another process shared the GPU (more GPU processes at once than the engine's `own`: TensorRT-LLM's LLM-API process
    and its MPI worker are two), and whether the box was declared shared. A CPU quota's throttling is the waves' own
    (servebench.cgroup), read per cell. And what it loaded and when: the model snapshot (rejected where it is not the
    pin) and the time its provenance was written."""
    meta = read_meta(d) or {}
    c = clocks.summarise(d / "clocks.csv") if (d / "clocks.csv").exists() else None
    n = clocks.processes(d / "gpu-processes.csv")
    rev, pin = meta.get("model_revision"), pinned((meta.get("unit") or {}).get("model") or "")
    lock, held = meta.get("sm_clock_lock_mhz"), (c or {}).get("sm_mhz_median_unthrottled")
    expected = reference or lock
    return {"sm_mhz": c and c["sm_mhz_median"], "sm_mhz_unthrottled": held,
            "slowdowns": (c or {}).get("slowdowns") or {},
            "gpu_shared": None if n is None else n > own, "shared_box": meta.get("idle_mode") == "activity",
            "runtime": meta.get("runtime", "docker"), "validation": bool(meta.get("validation")),
            "cuda_compat": cuda_compat(d), "model_revision": rev, "written": meta.get("written"),
            "version_set": meta.get("versions"),  # the dataset's top folder for this unit's files
            **({"sm_clock_lock_mhz": meta["sm_clock_lock_mhz"]} if "sm_clock_lock_mhz" in meta else {}),
            # the lock is judged where nothing held the clock down: under a power cap the clock sits below the lock by
            # design (0.9 of the busy samples at 14B on the pod), and the cap is reported (slowdowns), not a rejection
            **({"clock_problem": f"clock log at {held:.0f} MHz where nothing slowed it, expected {expected:.0f} MHz "
                                 f"({'what the ruler held' if reference else 'the lock asked'})"}
               if lock and held and expected and abs(held / expected - 1) > TOLERANCES["clock_lock"] else {}),
            # a unit that loaded another snapshot than configs/models/revisions.yaml pins measured another model
            **({"revision_problem": f"loaded the model at {rev[:8]}, not its pin {pin[:8]}"}
               if rev and pin and rev != pin else {})}


def capture_problem(conditions, reference=None):
    """Conditions that disqualify a reading, plus known provenance mismatches with its timing reading."""
    reasons = [x for x in (conditions_problem(conditions), conditions.get("revision_problem"),
                           conditions.get("clock_problem")) if x]
    for key in ("gpu", "model_revision", "sm_clock_lock_mhz"):
        if reference is None or key not in conditions or key not in reference:
            continue
        known = key == "sm_clock_lock_mhz" or (conditions[key] is not None and reference[key] is not None)
        if known and conditions[key] != reference[key]:
            reasons.append(f"{key} differs from timing ({conditions[key]!r} vs {reference[key]!r})")
    return "; ".join(reasons)


def trace_problem(conditions, rec, slot=None, reference=None):
    """Why a trace capture's reading cannot stand, or "": its unit's conditions (capture_problem, against its timing
    unit's where given), a llama-server wave in slots that are not one run (slot: llama_slots' message for the batch)
    and profile streams that failed (rec: the batch's profile record). One rule for the cells (collect) and for the
    noise block's occupancy readings (anatomy.noise)."""
    return "; ".join(filter(None, [capture_problem(conditions, reference), slot,
                                   "profile streams failed" if (rec or {}).get("errors") else None]))


def written(d):
    """When a unit's provenance was written (meta.json; 0 for one without the field)."""
    return (read_meta(d) or {}).get("written", 0)


def chosen_units(roots):
    """(session, dir, unit, GPU, other units' keys measuring the same batches, the batches read from this unit) per
    (cell, capture): each batch from the latest unit (meta.json's `written`) that finished ok and ran it, else the
    latest that ran it at all. A rerun or a resumed session supersedes what it reran, whatever order the sessions are
    given in, and no session drops a batch another session measured. (A refresh's cells are another release, another
    cell: they never clash with the older release's; metrics.current_versions chooses between them.)"""
    groups = {}
    for session, d, u, status, gpu in units(roots):
        if keyed_by_block(u.get("block")) or status is None:
            continue
        groups.setdefault((tuple(u[k] for k in KEY), u["capture"]), []).append((session, d, u, status, gpu))
    for found in groups.values():
        taken = {}
        for f in sorted(found, key=lambda f: (not f[3].get("ok"), -written(f[1]))):
            for b in f[2]["batches"]:
                taken.setdefault(b, f)
        for f in found:
            mine = [b for b in f[2]["batches"] if taken[b] is f]
            if mine:
                others = [source(g[0], g[1]) for g in found if g is not f and set(g[2]["batches"]) & set(mine)]
                yield f[0], f[1], f[2], f[4], others, mine


def source(session, path):
    """<session>/<path under it>: how a row names a unit directory or a trace."""
    return f"{session}/{unit_key(path, session)}"




def windows(unit_dir, capture, level, batches):
    """{batch: sqlite path} for a profiled unit's capture windows: L2 exports are named per batch (exactly: a prefix
    match gave batch 1 the batch-16 trace); L1 exports are numbered in the order the driver opened its
    windows, so they pair with its profile records only when the counts match, and otherwise not at all."""
    if level == "L2":
        return {b: p for b in batches if (p := unit_dir / f"{capture}-b{b}.sqlite").exists()}
    numbered = numbered_exports(unit_dir, capture)
    profiled = [r["batch"] for r in read_jsonl(unit_dir / "driver.jsonl") if r["kind"] == "profile"]
    return dict(zip(profiled, numbered)) if len(profiled) == len(numbered) else {}


def release_for(unit):
    """The unit's engine release (configs/engines), or None when its arms file is gone."""
    try:
        return Release.load(unit["engine"], unit["version"])
    except FileNotFoundError:
        return None


def anchor_for(unit):
    """What marks a step in this unit's traces: graph launches, or the arm's once-a-step kernel or copy."""
    r = release_for(unit)
    return r.arm(unit["arm"]).anchor if r and unit["arm"] in r.arm_names else "graph"


def own_processes(unit):
    """How many GPU processes the engine runs itself (configs/engines gpu_processes; 1 unless it says)."""
    r = release_for(unit)
    return int(r.extra.get("gpu_processes", 1)) if r else 1


STEP_NOISE = TOLERANCES["step_noise"]  # run-to-run noise between processes on a step time, relative
BUSY_INVARIANT = TOLERANCES["busy_invariant"]  # busy ms per step with the cores contended against the graph trace
HOSTLOAD_STRETCH = TOLERANCES["hostload_stretch"]  # the least stretch that tests it (and tracing's own)
ANCHOR_RATIO = tuple(TOLERANCES["anchor_ratio"])  # a trace's step period over the unprofiled step
MIN_STEPS = TOLERANCES["min_steps"]  # whole steps a window must hold for its per-step numbers to be read
MISSED_ANCHOR = TOLERANCES["missed_anchor"]  # a step's rows over the median step's: two steps under one anchor


def finish_spread(d, recs):
    """{batch: steps} how far apart a wave's sequences finish, in steps: the median over the unit's long waves of the
    spread of their last tokens' arrivals over the wave's step. Each sequence makes gen_len tokens at one rate, so they
    also started that far apart (vLLM and TensorRT-LLM mix prefill into decode: ~5 steps at batch 64; SGLang ~0)."""
    out = {}
    for r in recs:
        step = (r.get("cadence") or {}).get("step_ns") if r.get("kind") == "long" else None
        seqs = raw_waves(d, "long", r["batch"], r["repeat"]) if step else None
        ends = [s[-1][0] for s in seqs or () if s]
        if len(ends) > 1:
            out.setdefault(r["batch"], []).append((max(ends) - min(ends)) / step)
    return {b: median(xs) for b, xs in out.items()}


def trace_offset(anchor, spread):
    """Steps to move a trace window later so its middle sits at the timing window's context. The timing window is the
    middle of the span every sequence decodes in; a window counted back from the last anchor sits spread/2 steps
    earlier where the last anchor is the first sequence's end (graph launches and full-batch copies mark only the full
    batch's steps), and spread/2 later where it is the last sequence's end (a kernel every step runs)."""
    half = round((spread or 0) / 2)
    return half if anchor in FULL_BATCH else -half


def window_summary(path, record, anchor, later=0):
    """summarise() over the window the run placed (servebench.drivers.common.profile_window): profile_steps whole
    steps, counted back from the wave's last step, `later` steps later (trace_offset). A capture holding fewer anchors
    than that reaches back, or a batch with no profile record, has no such window: its steps are set to None, so
    anchor_check rejects it."""
    if not record or "window_from_end" not in record:
        return {**summarise(path, anchor=anchor), "steps": None}
    back = record["window_from_end"] - later
    s = summarise(path, anchor=anchor, skip=-back, steps=record["profile_steps"])
    if (s.get("anchors") or 0) < back:
        s["steps"] = None
    return s


def whole_steps(steps, row_ratio=None):
    """Whether a trace window holds MIN_STEPS whole steps, none of them MISSED_ANCHOR times the median step's activity
    rows (two steps under one anchor: servebench.nsys.parse.step_row_ratio). The part of anchor_check that needs no
    timing run: the hostload and metrics windows are judged by it alone."""
    return bool(steps) and steps >= MIN_STEPS and (row_ratio is None or row_ratio < MISSED_ANCHOR)


def anchor_check(period_ms, steps, step_ms, row_ratio=None):
    """Whether a trace's step anchors mark steps: its window holds MIN_STEPS whole steps, their period is the
    unprofiled step's within ANCHOR_RATIO, and no step holds MISSED_ANCHOR times the median step's activity rows (two
    steps under one anchor, a 1/64 miscount the period cannot see: servebench.nsys.parse.step_row_ratio). The trace
    checks itself against the timing run, so no token count from the driver or the client enters (TRT-LLM's tokens
    reach its driver many steps after the GPU made them, and a capture opened late once held 0, 18 and 36 steps of the
    64 counted). False without anchors; None without a step."""
    if not steps or (row_ratio is not None and row_ratio >= MISSED_ANCHOR):
        return False
    if not period_ms or not step_ms:
        return None
    return whole_steps(steps, row_ratio) and ANCHOR_RATIO[0] <= period_ms / step_ms <= ANCHOR_RATIO[1]


def trace_fields(capture, s):
    """What one window's trace summary (window_summary) adds to its cell, by capture kind."""
    ms = lambda ns: (ns or 0) / 1e6 or None  # noqa: E731
    per_step = lambda n: n / s["steps"] if s.get("steps") and n is not None else None  # noqa: E731
    if capture == "hostload":  # the same GPU work, the host slowed by CPU contention: the busy-time check
        return {"busy_ms_hostload": ms(s.get("occupied_ns_per_step")),
                "profiled_step_ms_hostload": ms(s.get("step_ns")),
                "hostload_steps": s.get("steps"), "hostload_row_ratio": s.get("step_row_ratio")}
    if capture == "metrics":  # DRAM counters over the trace's own steps: bytes the GPU moved
        dram = s.get("dram") if s.get("steps") else None
        return {"dram_pct": dram["dram_read_pct"] + (dram["dram_write_pct"] or 0.0) if dram else None,
                "dram_step_ms": ms(s.get("step_ns")), "metrics_steps": s.get("steps"),
                "metrics_row_ratio": s.get("step_row_ratio")}
    if capture == "graph":
        graph = {"occupied_trace": s.get("occupied"), "busy_ms": ms(s.get("occupied_ns_per_step")),
                 "profiled_step_ms": ms(s.get("step_ns")), "graph_launches_per_step": per_step(s.get("graph_launches")),
                 "graph_steps": s.get("steps"), "graph_window": s.get("window_source"),
                 "graph_row_ratio": s.get("step_row_ratio"),
                 # no graph launched in whole steps: every kernel went out on its own (CUDA graphs off)
                 "eager_launch": bool(s.get("steps")) and not s.get("graph_launches")}
        # with no graph launched (CUDA graphs off) the graph trace already holds every kernel as a row, which is what
        # a node trace adds: its kernel time, kernels and copies are read here, so a graphs-off cell has the five
        # parts without a node capture (collect lets a node capture's own win)
        eager = trace_fields("node", s) if s.get("steps") and not s.get("graph_launches") and s.get("kernels") else {}
        return graph | {"eager_kernels": {k: v for k, v in eager.items() if not k.startswith("node_")}} \
            if eager else graph
    return {"kernel_ms": ms(s.get("kernel_ns_per_step")), "kernels_per_step": per_step(s.get("kernels")),
            "overlapped_per_step": s.get("overlapped_per_step"), "overlap_ns": s.get("overlap_ns_mean"),
            "overlaps_ns": s.get("overlaps_ns"), "copy_ms": ms(s.get("copy_ns_per_step")),
            "node_steps": s.get("steps"), "node_step_ms": ms(s.get("step_ns")), "top_kernels": s.get("top_kernels"),
            "node_row_ratio": s.get("step_row_ratio"),
            "classes_ms": {c: v / 1e6 for c, v in (s.get("classes_ns_per_step") or {}).items() if v}}


def rerun_steps(dirs):
    """{batch: step ms} the runs a rerun superseded measured, newest first, read the way the kept one is: a rerun
    lands over its cell, and the cell keeps the ratio between the two (rerun_ratio), so a calmer second run is never
    published as if it were the only one (anatomy.report.derive.Noise widens the cell's noise by it). dirs: the
    superseded runs, a same-session --redo's set-aside attempt or another session's unit alike."""
    out = {}
    for d in sorted((d for d in dirs if d and (d / "driver.jsonl").exists()), key=written, reverse=True):
        for b, t in timing_cells(reestimated(d, read_jsonl(d / "driver.jsonl"))).items():
            if t.get("step_ms"):
                out.setdefault(b, t["step_ms"])
    return out


def finished_attempts(d):
    """A unit's set-aside attempts that finished ok: a same-session --redo moves a finished run to <dir>.attempt-<n>
    (anatomy.grid.set_aside), where units() never reads it; a failed attempt measured nothing to compare."""
    out = []
    for a in d.parent.glob(f"{d.name}.attempt-*"):
        if (read_status(a) or {}).get("ok"):
            out.append(a)
    return out


def collect(roots):
    """({cell key + batch: its fields}, {cell key + batch: {capture: source}}) from the chosen units: timing from the
    timing, served and runner units first (their finish spread places the trace windows), then the rest from each
    trace's window (trace_fields)."""
    cells, sources, spreads = {}, {}, {}
    held = clock_reference(roots)  # each session's clock under the lock, from its ruler: what a unit's is read against
    chosen = sorted(chosen_units(roots), key=lambda c: c[2]["capture"] not in TIMED)
    for session, d, u, gpu, duplicates, batches in chosen:
        key = tuple(u[k] for k in KEY)
        attempts = 1 + sum(1 for _ in d.parent.glob(f"{d.name}.attempt-*"))
        for b in batches:
            c = cells.setdefault((*key, b), {})
            c.setdefault("session", session)  # a cell measured only by traces still names its session
            c.setdefault("ruler_clock_held_mhz", held.get(session))  # the clock its session's ruler held under the lock
            c["attempts"] = max(c.get("attempts", 1), attempts)
            if duplicates:
                c.setdefault("duplicate_sources", []).extend(duplicates)
        if u["capture"] in TIMING or u["capture"] == "runner":
            recs = [] if u["capture"] == "runner" else reestimated(d, read_jsonl(d / "driver.jsonl"))
            timing = runner_timing(d, u["engine"]) if u["capture"] == "runner" else timing_cells(recs)
            stack = next((r.get("stack") for r in recs if r.get("kind") == "info"), None)
            conditions = unit_conditions(d, own_processes(u), held.get(session))
            spread = finish_spread(d, recs)
            server = (u["engine"], u["level"]) == ("llamacpp", "L2")
            slots, stalls = (llama_slots(d, recs), llama_prefill(d, recs)) if server else ({}, {})
            superseded = rerun_steps([*finished_attempts(d), *(resolve(x, roots) for x in duplicates)]) \
                if u["capture"] in TIMING else {}
            for b, t in timing.items():
                if b not in batches:
                    continue
                before = superseded.get(b)
                cells.setdefault((*key, b), {}).update(t, session=session, gpu=gpu, stack=stack, **conditions,
                                                       finish_spread_steps=spread.get(b),
                                                       **({"slots_problem": slots[b]} if b in slots else {}),
                                                       **({"prefill_problem": stalls[b]} if b in stalls else {}),
                                                       **({"rerun_ratio": t["step_ms"] / before}
                                                          if before and t.get("step_ms") else {}))
                sources.setdefault((*key, b), {})[u["capture"]] = source(session, d)
                spreads[(*key, b)] = spread.get(b)
        elif u["capture"] in KINDS:
            conditions = {**unit_conditions(d, own_processes(u), held.get(session)), "gpu": gpu, "session": session}
            anchor = anchor_for(u)
            records = read_jsonl(d / "driver.jsonl")
            recs = {r["batch"]: r for r in records if r["kind"] == "profile"}
            slots = llama_slots(d, records) if (u["engine"], u["level"]) == ("llamacpp", "L2") else {}
            for b, path in windows(d, u["capture"], u["level"], batches).items():
                if b not in batches:  # an L1 unit's numbered windows cover its whole sweep: this unit's share only
                    continue
                later = 0 if u["capture"] == "metrics" else trace_offset(anchor, spreads.get((*key, b)))
                c = cells.setdefault((*key, b), {})
                capture_conditions = {**conditions}  # a hostload unit's throttling is its own spinners': not read
                capture_conditions["cpu_throttled"] = throttled_share([recs.get(b)]) if u["capture"] != "hostload" \
                    else None
                problem = trace_problem(capture_conditions, recs.get(b), slots.get(b),
                                        c if u["capture"] != "metrics" else None)
                capture_conditions["problem"] = problem or None
                c.setdefault("capture_conditions", {})[u["capture"]] = capture_conditions
                if problem and u["capture"] != "metrics":
                    c.setdefault("problems", []).append(f"{u['capture']} trace: {problem}")
                if b in slots:  # the traced wave ran fragmented: its trace is not the batch's (the gate's slots item)
                    c["slots_problem"] = c.get("slots_problem") or f"{u['capture']} trace: {slots[b]}"
                fields = trace_fields(u["capture"], window_summary(path, recs.get(b), anchor, later))
                for k, v in fields.pop("eager_kernels", {}).items():  # never over a node capture's, which overwrites
                    c.setdefault(k, v)                                 # these whichever is read first
                c.update(fields)
                if u["capture"] != "metrics":
                    c["trace_offset_steps"] = later
                # what marked the steps and where the window sat (steps counted back from the capture's last anchor):
                # the page's strips cut the same window with the same anchor (anatomy.strip), never a second rule
                if u["capture"] != "metrics":
                    c["anchor"] = anchor
                    c["window_from_end"] = (recs.get(b) or {}).get("window_from_end")
                if u["capture"] == "hostload":  # how much of each contended core this batch's spinning loops took
                    rec = contention_for((read_meta(d) or {}).get("contention"), b) or {}
                    shares = rec.get("cpu_share")
                    c["hostload_contention"] = sum(shares) / len(shares) if shares else None
                sources.setdefault((*key, b), {})[u["capture"]] = source(session, path)
    return cells, sources


def occupancy(c, step, overhead):
    """(occupied, its source). The trace's own fraction, one process's, except where it cannot be: tracing costs the
    host time (a few us per traced eager kernel), which stretches the profiled step where the host sets the pace but
    not the GPU's work, so in a host-bound cell the trace's fraction reads low by the overhead. There the GPU's busy
    time per step over the unprofiled step is the number: where the trace itself shows idle beyond noise (its fraction
    under 1 - BEYOND_K x STEP_NOISE), tracing stretched the step beyond the noise between two processes (overhead over
    sqrt 2 x STEP_NOISE), and the busy time fits the untraced step within that noise. Anything else is two processes
    differing, not tracing: a power-capped card ran its traced process faster (-2.4% once, a 3.4% stall that was not
    there) or its busy time longer (SGLang at batch 64: busy 2.6% over the untraced step with a 99.6% trace), and busy
    over the untraced step would turn that into a host stall. Never clipped. Without whole steps (the activity
    fallback) the trace's own fraction is all there is. The noise the result carries is anatomy.report.derive's, per
    cell, from occupied_source."""
    busy, trace = c.get("busy_ms"), c.get("occupied_trace")
    stretched = overhead is not None and overhead > 2 ** 0.5 * STEP_NOISE
    idle = trace is not None and trace < 1 - BEYOND_K * STEP_NOISE
    fits = bool(busy and step) and busy <= step * (1 + BEYOND_K * 2 ** 0.5 * STEP_NOISE)
    if fits and (trace is None or (stretched and idle)):
        return busy / step, "trace over the untraced step"
    return trace, "trace"


def busy_invariance(c, overhead):
    """A slower host must leave the GPU's busy time per step where it was. Occupancy needs busy time unchanged over the
    whole stretch tracing itself adds (profiler_overhead), so the test counts only where the hostload trace (the same
    graph trace with the engine's cores contended) stretched the host at least that far again; a shorter one (a
    GPU-bound cell) tests nothing and says so (a -0.95% drift over +2.0% once stood against tracing's +24.5%), the
    hostload_stretch it reached recorded either way. busy_drift_untraced carries the
    drift linearly back to the untraced step, as a diagnostic only: two processes' busy times differ by their own noise,
    and the carry multiplies it."""
    stretch = c["profiled_step_ms_hostload"] / c["profiled_step_ms"] - 1
    drift = c["busy_ms_hostload"] / c["busy_ms"] - 1
    out = {"hostload_stretch": stretch}
    if stretch > 0:
        out["busy_drift_untraced"] = drift * max(overhead or 0, 0) / stretch
    out["busy_invariant"] = abs(drift) <= BUSY_INVARIANT if stretch >= max(HOSTLOAD_STRETCH, overhead or 0) else None
    return out


def cell_row(k, c, bw, cal, sources, metrics_cal=None):
    """One cell's row: its ids, its collected fields, the physics, the checks on its traces and the decomposition. An
    achievable MBU above 1 (the step moved more bytes than the card was measured to reach: the byte model or the
    bandwidth is wrong for this cell) leaves MBU and SOL unset with the reason in mbu_flag; the cell's step and
    occupancy do not depend on either, so it is not rejected for it."""
    ids = dict(zip((*KEY, "batch"), k))
    step, busy = c.get("step_ms"), c.get("busy_ms")
    peak = PEAKS.get(c.get("gpu"), H100_SXM)
    ph = physics(ids["model"], ids["batch"], ids["prompt_len"], ids["gen_len"], step, bw, peak=peak)
    ph["peak_gpu"] = peak.name
    ph["mbu_flag"] = None
    if (ph.get("mbu") or 0) > 1:
        ph |= {"mbu_flag": f"achievable MBU {ph['mbu']:.2f}: above 1, so the byte model or the bandwidth is wrong here",
               "mbu": None, "sol": None}
    overhead = (c["profiled_step_ms"] / step - 1) if step and c.get("profiled_step_ms") else None
    if "graph_window" in c:
        c["anchor_ok"] = anchor_check(c.get("profiled_step_ms"), c.get("graph_steps"), step,
                                      c.get("graph_row_ratio"))
    if "node_steps" in c:
        c["node_anchor_ok"] = anchor_check(c.get("node_step_ms"), c.get("node_steps"), step,
                                           c.get("node_row_ratio"))
    if "hostload_steps" in c and not whole_steps(c.get("hostload_steps"), c.get("hostload_row_ratio")):
        c.setdefault("problems", []).append("hostload trace: too few or missed step anchors")
    c["occupied"], c["occupied_source"] = occupancy(c, step, overhead)
    c["mixed_sessions"] = mixed_sessions(sources.get(k, {}))
    consistent = (overhead is None or overhead >= -OVERHEAD_OK) and \
        (not busy or not step or busy <= step * (1 + OVERHEAD_OK))
    if "ruler_clock_mhz" in cal and "sm_clock_lock_mhz" in c and cal["ruler_clock_mhz"] != c["sm_clock_lock_mhz"]:
        # the launch term prices each kernel at a node cost measured at the ruler's clock: at another, it is not this
        # cell's (an unlocked ruler beside locked cells, or the reverse)
        lock = lambda x: f"{x} MHz" if x else "an unlocked clock"  # noqa: E731
        c["clock_problem"] = "; ".join(filter(None, [c.get("clock_problem"),  # beside the clock log's, never over it
                                                     f"calibrated at {lock(cal['ruler_clock_mhz'])}, measured at "
                                                     f"{lock(c['sm_clock_lock_mhz'])}"]))
    overlaps = c.pop("overlaps_ns", None) or ()  # read here, never a column: one value per overlapped kernel
    # node tracing lengthens a graph's nodes (node_trace_inflation_ns, the ruler's); kernels launched one by one are
    # traced alike in either mode, so none comes off theirs: taken off, it made a gap of kernels x inflation that was
    # never there (0.4-0.5% of a graphs-off step, moved out of kernel excess). Their launch term still prices each
    # kernel at the empty graph node's cost, the only per-launch cost the ruler measures (eager_launch says which)
    inflation = 0.0 if c.get("eager_launch") else cal.get("node_trace_inflation_ns")
    release = cal.get("pdl_release_ns")  # what a PDL launch leaves exposed however early it started (ruler.pdl)
    wide = cal.get("pdl_wide_exposed_ns_per_node")  # the same for the widest node the chain ran, a real kernel's blocks
    dec = decompose(step, c.get("occupied"), c.get("kernel_ms"), ph["floor_ms"] or ph["floor_ms_datasheet"],
                    kernels_per_step=c.get("kernels_per_step"), node_floor_us=cal.get("graph_node_floor_us"),
                    inflation_ns=inflation, overlaps_ns=overlaps, steps=c.get("node_steps") or c.get("graph_steps"),
                    copy_ms=c.get("copy_ms"), overlap_model=TOLERANCES["overlap_model"], release_ns=release)
    if dec is not None:  # whose ruler measured the release the overlapped kernels are charged at least
        dec["launch_release_from"] = (cal.get("pdl_release_from") or c.get("session")) if release is not None else None
        dec["launch_release_wide_ns"] = wide
    span = cal.get("graph_node_floor_range_us")
    if dec and span and c.get("kernels_per_step") and step:
        # the node-cost part of the launch term at either end of the per-node cost the ruler bounds, with and without
        # node tracing's inflation taken off each overlap, and with no PDL release, the one-block chain's and the
        # widest node's (the release grows with the blocks a node runs: 488 against 652 ns on the H100 SXM; ruler.pdl
        # sweeps the grids, and the cells are priced at the one-block plateau until the sweep says otherwise), the
        # gaps left out: a lower bound on launch cost, and how much of it the launch model decides
        ends = [exposed_launch_ms(c["kernels_per_step"], x, overlaps, c.get("node_steps") or c.get("graph_steps") or 1,
                                  cut, rel) / step
                for x in span for cut in {0.0, inflation or 0.0}
                for rel in {0.0, release or 0.0, wide or release or 0.0}]
        dec["launch_floor_share_range"] = [min(ends), max(ends)]
    metrics_cal = cal if metrics_cal is None else metrics_cal
    metrics_conditions = (c.get("capture_conditions") or {}).get("metrics")
    if metrics_conditions:
        why = metrics_conditions.get("problem")
        if not whole_steps(c.get("metrics_steps"), c.get("metrics_row_ratio")):
            why = why or "metrics trace has too few or missed step anchors"
        if not metrics_cal.get("dram_counter_scale"):
            why = why or "no calibration for the metrics capture's card"
        if c.get("dram_pct") is None or not c.get("dram_step_ms"):
            why = why or "no DRAM counters over a measured step"
        for field, calibrated in (("gpu", "ruler_gpu"), ("sm_clock_lock_mhz", "ruler_clock_mhz")):
            if calibrated in metrics_cal and field in metrics_conditions \
                    and metrics_cal[calibrated] != metrics_conditions[field]:
                why = why or f"metrics {field} differs from its ruler"
        c["metrics_problem"] = why
        c["metrics_valid"] = not why
    if c.get("dram_pct") is not None and c.get("dram_step_ms") and metrics_cal.get("dram_counter_scale") \
            and c.get("metrics_valid") is not False:
        # the counters read against what the ruler measured them to read on this card; without that scale the ratio
        # would look calibrated and is not, so it stays empty
        metrics_peak = PEAKS.get((metrics_conditions or {}).get("gpu"), peak)
        moved = (c["dram_pct"] / 100 * metrics_peak.bandwidth * metrics_cal["dram_counter_scale"]
                 * c["dram_step_ms"] / 1e3)
        ph |= {"dram_bytes_per_step": moved, "dram_over_model": moved / ph["bytes_per_step"]}
    if c.get("busy_ms_hostload") and busy and c.get("profiled_step_ms"):
        c |= busy_invariance(c, overhead)
    row = {**ids, **c, **ph, "profiler_overhead": overhead,
           "overhead_ok": None if overhead is None else abs(overhead) <= OVERHEAD_OK,
           "trace_consistent": None if overhead is None and not busy else consistent,
           **(dec or {}), "sources": sources.get(k, {})}
    row["rejection"] = rejection(row)  # the cell's own reason: no reader rebuilds it from seven flags
    row["accepted"] = not row["rejection"]
    row["dataset_status"] = row_status(row)  # in every row, so a downloaded cells.csv says what it is
    return row


def mixed_sessions(sources):
    """Whether a cell's timing and its traces come from different sessions (another rented host): its occupancy and
    profiler overhead would set one host's busy time against another's step. A rerun of the timing alone on a new host
    does this; the cell is rejected for it (anatomy.metrics.rejection), never read across hosts. Metrics captures
    are independent byte diagnostics, calibrated by their own session, so they do not participate in this join."""
    timed = {name.split("/", 1)[0] for cap, name in sources.items() if cap in TIMED}
    traced = {name.split("/", 1)[0] for cap, name in sources.items() if cap in ("graph", "node", "hostload")}
    return bool(timed and traced and timed != traced)


def build(roots, bw=None, calibration=None, calibrate=()):
    """One row per cell. For each (cell, capture, batch) one unit is read (chosen_units): the latest that finished ok,
    else the latest that finished at all (its per-batch problems then reject what they must); a unit with no
    status.json never finished and is skipped. The others are listed under duplicate_sources, never mixed in: a rerun
    lands over the cell it reran, in whatever order the sessions are given. Sentinels and noise units go to
    sentinel_drift and noise_table, never to cells. Each cell is read at its own session's bandwidth and ruler
    calibration, else the calibrating sessions' (S1, which runs the ruler, without its bring-up cells displacing the
    main grid's: they are read for calibration only); bw and calibration, when given, override both for every cell."""
    own, (bw_from, bw_any) = calibrations(roots, calibrate)
    cells, sources = collect(roots)
    rows = []
    for k, c in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        s_bw, s_cal, s_from = own.get(c.get("session"), (None, {}, None))
        c["bandwidth_from"] = "given" if bw else (c.get("session") if s_bw else bw_from)
        c["ruler_from"] = "given" if calibration is not None else s_from
        c["calibration_lent"] = None if calibration is not None else s_cal.get("calibration_lent")
        metrics = (c.get("capture_conditions") or {}).get("metrics")
        metrics_cal = None
        if metrics:
            _, metrics_cal, metrics_from = own.get(metrics["session"], (None, {}, None))
            c["metrics_ruler_from"] = "given" if calibration is not None else metrics_from
            if calibration is not None:
                metrics_cal = calibration
        rows.append(cell_row(k, c, bw or s_bw or bw_any, calibration if calibration is not None else s_cal, sources,
                             metrics_cal=metrics_cal))
    return rows


JSON_COLUMNS = ("classes_ms", "sources", "slowdowns", "duplicate_sources", "problems", "stack",
                "top_kernels", "capture_conditions")  # stored as JSON


def write(rows, out):
    """results/cells.parquet and cells.csv: the cell table, read back only by load_cells."""
    import pandas as pd

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in JSON_COLUMNS:
        if col in df:
            df[col] = df[col].map(lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v)
    df.to_parquet(out / "cells.parquet", index=False)
    df.to_csv(out / "cells.csv", index=False)
    return df


def load_cells(path):
    """The cell table as rows of plain Python values: missing values None (never NaN: on a float column
    where(notna, None) puts NaN back, and NaN passes every `is not None` check downstream), nested columns decoded,
    list columns as lists, each model's parameter count."""
    import pandas as pd

    from servebench.physics import load_shape

    df = pd.read_parquet(path)
    rows = df.astype(object).where(df.notna(), None).to_dict("records")
    for r in rows:
        r["params"] = load_shape(r["model"]).params
        for col in JSON_COLUMNS:
            if isinstance(r.get(col), str):
                r[col] = json.loads(r[col])
        for col, v in r.items():  # a list column comes back from Parquet as an array: rows are plain Python, JSON-ready
            if hasattr(v, "tolist"):
                r[col] = v.tolist()
    return rows


def reruns(roots):
    """reruns.csv's rows: every chosen timing or served unit with a batch whose repeat range or per-pair differential
    range is above range_ok (metrics.timing_cells' `rerun`), with that batch and its ranges. Read from the timing
    alone, so the box can write it at the end of a grid without a bandwidth or calibration (--reruns-only)."""
    out = {}
    for session, d, u, _gpu, _others, batches in chosen_units(roots):
        if u["capture"] not in TIMING:
            continue
        for b, t in timing_cells(reestimated(d, read_jsonl(d / "driver.jsonl"))).items():
            if b in batches and t.get("rerun"):
                row = out.setdefault((session, unit_key(d, session)), {"session": session, "unit": unit_key(d, session),
                                                                      "batches": [], "ranges": []})
                row["batches"].append(b)
                row["ranges"].append(f"B={b}: repeats {t.get('step_range') or 0:.1%}, differential "
                                     f"{t.get('differential_range') or 0:.1%}")
    return [{**r, "batches": " ".join(map(str, r["batches"])), "ranges": "; ".join(r["ranges"])}
            for _, r in sorted(out.items(), key=str)]


def distinct(paths):
    """Paths with any given twice (as a session and to --calibrate, say) kept once, in order: a session read twice
    would count its noise processes twice."""
    seen, out = set(), []
    for p in map(Path, paths):
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def unit_key(unit_dir, session):
    """A unit's key (its directory under the session) from its directory, or None."""
    parts = Path(unit_dir).parts if unit_dir else ()
    return "/".join(parts[parts.index(session) + 1:]) if session in parts else None


DERIVED = ("cells.parquet", "cells.csv", "strips.json", "cell-strips.json", "reruns.csv", "sentinels.csv",
           "noise.csv", "agreement.csv", "tolerances-suggested.yaml", "patterns.json")


@contextmanager
def staged_output(out):
    """Build a complete generation before publishing its directory. A failed build leaves the previous one intact.

    Preserve unrelated files (for example results/README.md). During the two renames the output may briefly be
    absent, but a reader can never open a directory containing half old and half new tables.
    """
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-tidy-", dir=out.parent) as work:
        stage, previous = Path(work) / "new", Path(work) / "old"
        if out.exists():
            shutil.copytree(out, stage, copy_function=os.link)
        else:
            stage.mkdir()
        for name in DERIVED:
            (stage / name).unlink(missing_ok=True)
        yield stage
        if out.exists():
            out.rename(previous)
        try:
            stage.rename(out)
        except BaseException:
            if previous.exists():
                previous.rename(out)
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sessions", nargs="*", type=Path, help="the sessions whose cells make the table")
    p.add_argument("--calibrate", nargs="*", type=Path, default=[],
                   help="sessions read for the ruler, the noise block and the sentinels only, never cells (S1); alone, "
                        "it writes the noise, the sentinels and the suggested tolerances")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--bandwidth", type=float, help="achievable read bandwidth, B/s (default: the sessions' ruler)")
    p.add_argument("--reruns-only", action="store_true",
                   help="write reruns.csv for the sessions from their timing alone (no ruler needed): the box's own "
                        "check at the end of a grid")
    a = p.parse_args(argv)
    import pandas as pd

    sessions, calibrate = distinct(a.sessions), distinct(a.calibrate)
    if not sessions and not calibrate:
        p.error("give the sessions whose cells make the table, or --calibrate <S1> alone")
    if any(s.resolve().is_relative_to(a.out.resolve()) for s in [*sessions, *calibrate]):
        p.error("--out must not contain a raw session directory")
    if a.reruns_only:
        a.out.mkdir(parents=True, exist_ok=True)
        table = reruns(sessions)
        pd.DataFrame(table, columns=["session", "unit", "batches", "ranges"]).to_csv(a.out / "reruns.csv", index=False)
        print(f"{len(table)} units to rerun -> {a.out}/reruns.csv (anatomy.grid --redo)")
        return
    every = distinct([*sessions, *calibrate])
    if sessions and a.bandwidth is None and bandwidth(every) is None:
        raise SystemExit("no ruler/bandwidth.json in these sessions: pass the session that ran the ruler with "
                         "--calibrate (S1), or --bandwidth; without it every floor, MBU and launch term is missing")
    with staged_output(a.out) as out:
        written_out = []
        if sessions:
            rows = build(sessions, a.bandwidth, calibrate=[c for c in calibrate if c.resolve() not in
                                                           {s.resolve() for s in sessions}])
            df = write(rows, out)
            from anatomy.strip import cell_strips, strips

            cut = strips(rows, every)
            (out / "strips.json").write_text(json.dumps(cut) + "\n")  # the page's opening picture, from the node traces
            per_cell = cell_strips(rows, every)  # the cell explorer's, one per cell with a node trace
            (out / "cell-strips.json").write_text(json.dumps(per_cell, separators=(",", ":")) + "\n")
            written_out += [f"{len(df)} cells -> cells.parquet, cells.csv",
                            f"trace strips for {sorted(cut)} -> strips.json",
                            f"{len(per_cell['cells'])} cells' strips -> cell-strips.json"]
            table = reruns(sessions)
            pd.DataFrame(table, columns=["session", "unit", "batches", "ranges"]).to_csv(out / "reruns.csv",
                                                                                      index=False)
            written_out.append(f"{len(table)} units to rerun -> reruns.csv (anatomy.grid --redo)")
        from anatomy.agreement import agreement
        from anatomy.noise import BANDS, noise_table, sentinel_drift, suggested_tolerances

        drift, noise, agree = sentinel_drift(every), noise_table(every), agreement(sessions)
        for name, table in (("sentinels.csv", drift), ("noise.csv", noise), ("agreement.csv", agree)):
            if table:  # an empty table is no file: every reader takes a missing one as empty, and staged_output has
                pd.DataFrame(table).to_csv(out / name, index=False)  # already removed the previous generation's
        written_out += [f"{len(drift)} sentinel rows -> sentinels.csv", f"{len(noise)} noise rows -> noise.csv",
                        f"{len(agree)} greedy-token comparisons -> agreement.csv"]
        if calibrate:
            import yaml

            suggested = suggested_tolerances(calibrate, set_from=", ".join(c.name for c in calibrate))
            if suggested:
                (out / "tolerances-suggested.yaml").write_text(
                    f"# suggested by anatomy.tidy from {suggested['set_from']}'s noise block, by the rule in "
                    "configs/tolerances.yaml's header: copy set_from, overlap_model and the four tolerance values "
                    "above `bands` into it\n"
                    + yaml.safe_dump(suggested, sort_keys=False))
                print("suggested tolerances:",
                      {k: suggested[k] for k in ("set_from", "overlap_model", *BANDS) if k in suggested},
                      f"-> {a.out}/tolerances-suggested.yaml")
                for f in suggested["g1_failures"]:
                    print(f"  G1 FAILURE: {f['engine']} {f['level']} {f['statistic']} {f['value']} above {f['most']}")
        print(f"{a.out}: " + "; ".join(written_out))


if __name__ == "__main__":
    main()
