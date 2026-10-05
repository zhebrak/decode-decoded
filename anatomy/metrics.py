"""From unit records to one row per cell: the numbers the page and the post show, every one traceable to a unit
directory. Pure functions over the records servebench writes, so they are tested without a GPU.

A cell is (engine, version, arm, level, model, prompt_len, gen_len, batch). Its row joins:
- timing (unprofiled: the L1 timing unit or the L2 served unit): step_ms = median over repeats of the cadence rate
  estimate (the mean step); the repeats' range; the differential estimate as the independent check, and whether it
  agrees within AGREE (configs/tolerances.yaml). The median token interval is not a check: it is a different
  statistic of the same arrivals, and where steps are irregular (llama-server at B8, validation on an A100: median
  5.78 ms, mean 6.09, p90 up to 7.3) the two differ although both are right. Their gap is reported as `irregularity`
  (mean over median - 1). An L2 cell has its differential from short AIPerf waves, and one more check where the
  server reports one: its own time per token (servebench.aiperf.server_step_ns) against the client's over the same
  whole requests (client_whole_ns), which confirms the client's timestamps by the server's clock.
  The differential spans tokens short_gen_len..gen_len, so it is judged against the cadence over those same tokens
  (cadence_tail), not the whole window: KV bytes per step grow with context, and a 50-token difference in mean context
  is several percent on a KV-heavy cell (validation, A100, B64: 2.9% against the whole window, under 0.1% against the
  tail).
  Every sequence must have produced exactly gen_len tokens, and no in-window stall (a preempted or queued sequence)
  may appear: either one invalidates the cell (`valid`), with the reason kept;
- graph capture: the GPU's busy time per step, so occupied = busy / step_ms (anatomy.tidy), and the profiled step
  time, whose ratio to step_ms is the profiler overhead (reported; it does not bias occupied);
- node capture: kernel-active ms per step and per kernel class;
- physics: bytes and FLOPs per step at the window's mean context, the bandwidth floor at datasheet and at the
  achievable bandwidth measured on the box (ruler/bandwidth.py), MBU at both, MFU, SOL, arithmetic intensity, and
  whether the cell leans compute-bound (intensity above half the ridge: MBU is shown faded there);
- the decomposition (anatomy.decompose).

Window context: the measured steps are the all-active span less `margin` steps each end, so a sequence's mean context
over them is about prompt_len + gen_len / 2. Sequences start a few steps apart where prefill mixes into decode (vLLM,
TensorRT-LLM: the last tokens spread over ~5 steps at batch 64); the timing window's middle sits at that spread's
middle, and anatomy.tidy moves each trace window to the same place (trace_offset).
"""
import functools
import math
from statistics import median

import yaml

from servebench import ROOT
from servebench.cgroup import throttled_share
from servebench.engines.base import version_sets
from servebench.physics import H100_SXM, floors, load_shape

CONFIGS = ROOT / "configs"
# arms that change only the host's schedule: the host runs before the GPU every step, so its time shows in every step
# (the fit on them cross-checks the calculator); and every arm that changes no arithmetic, whose greedy tokens must
# be its defaults' very tokens (the agreement check)
SERIAL_ARMS = ("async-off", "overlap-off")
IDENTICAL_ARMS = (*SERIAL_ARMS, "mrv2-off", "gpu-sampling")
TOLERANCES = yaml.safe_load((CONFIGS / "tolerances.yaml").read_text())  # when a cell counts: quality control
AGREE, OVERHEAD_OK, RANGE_OK, CPU_THROTTLED = (TOLERANCES[k] for k in ("agree", "overhead_ok", "range_ok",
                                                                      "cpu_throttled"))
BEYOND_K = TOLERANCES["beyond_k"]  # a difference counts beyond this many times its noise: the patterns', the stall's
NOISE_FLOOR = TOLERANCES["noise_floor"]  # the least relative noise any comparison assumes
STUDY_MODELS = tuple(yaml.safe_load((CONFIGS / "grid" / "v1.yaml").read_text()).get("qwen3") or ())  # one family


HEADLINE = {"vllm": "L1", "sglang": "L1", "trtllm": "L1", "llamacpp": "L2"}  # the engine level, else its server's
STUDY_CARD = H100_SXM.name  # the study's card: cells from any other are validation, never the study


# why a row is not the study's, in the words the page's banner uses
NOT_STUDY = {"validation": "measured by a validation grid", "card": "measured on a card other than the study's",
             "unchecked": "read with another session's calibration, which this card could not check"}


def not_study(r):
    """The NOT_STUDY keys that make a measured row validation data, [] for a study row. A row with no card recorded (a
    unit that failed before its metadata was written: SGLang's runner out of memory at 14B in S2-20261003) is missing
    evidence, not evidence of another card, and names nothing here (row_status: unknown)."""
    return [k for k, holds in (("validation", r.get("validation")),
                               ("card", bool(r.get("gpu")) and r.get("gpu") != STUDY_CARD),
                               # a session that measured nothing of its own card was priced with another rental's
                               ("unchecked", "(unchecked)" in str(r.get("ruler_from") or ""))) if holds]


def row_status(r):
    """One row's dataset status: "preview" (the synthetic layout preview), "validation" (a validation grid's cell, a
    card other than the study's, or a session read with another session's calibration it could not check: not_study),
    else "study"."""
    if r.get("synthetic"):
        return "preview"
    if not r.get("gpu") and not accepted(r):
        return None  # a failed unit without its metadata: unknown, and it decides nothing about the table
    return "validation" if not_study(r) else "study"


def dataset_status(rows):
    """A table's status: the least of its rows' (any preview row makes it a preview, any validation row validation; a
    row of unknown status, a failed unit without metadata, counts for nothing; an empty table, or one of unknown rows
    alone, is no study)."""
    found = {s for r in rows if (s := row_status(r))}
    return next((s for s in ("preview", "validation") if s in found), "study" if found else "validation")
PROMPT = 896  # the headline context; the long-context block's cells are read on their own


@functools.cache
def release_ranks():
    """{(engine, version): the index of the newest version-set file listing it}."""
    return {(e, v): i for i, (_, vs) in enumerate(version_sets()) for e, v in vs.items()}


def current_versions(rows):
    """{engine: version} the views show: per engine, the newest release among the rows, ranked by the version-set files
    (configs/versions, named by date: a release ranks with the newest file listing it, one in no file below every
    listed one). A refresh's release replaces the older one in every view once its cells are in the table; the older
    cells stay in the table."""
    rank = release_ranks()
    out = {}
    for r in rows:
        e, v = r.get("engine"), r.get("version")
        if e is not None and v is not None and (e not in out or rank.get((e, v), -1) > rank.get((e, out[e]), -1)):
            out[e] = v
    return out


def defaults(rows, only_accepted=True, arms=("defaults",)):
    """The cells the views read: the study's model family (the grid's qwen3 list; the second family's cells are read
    on their own) at each engine's current release (current_versions), the headline prompt, the given arms (None:
    every arm) and any level; accepted ones unless only_accepted is False. The one place this filter is written (with
    headline_rows below): copies of it drifted, and two let rejected cells into the frontier and the mitigations."""
    newest = current_versions(rows)
    return [r for r in rows if (arms is None or r.get("arm") in arms) and r.get("prompt_len", PROMPT) == PROMPT
            and (not STUDY_MODELS or r.get("model") is None or r["model"] in STUDY_MODELS)
            and (r.get("version") is None or r["version"] == newest.get(r.get("engine"), r["version"]))
            and (accepted(r) or not only_accepted)]


def headline_rows(rows, only_accepted=True, arms=("defaults",)):
    """defaults at each engine's headline level: what the map, the anatomy, the frontier and the calculator read (and,
    with arms=None, the mitigations)."""
    return [r for r in defaults(rows, only_accepted, arms) if r.get("level") == HEADLINE.get(r.get("engine"))]


def accepted(row):
    """Whether a cell's numbers may be used: by the patterns, the fit, the map and the page. Its timing is valid (no
    stalls, every sequence ran its full length), its estimators do not disagree, its trace is consistent with its
    timing (trace_consistent: anatomy.tidy), its traces' step anchors mark the steps (anchor_ok, node_anchor_ok), and
    its busy time held under further host load where that was tested (busy_invariant), and its timing and traces come
    from one session (one host).
    Profiler overhead alone does not reject a cell: occupancy is the trace's GPU busy time per step over the unprofiled
    step, which host-side tracing cost does not bias (overhead_ok is reported, for the page's note). A flag that was
    never computed (no differential, no trace) does not reject a cell; a flag computed False does. The rules are
    rejection's: a cell is accepted exactly when it has no reason to be rejected."""
    return not rejection(row)


def conditions_problem(row):
    """A thermal or hardware slowdown while the cell was timed, another process on a GPU that was not declared shared,
    or a CPU quota throttling the host during the cell's measured waves (servebench.cgroup; a runner's whole run);
    power capping is the card at its configured limit, reported only."""
    from servebench.clocks import FAULTS
    faults = {k: v for k, v in (row.get("slowdowns") or {}).items() if k in FAULTS}
    if faults:
        return "GPU slowed while timed: " + ", ".join(f"{k} in {v:.0%} of samples" for k, v in faults.items())
    if row.get("gpu_shared") and not row.get("shared_box"):
        return "another process shared the GPU while it was timed"
    if (row.get("cpu_throttled") or 0) > CPU_THROTTLED:  # a CPU quota stretched host loops: not the engine's host cost
        return f"a CPU quota throttled {row['cpu_throttled']:.1%} of the measured waves' quota periods"
    return None


def rejection(row):
    """Why a cell is not accepted, in words ("" when it is): the cell table's rejection column, and accepted's rules."""
    reasons = list(row.get("problems") or [])
    if row.get("step_ms") is None:
        reasons.append(row.get("error") or "no step time")
    if row.get("estimators_agree") is False:
        reasons.append(f"an independent check disagrees by more than {AGREE:.0%} (the differential with the cadence, "
                       "or the server's time per token with the client's)")
    if row.get("trace_consistent") is False:
        busy, step = row.get("busy_ms"), row.get("step_ms")
        if busy and step and busy > step * (1 + OVERHEAD_OK):
            reasons.append(f"GPU busy {busy:.3f} ms per step in the trace, above the unprofiled step {step:.3f} ms")
        else:
            reasons.append(f"profiled step {100 * (row.get('profiler_overhead') or 0):+.1f}%: faster than the "
                           f"unprofiled one, so not the same conditions")
    if row.get("busy_invariant") is False:
        reasons.append("the GPU's busy time per step moved when the host was loaded further: occupancy's assumption "
                       "that tracing slows the host and not the GPU fails here")
    if row.get("anchor_ok") is False or row.get("node_anchor_ok") is False:
        reasons.append("the trace's step anchors do not mark the steps (period off the unprofiled step, too few, or "
                       "one missed)")
    if row.get("revision_problem"):  # the unit loaded another snapshot than the pin (configs/models/revisions.yaml)
        reasons.append(row["revision_problem"])
    if row.get("slots_problem"):  # llama-server's batch split across slots that are not one run (anatomy.tidy)
        reasons.append(row["slots_problem"])
    if row.get("clock_problem"):  # the lock did not hold where nothing slowed the clock, or the ruler that priced its
        # launches ran at another SM clock (anatomy.tidy unit_conditions, cell_row)
        reasons.append(row["clock_problem"])
    if row.get("mixed_sessions"):  # a rerun of the timing alone on another host: occupancy would divide two hosts
        reasons.append("timing and traces from different sessions")
    if conditions_problem(row):
        reasons.append(conditions_problem(row))
    return "; ".join(reasons)


def runner_cells(steps):
    """{batch: timing summary} in timing_cells' shape for an L0 runner, from {batch: [step ms per run]}: its median and
    range, and the same rerun rule; no estimators to compare (a runner reports its own step)."""
    out = {}
    for b, xs in sorted(steps.items()):
        out[b] = {"step_ms": median(xs), "step_range": spread(xs), "repeats": len(xs), "differential_ms": None,
                  "interval_ms": None, "estimators_agree": None, "error": None, "rerun": spread(xs) > RANGE_OK}
    return out


def spread(xs):
    """(max - min) / median: the repeat range a cell is rerun for above RANGE_OK."""
    xs = [x for x in xs if x]
    return (max(xs) - min(xs)) / median(xs) if len(xs) > 1 else 0.0


def _cad(r, key, field):
    c = r.get(key) or {}
    return c.get(field)


def usable(x):
    """A step or differential that can be read: finite and positive (a NaN, an infinity, a zero or a negative is a
    broken record, never a fast cell)."""
    return isinstance(x, (int, float)) and math.isfinite(x) and x > 0


def wave_problems(waves, kind, planned):
    """What is missing or wrong in one batch's long or short waves: fewer than planned, a repeat index twice, a wave
    whose recorded streams are not its batch (a client that dropped a request: 255 streams read as batch 256), or a
    sequence that did not produce its wave's gen_len tokens."""
    out = []
    if planned and len(waves) < planned:
        out.append(f"{len(waves)} of {planned} {kind} waves recorded")
    repeats = [w.get("repeat") for w in waves]
    if len(set(repeats)) < len(repeats):
        out.append(f"{kind} repeats {sorted(r for r in set(repeats) if repeats.count(r) > 1)} recorded twice")
    streams = [w["repeat"] for w in waves if w.get("tokens") is not None and len(w["tokens"]) != w["batch"]]
    if streams:
        out.append(f"{kind} repeats {streams} recorded a stream count other than the batch")
    short = [w["repeat"] for w in waves if not w.get("tokens") or any(t != w.get("gen_len") for t in w["tokens"])]
    if short:
        out.append(f"{kind} repeats {short} did not produce gen_len tokens in every sequence")
    # a server's own count of AIPerf's text prompts (every server read 896 of 896 on the pod): another length is
    # another cell, its prefill and its KV cache not the ones the row names
    prompted = [w["repeat"] for w in waves if w.get("prompt_len")
                and any(p is not None and p != w["prompt_len"] for p in w.get("prompt_tokens") or [])]
    if prompted:
        out.append(f"{kind} repeats {prompted} sent prompts of another length than prompt_len")
    return out


def timing_cells(records):
    """{batch: timing summary} from one timing or served unit's driver.jsonl records. A batch cut short is invalid
    with its reason: an error or skip recorded for it (a crash after some repeats kept the earlier ones), its long or
    short waves incomplete (wave_problems: fewer than the info record's repeats, a stream missing, a sequence short of
    its tokens), a long wave without a cadence, or fewer usable differentials than repeats (the independent check
    did not run, or ran on broken spans). A step that is not finite and positive is never read."""
    out = {}
    longs, shorts, diffs, errors = {}, {}, {}, {}
    planned = next((r.get("repeats") for r in records if r.get("kind") == "info"), None)
    for r in records:
        b = r.get("batch")
        if r["kind"] == "long":
            longs.setdefault(b, []).append(r)
        elif r["kind"] == "short":
            shorts.setdefault(b, []).append(r)
        elif r["kind"] == "differential":
            diffs[b] = list(r["step_ns"] or [])
        elif r["kind"] in ("error", "skipped"):
            errors[b] = r.get("error") or f"skipped: {r.get('reason')}"
    for b in sorted(set(longs) | set(errors)):
        rs = longs.get(b, [])
        steps = [_cad(r, "cadence", "step_ns") for r in rs]
        rate = [x for x in steps if usable(x)]
        interval = [x for r in rs if usable(x := _cad(r, "cadence", "interval_ns"))]
        tail = [x for r in rs if usable(x := _cad(r, "cadence_tail", "step_ns"))]
        good_diffs = [d for d in diffs.get(b) or [] if usable(d)]
        step = median(rate) / 1e6 if rate else None
        diff = median(good_diffs) / 1e6 if good_diffs else None
        inter = median(interval) / 1e6 if interval else None
        tail_ms = median(tail) / 1e6 if tail else None
        server = [x for r in rs if usable(x := r.get("server_step_ns"))]
        whole = [x for r in rs if usable(x := r.get("client_whole_ns"))]
        server_ms = median(server) / 1e6 if server else None
        whole_ms = median(whole) / 1e6 if whole else None
        checks = [(diff, tail_ms or step), (server_ms, whole_ms)]
        checks = [(x, ref) for x, ref in checks if x and ref]
        agree = None if step is None or not checks else all(abs(x / ref - 1) <= AGREE for x, ref in checks)
        stalls = sum((_cad(r, "cadence", "stalls") or 0) for r in rs)
        short = [r["repeat"] for r in rs if r.get("tokens") and any(t != r.get("gen_len") for t in r["tokens"])]
        # the short waves carry half the differential: a stall in one is a problem as in a long one, a pause is counted
        # (short_hiccups), and the per-pair differentials' own range joins the rerun rule (an 859 ms short wave
        # against 552 once passed unseen under the median of three)
        stalled = [r["repeat"] for r in shorts.get(b, []) if _cad(r, "cadence", "stalls")]
        broken = [x for x in steps if x is not None and not usable(x)]
        incomplete = b in errors and not rs
        checks = [  # (what is wrong, in words) where it is: the batch's reasons, in the order a reader looks
            (stalls, f"{stalls} sequences fell behind the batch (preemption or queueing)"),
            (not incomplete, wave_problems(rs, "long", planned)),
            (stalled, f"short waves {stalled}: sequences fell behind the batch"),
            (b in errors and rs, f"the batch was cut short: {errors.get(b)}"),
            (planned and rs, wave_problems(shorts.get(b, []), "short", planned) if planned else []),
            (rate and len(rate) < len(rs), f"{len(rate)} of {len(rs)} long waves gave a cadence"),
            (broken, f"long-wave steps {broken} are not finite and positive"),
            (planned and rs and len(good_diffs) < planned, f"{len(good_diffs)} of {planned} differentials usable"),
            (rs and not rate and not short and b not in errors,
             "the batch never had every sequence decoding at once (the last started after the first finished: an "
             "engine admitting fewer than B)"),
        ]
        problems = [x for found, reason in checks if found for x in (reason if isinstance(reason, list) else [reason])]
        diff_range = spread(good_diffs)
        throttled = throttled_share([*rs, *shorts.get(b, [])])  # a CPU quota's, over the measured waves
        out[b] = {"step_ms": step, "step_range": spread(rate), "repeats": len(rate), "differential_ms": diff,
                  "tail_ms": tail_ms, "interval_ms": inter, "estimators_agree": agree, "error": errors.get(b),
                  "irregularity": step / inter - 1 if step and inter else None, "server_step_ms": server_ms,
                  "stalls": stalls, "valid": step is not None and not problems, "problems": problems,
                  "differential_range": diff_range,
                  "short_hiccups": sum((_cad(r, "cadence", "long_gaps") or 0) for r in shorts.get(b, [])),
                  # the longest pause that held every sequence, in steps: a host hiccup is part of the engine's step,
                  # reported (vLLM and SGLang at 256 sequences on 4 cores paused up to 26 steps, unthrottled)
                  "max_pause_steps": max((_cad(r, "cadence", "max_gap_steps") or 0 for r in rs), default=0) or None,
                  "cpu_throttled": throttled,
                  "rerun": bool(rate) and max(spread(rate), diff_range) > RANGE_OK}
    return out


def mean_context(prompt_len, gen_len):
    return prompt_len + gen_len / 2


def physics(model, batch, prompt_len, gen_len, step_ms, bw_achievable=None, peak=H100_SXM):
    """Bytes, FLOPs, floors and utilisations of one cell; None fields where step_ms is missing."""
    shape = load_shape(model)
    ctx = [round(mean_context(prompt_len, gen_len))] * batch
    f = floors(shape, ctx, peak)
    row = {"bytes_per_step": f["bytes"], "flops_per_step": f["flops"], "intensity": f["intensity"],
           "floor_ms_datasheet": f["floor_s"] * 1e3,
           "floor_ms": f["bytes"] / bw_achievable * 1e3 if bw_achievable else None,
           "bw_achievable": bw_achievable, "ridge": peak.flops / (bw_achievable or peak.bandwidth)}
    row["compute_leaning"] = f["intensity"] > row["ridge"] / 2
    if step_ms:
        s = step_ms / 1e3
        row |= {"mbu_datasheet": f["bytes"] / s / peak.bandwidth,
                "mbu": f["bytes"] / s / bw_achievable if bw_achievable else None,
                "mfu": f["flops"] / s / peak.flops,
                "sol": row["floor_ms"] / step_ms if row["floor_ms"] else row["floor_ms_datasheet"] / step_ms}
    return row
