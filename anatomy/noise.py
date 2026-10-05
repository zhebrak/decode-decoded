"""The study's own uncertainty, read from the first session: the noise block's processes (noise_readings,
noise_table: the spread between separate processes per cell, what every comparison is judged against), the sentinels
(sentinel_drift: drift within a session, Gate G2), and the tolerances the noise block sets by one rule
(suggested_tolerances, configs/tolerances.yaml's header). Every reading is judged per batch by the cells' rules
(reading_problems) before it counts. anatomy.tidy writes their tables beside the cell table."""
import math
import statistics
from pathlib import Path

from anatomy.llama_log import llama_prefill, llama_slots
from anatomy.metrics import BEYOND_K, TOLERANCES, timing_cells
from anatomy.tidy import (
    OVERHEAD_OK,
    TIMED,
    anchor_check,
    anchor_for,
    clock_reference,
    finish_spread,
    occupancy,
    own_processes,
    reestimated,
    runner_timing,
    trace_fields,
    trace_offset,
    trace_problem,
    unit_conditions,
    units,
    window_summary,
    windows,
    written,
)
from servebench.cgroup import throttled_share
from servebench.results import is_noise, is_sentinel, read_jsonl


def reading_problems(d, u, timing, recs, reference=None):
    """{batch: why this unit's reading there is no measure of spread or drift}, by the rules a cell is judged by
    (anatomy.metrics.rejection's, on what tidy knows before the traces): a stalled or cut-short wave, llama-server's
    batch in slots that are not one run or its prefill stalled (llama_prefill: a pause the span differential spans and
    the median cadence does not), the GPU slowed or shared while it was timed, its clock off the lock where
    nothing slowed it, a CPU quota throttling the host, the wrong model snapshot. Not the estimators' agreement nor the
    repeats' range: the noise block is what sets those
    tolerances (suggested_tolerances), so a reading cannot be excluded by them. A noise or sentinel reading taken
    under any of these faults would pass as noise or drift without the check. Per batch where the record is per batch
    (the timing, a CPU quota's throttling, llama-server's slots): one batch's fault leaves the unit's other batches
    standing, since noise units are never rerun. The GPU's conditions (its clock log, the processes on it) are
    recorded per unit, so a slowdown or a foreign process anywhere in the unit excludes every batch of it. reference:
    the clock the session's ruler held under the lock (anatomy.tidy.clock_reference), what the unit's is read
    against."""
    conditions = unit_conditions(d, own_processes(u), reference)
    server = (u["engine"], u["level"]) == ("llamacpp", "L2")
    slots, stalls = (llama_slots(d, recs), llama_prefill(d, recs)) if server else ({}, {})
    out = {}
    for b, t in timing.items():
        reasons = list(t.get("problems") or [])
        if t.get("step_ms") is None:
            reasons.append(t.get("error") or "no step time")
        reasons += [x for x in (trace_problem({**conditions, "cpu_throttled": t.get("cpu_throttled")}, None,
                                              slots.get(b)), stalls.get(b)) if x]
        if reasons:
            out[b] = "; ".join(reasons)
    return out


def noise_readings(roots):
    """The noise block's units read like cells (one estimator for both): per unit (session, unit, position, timing,
    occupancy). timing is {batch: timing summary} for a timing or served unit (through reestimated) or a runner
    (runner_timing); occupancy is {batch: occupied} for a graph unit, its window placed and its occupancy chosen as a
    cell's are (window_summary, trace_offset, occupancy), against the timing unit of its own block (the same process
    position), and only where its anchors mark the steps and its busy time fits the step."""
    found, timing = [], {}
    held = clock_reference(roots)  # each session's clock under the lock, from its ruler
    for session, d, u, status, _gpu in units(roots):
        if is_noise(u.get("block")) and status is not None:  # a unit that never finished at all measured nothing
            found.append((session, d, u))
    for session, d, u in found:  # timing first: a graph unit's window is placed by its block's timing
        if u["capture"] in TIMED:
            recs = [] if u["capture"] == "runner" else reestimated(d, read_jsonl(d / "driver.jsonl"))
            t = runner_timing(d, u["engine"]) if u["capture"] == "runner" else timing_cells(recs)
            for b, problem in reading_problems(d, u, t, recs, held.get(session)).items():  # per batch, as a cell
                t[b]["noise_problem"] = problem
            timing[(session, u["block"], *(u[k] for k in ("engine", "arm", "level", "model")))] = \
                (t, finish_spread(d, recs) if recs else {}, unit_conditions(d, own_processes(u), held.get(session)))
    out = []
    for session, d, u in found:
        position = u["block"].split("-")[1]
        mine = timing.get((session, u["block"], *(u[k] for k in ("engine", "arm", "level", "model"))), ({}, {}, {}))
        if u["capture"] in TIMED:
            out.append((session, u, position, mine[0], {}, {}))
        elif u["capture"] == "graph":  # an occupancy reading per batch, or why there is none (occupied_reasons)
            anchor, occ, why = anchor_for(u), {}, {}
            conditions = unit_conditions(d, own_processes(u), held.get(session))
            records = read_jsonl(d / "driver.jsonl")
            recs = {r["batch"]: r for r in records if r["kind"] == "profile"}
            slots = llama_slots(d, records) if (u["engine"], u["level"]) == ("llamacpp", "L2") else {}
            found = windows(d, "graph", u["level"], u["batches"])
            for b in u["batches"]:
                if b not in found:
                    why[b] = "no trace export for the batch"
            for b, path in found.items():
                rec = recs.get(b) or {}
                fault = trace_problem({**conditions, "cpu_throttled": throttled_share([rec])}, rec, slots.get(b),
                                      mine[2]) or (mine[0].get(b) or {}).get("noise_problem")
                if fault:
                    why[b] = fault
                    continue
                step = (mine[0].get(b) or {}).get("step_ms")
                c = trace_fields("graph", window_summary(path, recs.get(b), anchor,
                                                         trace_offset(anchor, mine[1].get(b))))
                if not step:
                    why[b] = "no step time in its block's timing unit to read the trace against"
                    continue
                if not anchor_check(c.get("profiled_step_ms"), c.get("graph_steps"), step, c.get("graph_row_ratio")):
                    why[b] = "the trace's step anchors do not mark the steps"
                    continue
                overhead = c["profiled_step_ms"] / step - 1
                if overhead < -OVERHEAD_OK or (c.get("busy_ms") and c["busy_ms"] > step * (1 + OVERHEAD_OK)):
                    # not the same conditions as its timing: no occupancy to read (a cell is rejected here)
                    why[b] = f"not the timing's conditions (profiled step {overhead:+.1%}, or busy time over the step)"
                    continue
                occ[b] = occupancy(c, step, overhead)[0]
            out.append((session, u, position, {}, occ, why))
    return out


def noise_table(roots):
    """Run-to-run spread from the noise block (anatomy.grid.noise), one row per session and cell: its processes at
    both ends of the session pooled (step_ms their mean, sd their standard deviation relative to it, so drift over the
    session counts), the shift from the start's mean to the end's, the largest repeat range within one process, and,
    where the block traced the cell, occupied_sd: the SD of its occupancy between processes (absolute, in occupancy's
    units). Valid readings only (a stalled or cut-short wave is no measure of noise), the invalid ones counted. What
    configs/tolerances.yaml's values are set from (suggested_tolerances), and the noise anatomy.report.derive.Noise
    takes per engine, level, arm, model and batch."""

    runs, invalid, reasons, occs, occ_invalid, occ_reasons = {}, {}, {}, {}, {}, {}
    for session, u, position, timing, occ, why in noise_readings(roots):
        cell = (session, u["engine"], u["arm"], u["level"], u["model"])
        for b, t in timing.items():
            if not t.get("step_ms") or t.get("valid") is False or t.get("noise_problem"):
                invalid[(*cell, b)] = invalid.get((*cell, b), 0) + 1
                reason = (t.get("noise_problem") or "; ".join(t.get("problems") or []) or t.get("error")
                          or "no step time")
                reasons.setdefault((*cell, b), []).append(f"{u['block']}: {reason}")
                continue
            runs.setdefault((*cell, b), []).append((position, t["step_ms"], t.get("step_range") or 0.0))
        for b, o in occ.items():
            if o is not None:
                occs.setdefault((*cell, b), []).append(o)
        for b, w in why.items():
            occ_invalid[(*cell, b)] = occ_invalid.get((*cell, b), 0) + 1
            occ_reasons.setdefault((*cell, b), []).append(f"{u['block']}: {w}")
    out = []
    for key in sorted(set(runs) | set(invalid) | set(occs) | set(occ_invalid), key=str):
        session, engine, arm, level, model, b = key
        xs, os_ = runs.get(key, []), occs.get(key, [])
        steps = [x for _, x, _ in xs]
        mean = statistics.mean(steps) if steps else None
        at = {pos: statistics.mean([x for p, x, _ in xs if p == pos]) for pos in ("start", "end")
              if any(p == pos for p, _, _ in xs)}
        out.append({"session": session, "engine": engine, "arm": arm, "level": level, "model": model, "batch": b,
                    "processes": len(steps), "invalid": invalid.get(key, 0),
                    "reasons": " | ".join(reasons.get(key, [])), "step_ms": mean,
                    "sd": statistics.stdev(steps) / mean if len(steps) > 1 else None,
                    "range": (max(steps) - min(steps)) / mean if steps else None,
                    "start_to_end": at["end"] / at["start"] - 1 if len(at) == 2 else None,
                    "within_range_max": max((r for _, _, r in xs), default=None),
                    "occupied": statistics.mean(os_) if os_ else None, "occupied_processes": len(os_),
                    "occupied_sd": statistics.stdev(os_) if len(os_) > 1 else None,
                    "occupied_invalid": occ_invalid.get(key, 0),
                    "occupied_reasons": " | ".join(occ_reasons.get(key, [])),
                    # each process's step, for the A/A check: processes of one cell differ by nothing, so every
                    # pair the comparisons' rule would call different is a false positive (derive.Noise.aa)
                    "process_steps": " ".join(f"{x:.6g}" for x in steps)})
    return out


def sentinel_drift(roots):
    """Sentinel step times per session and position (start, end, unlocked; a redo's sentinels, block
    sentinel-redo<n>-<position>, carry their round). Gate G2 is drift within a session: each end and redo reading
    against the same sentinel's opening reading in its own session (drift; drift_ok within the tolerances'
    sentinel_drift), which is what the same-session pairs depend on. Across sessions every session is another rented
    host, so a session's opening reading against the earliest session's (host_drift) is a host difference, reported and
    fed to the comparisons' noise (anatomy.report.derive.Noise), never a gate. The unlocked sentinel is its own series
    (what locking the clock changes): its drift is against the session's locked opening reading, never gated. Valid
    readings only make a reference."""
    rows, start, earliest = [], {}, {}
    held = clock_reference(roots)  # each session's clock under the lock, from its ruler
    found = [(written(d), session, d, u) for session, d, u, status, _gpu in units(roots)
             if is_sentinel(u.get("block")) and status is not None]
    for _, session, d, u in sorted(found, key=lambda x: x[0]):
        parts = u["block"].split("-")
        position, redo = parts[-1], next((x for x in parts if x.startswith("redo")), None)
        recs = reestimated(d, read_jsonl(d / "driver.jsonl"))
        timing = timing_cells(recs)
        problems = reading_problems(d, u, timing, recs, held.get(session))
        for b, t in timing.items():
            key = (u["engine"], u["arm"], u["level"], u["model"], b)
            valid = bool(t["step_ms"]) and bool(t["valid"]) and not problems.get(b)
            if position == "start" and not redo and valid:
                start.setdefault((session, *key), t["step_ms"])
                earliest.setdefault(key, t["step_ms"])
            rows.append({"session": session, "position": position, "redo": redo, "engine": u["engine"],
                         "arm": u["arm"], "level": u["level"], "model": u["model"], "batch": b,
                         "step_ms": t["step_ms"], "valid": valid,
                         "problem": problems.get(b), "key": key})
    out = []
    for r in rows:
        key = r.pop("key")
        ref, first = start.get((r["session"], *key)), earliest.get(key)
        opening = r["position"] == "start" and not r["redo"]
        drift = r["step_ms"] / ref - 1 if ref and r["step_ms"] and not opening else None
        gated = not opening and r["position"] != "unlocked"
        out.append({**r, "drift": drift,
                    "drift_ok": bool(r["valid"] and drift is not None and abs(drift) <= TOLERANCES["sentinel_drift"])
                                if gated else None,
                    "host_drift": ref / first - 1 if ref and first else None})
    return out


BANDS = {k: tuple(v) for k, v in TOLERANCES["bands"].items()}  # what each set tolerance may take


def suggested_tolerances(roots, set_from=None):
    """configs/tolerances.yaml's header rule, applied to the noise block in roots (valid cells only, per engine and
    level): step_noise the largest between-process SD (to 0.1% up), agree the 99th percentile of |differential / tail
    - 1| and range_ok of the larger of the repeat and per-pair differential ranges (both to 0.5% up), sentinel_drift
    BEYOND_K x step_noise. Each global
    value is the largest over the engines and levels inside its band, clamped to the band (BANDS); an engine and level
    above the band is listed in g1_failures, and never loosens a tolerance for the others. {} without a noise
    block."""
    stats = {}  # {(engine, level): {"sd": [...], "agree": [...], "range": [...]}}
    for _session, u, _position, timing, _occ, _why in noise_readings(roots):
        s = stats.setdefault((u["engine"], u["level"]), {"sd": [], "agree": [], "range": []})
        for t in timing.values():
            if not t.get("step_ms") or t.get("valid") is False or t.get("noise_problem"):
                continue  # the readings noise_table counts, no other: an excluded reading sets no tolerance
            ref = t.get("tail_ms") or t["step_ms"]
            if t.get("differential_ms"):
                s["agree"].append(abs(t["differential_ms"] / ref - 1))
            s["range"].append(max(t.get("step_range") or 0, t.get("differential_range") or 0))
    for r in noise_table(roots):
        if r.get("sd") is not None:
            stats.setdefault((r["engine"], r["level"]), {"sd": [], "agree": [], "range": []})["sd"].append(r["sd"])
    if not any(s["sd"] for s in stats.values()):
        return {}
    up = lambda x, to: round(math.ceil(x / to - 1e-9) * to, 4)  # noqa: E731
    q99 = lambda xs: sorted(xs)[max(0, math.ceil(0.99 * len(xs)) - 1)] if xs else None  # noqa: E731  nearest rank
    per = [{"engine": e, "level": lvl, "noise_cells": len(s["sd"]), "readings": len(s["range"]),
            "step_noise": up(max(s["sd"]), 0.001) if s["sd"] else None,
            "agree": up(q99(s["agree"]), 0.005) if s["agree"] else None,
            "range_ok": up(q99(s["range"]), 0.005) if s["range"] else None}
           for (e, lvl), s in sorted(stats.items())]
    for p in per:
        p["sentinel_drift"] = round(BEYOND_K * p["step_noise"], 4) if p["step_noise"] is not None else None
    from anatomy.calibration import ruler_calibration
    from anatomy.decompose import suggested_overlap_model

    model, evidence = suggested_overlap_model(ruler_calibration(roots))
    out, failures = {"set_from": set_from, **({"overlap_model": model} if model else {})}, []
    for name, (lo, hi) in BANDS.items():
        values = [p[name] for p in per if p[name] is not None]
        inside = [v for v in values if v <= hi]
        out[name] = round(min(max(max(inside, default=lo), lo), hi), 4)
        failures += [{"engine": p["engine"], "level": p["level"], "statistic": name, "value": p[name], "most": hi}
                     for p in per if p[name] is not None and p[name] > hi]
    if out["sentinel_drift"] < round(BEYOND_K * out["step_noise"], 4):  # the drift line is never inside the noise
        out["sentinel_drift"] = min(round(BEYOND_K * out["step_noise"], 4), BANDS["sentinel_drift"][1])
    return {**out, "bands": {k: list(v) for k, v in BANDS.items()}, "overlap_model_evidence": evidence,
            "g1_failures": failures, "per_engine": per,
            "from": {"engine_levels": len(per), "noise_cells": sum(p["noise_cells"] for p in per),
                     "readings": sum(p["readings"] for p in per)}}


SET_KEYS = ("set_from", "overlap_model", *BANDS)  # what a suggestion sets in configs/tolerances.yaml, in its words


def apply_suggested(suggested, tolerances):
    """Write a suggestion's values (tolerances-suggested.yaml: set_from, overlap_model when the chain decided it, and
    the four set tolerances) into configs/tolerances.yaml in place, each on its own line, every comment kept: the
    step between S1's gate and S2's start, so it is one command and a commit. Returns {key: (before, after)} for the
    keys it changed; a suggestion listing g1_failures is applied all the same, since its values are inside the bands
    by the rule, and the failing engine's handling is the decision table's."""
    import re

    import yaml
    want = yaml.safe_load(Path(suggested).read_text()) or {}
    path = Path(tolerances)
    lines = path.read_text().splitlines(keepends=True)
    changed = {}
    for key in SET_KEYS:
        if key not in want or want[key] is None:
            continue
        value = want[key]
        for i, line in enumerate(lines):
            m = re.match(rf"^({key}):(\s+)(\S+)(.*)$", line.rstrip("\n"))
            if m:
                before = yaml.safe_load(m.group(3))
                if before != value:
                    changed[key] = (before, value)
                    pad = " " * max(0, len(m.group(3)) - len(str(value)))  # a shorter value: the comment stays put
                    lines[i] = f"{m.group(1)}:{m.group(2)}{value}{pad}{m.group(4)}\n"
                break
        else:
            raise KeyError(f"{tolerances} has no top-level {key}")
    path.write_text("".join(lines))
    return changed


if __name__ == "__main__":
    import argparse
    import sys

    import yaml

    p = argparse.ArgumentParser(description="apply <tolerances-suggested.yaml>: write its values into "
                                            "configs/tolerances.yaml")
    p.add_argument("action", choices=["apply"])
    p.add_argument("suggested", type=Path)
    p.add_argument("--tolerances", type=Path, default=Path(__file__).resolve().parents[1] / "configs/tolerances.yaml")
    a = p.parse_args()
    sug = yaml.safe_load(a.suggested.read_text()) or {}
    for f in sug.get("g1_failures") or []:  # applied all the same: the engine's handling is the decision table's
        print(f"g1 failure: {f['engine']} {f['level']} {f['statistic']} {f['value']} above {f['most']}",
              file=sys.stderr)
    done = apply_suggested(a.suggested, a.tolerances)
    print(f"{a.tolerances}: " + (", ".join(f"{k} {b} -> {v}" for k, (b, v) in done.items())
                                 or "already at the suggested values"))
