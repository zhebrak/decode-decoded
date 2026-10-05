"""Gate G1's ruler check, host side: the parser's occupancy from the synthetic workload's traces against the ground
truth the workload recorded with CUDA events (ruler/synthetic_busy.py), one capture window per host-wait setting.

    python -m ruler.validate ~/decode-anatomy-runs/<session>/ruler

Reads, per nsys mode and variant, the ground truth busy-<mode>-<variant>.json that run wrote and the exports
busy-<mode>-<variant>.<n>.sqlite (n is the window, in the order the workload opened them: nsys writes one export per
capture range under --capture-range-end=repeat, numbered from 1). Each trace is judged against its own run's truth.
Checks:
- graph-mode occupied vs ground-truth busy, every window, eager and graph variants: within TOLERANCE_PP points;
- node-mode kernel-active vs ground truth for the eager variant (no graph: the two must agree);
- for the graph variant, reports the in-graph per-node gap: (that run's device time per step - node-mode kernel time)
  / nodes, the device-side cost of a graph node on this card, beside launch_latency.py's graph_node_us.
A window during which a second process used the GPU (gpu-processes.log, written by ruler.run) is contaminated:
reported, excluded from the pass, and the report says to rerun it. A resident (gpu-residents.txt: a process already
on the GPU when the ruler started, another user's idle server) is tolerated while nvidia-smi pmon shows it idle around
the window (gpu-activity.log); active, or with no pmon sample near, it contaminates the window or leaves it unchecked.
Calibration for the decomposition (anatomy.decompose), from the graph variant's first clean window:
- node_trace_inflation_ns: how much node-mode tracing lengthens each kernel (node-mode kernel time per step minus the
  graph-mode span per step, over the nodes): kernel time from node-mode traces is corrected by it;
- graph_node_floor_us: what one node of a CUDA graph costs on this card at minimum (launch_latency.py), the per-kernel
  launch cost CUPTI folds into kernel durations rather than showing as a gap (64 ns between nodes in validation).
And the DRAM counters, where ruler.run could sample them (root): dram_counter_scale is the bandwidth a stream
of known bytes reached (bandwidth.py --profile; bytes over the reduction kernels' own durations in the trace) over
what the counters said of it (mean DRAM read throughput % in the samples inside those kernels x the datasheet peak).
anatomy.tidy scales every metrics capture's counters by it.
The report names the card (host.json and clock.json, written by ruler.run) and lists `problems`: evidence that is
missing or malformed (a run without its traces, window counts that differ, a non-finite number, a calibration
without its keys), any of which fails it as a failed window does.
Writes ruler_report.json and exits non-zero if any check fails, any window is contaminated or anything is missing.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

from ruler.pdl import pdl_calibration
from servebench import clocks
from servebench.nsys.parse import Trace, dram, numbered_exports, summarise
from servebench.physics import PEAKS
from servebench.results import write_json

TOLERANCE_PP = 1.0
ANCHOR = {"graph": "graph", "eager": "spin"}  # graph launches; the spin kernel (_spin, or torch.cuda._sleep's)


def traces(folder, mode, variant):
    return numbered_exports(folder, f"busy-{mode}-{variant}")


def gpu_processes(folder):
    """[(unix time, {pids})] from ruler.run's log; [] when there is none."""
    log = Path(folder) / "gpu-processes.log"
    out = []
    if log.exists():
        for line in log.read_text().splitlines():
            t, _, pids = line.partition(" ")
            try:
                out.append((float(t), {p.strip() for p in pids.split(",") if p.strip()}))
            except ValueError:
                continue
    return out


def residents(folder):
    f = Path(folder) / "gpu-residents.txt"
    return frozenset(x.strip() for x in f.read_text().split() if x.strip()) if f.exists() else frozenset()


def gpu_activity(folder):
    """[(unix time, {pid: SM %})] from the pmon log; a pid pmon shows as "-" (no sample) counts as 0."""
    log = Path(folder) / "gpu-activity.log"
    out = []
    if log.exists():
        for line in log.read_text().splitlines():
            t, _, rest = line.partition(" ")
            try:
                out.append((float(t), {p: (int(v) if v.isdigit() else 0) for p, _, v in
                                       (x.partition(":") for x in rest.split(",") if x.strip())}))
            except ValueError:
                continue
    return out


SLACK_S = 0.3  # a window is judged by the samples inside it and the nearest ones around it
PMON_S = 1.0   # one pmon sample covers about the second after its timestamp


def contention(samples, window, resident=frozenset(), activity=()):
    """"clean", "contaminated" (another process on the GPU at a sample in or bracketing the window) or "unchecked" (no
    sample near enough: a window shorter than the logger's period is not clean by default). The ruler runs one GPU
    process, so a sample with a second one that is not a resident is another user's; a resident present around the
    window must show no SM activity in the pmon samples covering it."""
    if not samples or not window:
        return "unchecked"
    lo, hi = window
    near = [pids for t, pids in samples if lo - SLACK_S <= t <= hi + SLACK_S]
    if not near:
        return "unchecked"
    if any(len(p - resident) > 1 for p in near):
        return "contaminated"
    present = set().union(*near) & resident
    if present:
        # only samples whose [t, t + PMON_S] overlaps the window, and together they must cover all of it: a window in
        # the gap between two pmon calls is unchecked, never clean
        cover = sorted((t, a) for t, a in activity if lo - PMON_S <= t <= hi)
        reach = lo
        for t, _ in cover:
            if t > reach:
                break
            reach = max(reach, t + PMON_S)
        if reach < hi:
            return "unchecked"
        if any(a.get(pid, 0) > 0 for _, a in cover for pid in present):
            return "contaminated"
    return "clean"


def calibration(folder, judge):
    out = {}
    lat = folder / "launch_latency.json"
    if lat.exists():
        # the per-kernel launch cost, bounded by two measurements of a graph node's own cost: an empty node
        # (launch_latency.py, the primary) and the overhead per node of the graph variant's untraced spin graph, its
        # back-to-back step less its known spin work. Not a split-GEMV slope: small GEMVs cannot fill the memory
        # system (7.1 us a node in validation, more than a whole step at 330 kernels), which is kernel inefficiency,
        # already in kernel excess
        est = [json.loads(lat.read_text())["graph_node_us"]]
        # the untraced run's: tracing adds ~30 ns a node (825 against 792 ns in validation)
        spin = next((f for f in (folder / f"busy-{m}-graph.json" for m in ("none", "graph")) if f.exists()), None)
        if spin:
            t = json.loads(spin.read_text())
            per_node = sorted((w["step_device_ns"] - t["spin_us"] * 1e3) / t["nodes"] / 1e3 for w in t["windows"]
                              if w.get("step_device_ns"))  # rulers before it was recorded: the empty node alone
            if per_node:
                est.append(per_node[len(per_node) // 2])
                out["graph_node_spin_us"] = est[1]  # the range's other end, apart: a session lent this calibration
                # sets it beside its own empty node (anatomy.calibration.calibrations)
        out["graph_node_floor_us"] = est[0]
        out["graph_node_floor_range_us"] = [min(est), max(est)]
    truth_g, truth_n = folder / "busy-graph-graph.json", folder / "busy-node-graph.json"
    if not (truth_g.exists() and truth_n.exists()):
        return out
    tg, tn = json.loads(truth_g.read_text()), json.loads(truth_n.read_text())
    for i, (wg, wn) in enumerate(zip(tg["windows"], tn["windows"])):
        gtr, ntr = traces(folder, "graph", "graph"), traces(folder, "node", "graph")
        if i >= min(len(gtr), len(ntr)) or judge(wg.get("unix")) != "clean" or judge(wn.get("unix")) != "clean":
            continue
        g, n = summarise(gtr[i], anchor="graph"), summarise(ntr[i], anchor="graph")
        if g.get("occupied_ns_per_step") and n.get("kernel_ns_per_step"):
            out["node_trace_inflation_ns"] = (n["kernel_ns_per_step"] - g["occupied_ns_per_step"]) / tg["nodes"]
            out["calibration_window"] = i + 1
            break
    return out


STREAM_KERNEL_NS = 500_000  # the 4 GiB reductions take milliseconds; anything shorter is not the stream


def empty_node_split(folder):
    """{empty_node_kernel_ns, empty_node_gap_ns, empty_node_period_ns}: an empty graph's replays traced node by node
    (launch_latency.py --profile), the median kernel duration, gap to the next kernel on the stream, and start-to-start
    period, raw (node tracing's own inflation included); {} without the trace. Whether a node's cost shows as duration
    or as gap says how to read the in-graph gaps of the engines' graphs (0.1-0.5 us a node on Hopper)."""
    paths = sorted(folder.glob("empty-node*.sqlite"))
    if not paths:
        return {}
    with Trace(paths[0]) as t:
        ks = sorted(t.kernels(), key=lambda k: k.start)
    pairs = [(a, b) for a, b in zip(ks, ks[1:]) if a.stream == b.stream and b.start - a.start < 20_000]  # in a replay
    if not pairs:
        return {}
    return {"empty_node_kernel_ns": statistics.median(k.ns for k in ks),
            "empty_node_gap_ns": statistics.median(b.start - a.end for a, b in pairs),
            "empty_node_period_ns": statistics.median(b.start - a.start for a, b in pairs)}


def dram_calibration(folder, judge=lambda window: "unchecked"):
    """{dram_counter_scale, ...} from the DRAM-counter run, {} without one, or {"dram_calibration": state} when another
    process shared the GPU during it."""
    meta, paths = folder / "bandwidth-profiled.json", sorted(folder.glob("dram-calibration*.sqlite"))
    if not meta.exists() or not paths:
        return {}
    info = json.loads(meta.read_text())
    peak = PEAKS.get(info.get("device"))
    state = judge(info.get("unix"))
    if peak is None or state != "clean":
        return {"dram_calibration": state if peak else f"no datasheet peak for {info.get('device')!r}"}
    with Trace(paths[0]) as t:
        reads = [k for k in t.kernels() if k.ns >= STREAM_KERNEL_NS]
        stamps = sorted({ts for series in t.gpu_metrics().values() for ts, _ in series})
        if not reads or len(stamps) < 2:
            return {"dram_calibration": "no stream kernels or no counter samples in the trace"}
        period = statistics.median(b - a for a, b in zip(stamps, stamps[1:]))
        d = dram(t, [(k.start + period, k.end) for k in reads])  # a sample averages the period before it
    if not d:
        return {"dram_calibration": "no counter samples inside the stream kernels (check the metrics' clock)"}
    # every byte of every call over all the stream kernels' time: x.sum() over 4 GiB runs as two 2 GiB kernels
    # (ATen splits a reduction whose offsets overflow 32 bits), so one kernel's duration would double the bandwidth
    measured = info["bytes"] * info.get("calls", len(reads)) / (sum(k.ns for k in reads) / 1e9)
    if measured > peak.bandwidth:
        return {"dram_calibration": f"stream measured at {measured / 1e12:.2f} TB/s, above the "
                                    f"{peak.bandwidth / 1e12:.2f} TB/s datasheet: kernels or byte count wrong"}
    return {"dram_counter_scale": measured / (d["dram_read_pct"] / 100 * peak.bandwidth),
            "dram_stream_bytes_per_s": measured, "dram_stream_read_pct": d["dram_read_pct"],
            "dram_stream_samples": d["dram_read_samples"]}


def cross_check(folder, variant, judge):
    """Rows of mode "cross": each graph-mode window's busy time per step over the untraced run's step for the same
    host wait, against the untraced run's truth. This is anatomy.tidy's occupancy (busy / unprofiled step), which the
    windows above, each judged within its own run, do not test."""
    untraced = folder / f"busy-none-{variant}.json"
    traced = folder / f"busy-graph-{variant}.json"
    if not untraced.exists() or not traced.exists():
        return []
    rows = []
    plain, windows = json.loads(untraced.read_text())["windows"], json.loads(traced.read_text())["windows"]
    for i, (w0, path) in enumerate(zip(plain, traces(folder, "graph", variant))):
        row = {"mode": "cross", "window": i + 1, "host_us": w0["host_us"], "truth_busy": w0["busy"]}
        state = judge(w0.get("unix"))
        if state != "clean" or judge(windows[i].get("unix")) != "clean":
            rows.append(row | {"contaminated": state if state != "clean" else "traced run"})
            continue
        s = summarise(path, anchor=ANCHOR[variant])
        if not s.get("occupied_ns_per_step"):
            rows.append(row | {"ok": False, "error": "no whole steps in the trace"})
            continue
        occ = s["occupied_ns_per_step"] / w0["step_ns"]
        rows.append(row | {"occupied": occ, "error_pp": 100 * (occ - w0["busy"]),
                           "ok": abs(100 * (occ - w0["busy"])) <= TOLERANCE_PP})
    return rows


REQUIRED = ("graph_node_floor_us", "graph_node_floor_range_us", "node_trace_inflation_ns")  # what tidy reads


def finite(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


def check(folder):
    """The ruler's report: every window of every run judged, the calibration, the card it ran on, and `problems`, the
    evidence that is missing or malformed. ok only when every window passed, none was contaminated, and nothing is
    missing: each mode and variant's truth with exactly one trace per window, both untraced runs, finite numbers, and
    the calibration the decomposition reads (an empty folder, or truth whose traces are gone, is not a pass)."""
    folder = Path(folder)
    report, ok, problems = {"tolerance_pp": TOLERANCE_PP, "variants": {}}, True, []
    samples, resident, activity = gpu_processes(folder), residents(folder), gpu_activity(folder)
    judge = lambda window: contention(samples, window, resident, activity)  # noqa: E731
    report["contention_logged"] = bool(samples)
    report["residents"] = sorted(resident)
    dirty = 0
    for variant in ("eager", "graph"):
        rows = []
        for mode in ("graph", "node"):
            truth_file = folder / f"busy-{mode}-{variant}.json"
            if not truth_file.exists():
                problems.append(f"no {truth_file.name}")
                continue
            truth, paths = json.loads(truth_file.read_text()), traces(folder, mode, variant)
            if len(truth["windows"]) != len(paths) or not paths:
                problems.append(f"{truth_file.name}: {len(truth['windows'])} windows against {len(paths)} traces")
            for i, (w, path) in enumerate(zip(truth["windows"], paths)):
                s = summarise(path, anchor=ANCHOR[variant])
                row = {"mode": mode, "window": i + 1, "host_us": w["host_us"], "truth_busy": w["busy"],
                       "steps": s["steps"]}
                state = judge(w.get("unix"))
                if state != "clean":
                    row["contaminated"] = state
                    dirty += 1
                    rows.append(row)
                    continue
                if mode == "graph":
                    row |= {"occupied": s["occupied"], "error_pp": 100 * (s["occupied"] - w["busy"])
                            if finite(s["occupied"], w["busy"]) else None}
                elif variant == "eager":
                    row |= {"kernel_active": s["kernel_active"], "error_pp": 100 * (s["kernel_active"] - w["busy"])
                            if finite(s["kernel_active"], w["busy"]) else None}
                elif s.get("kernel_ns_per_step"):
                    gpu_per_step = w["gpu_ns"] / w["steps"]
                    row["in_graph_gap_us_per_node"] = (gpu_per_step - s["kernel_ns_per_step"]) / truth["nodes"] / 1e3
                if "error_pp" in row:
                    row["ok"] = row["error_pp"] is not None and abs(row["error_pp"]) <= TOLERANCE_PP
                    if row["error_pp"] is None:
                        problems.append(f"{mode}/{variant} window {i + 1}: no finite measurement")
                    ok &= row["ok"]
                rows.append(row)
        if rows:
            report["variants"][variant] = rows
    for variant in ("eager", "graph"):  # the estimator the cells use: busy from a trace over an untraced run's step
        rows = cross_check(folder, variant, judge)
        if not rows:
            problems.append(f"no cross check for the {variant} variant (busy-none-{variant}.json and its graph run)")
        ok &= all(r.get("ok", True) for r in rows)
        dirty += sum(1 for r in rows if r.get("contaminated"))
        if rows:
            report["variants"].setdefault(variant, []).extend(rows)
    cal = calibration(folder, judge)
    pdl = pdl_calibration(folder, cal.get("graph_node_floor_us"), cal.get("node_trace_inflation_ns"))
    report["calibration"] = {**cal, **dram_calibration(folder, judge), **empty_node_split(folder), **pdl}
    missing = [k for k in REQUIRED if k not in report["calibration"]]
    if missing:
        problems.append(f"calibration without {', '.join(missing)}")
    report["card"] = card(folder)
    report["contaminated_windows"] = dirty
    report["problems"] = problems
    report["ok"] = ok and not problems and dirty == 0
    if dirty:
        report["advice"] = "another process used the GPU during some windows: rerun the ruler when the GPU is free"
    return report


def card(folder):
    """What the ruler ran on, from the files ruler.run writes beside its results: the host's id and GPU (host.json,
    servebench.env.host), the SM clock lock asked for (clock.json) and the clock the card held through the run
    (sm_mhz_held: clocks.csv's busy samples nothing slowed; a card may hold less than asked, and the cells are read
    against what it held: anatomy.tidy.unit_conditions); {} for a run that wrote none."""
    host = json.loads((folder / "host.json").read_text()) if (folder / "host.json").exists() else {}
    clock = json.loads((folder / "clock.json").read_text()) if (folder / "clock.json").exists() else {}
    log = folder / "clocks.csv"
    held = (clocks.summarise(log) if log.exists() else None) or {}
    known = {"host_id": host.get("id"), "gpu": (host.get("gpu") or {}).get("name"),
             "nsys": (host.get("tools") or {}).get("nsys"), "sm_mhz_held": held.get("sm_mhz_median_unthrottled")}
    return {k: v for k, v in known.items() if v is not None} | clock


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    report = check(a.folder)
    write_json(a.folder / "ruler_report.json", report)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
