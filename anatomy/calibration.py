"""What a session's cells are read against: the achievable bandwidth its ruler measured (the floor's denominator),
the ruler's calibration (the empty graph node's cost, node tracing's inflation, the DRAM counters' scale, the PDL
chain), and, where a session's own ruler did not pass, what it may borrow from the calibrating session (lend: the
same card within card_match). anatomy.tidy reads every session through calibrations()."""
import json
from pathlib import Path

from anatomy.metrics import TOLERANCES


def bandwidth(roots):
    """The achievable bandwidth measured in the sessions (ruler/bandwidth.py), or None: the fastest of its streaming
    read, copy and GEMV medians. A real kernel beating the denominator puts MBU above 1: on an H100 NVL a cuBLAS GEMV
    (3.44 TB/s) outran the x.sum() read (3.31), which the A100's order (read 1.43 > GEMV 1.32) had hidden."""
    for root in map(Path, roots):
        f = root / "ruler" / "bandwidth.json"
        if f.exists():
            b = json.loads(f.read_text())
            # a file without a measured figure (a fingerprint that failed partway) is no bandwidth: the session is
            # then read by the calibrating session's, as calibrations says
            found = [b[k] for k in ("read_bytes_per_s", "copy_bytes_per_s", "gemv_bytes_per_s") if b.get(k)]
            if found:
                return max(found)
    return None


def ruler_calibration(roots):
    """The ruler's calibration for the decomposition (ruler/validate.py: graph-node floor, node-trace inflation, the
    DRAM counter scale), from the first report that passed: a failed one's numbers (a parser off its truth, a
    contaminated window, evidence missing) never price a cell."""
    for root in map(Path, roots):
        f = root / "ruler" / "ruler_report.json"
        if f.exists():
            report = json.loads(f.read_text())
            if report.get("ok") and report.get("calibration"):
                card = report.get("card") or {}  # the clock its launch costs were measured at, where recorded
                clock = {"ruler_clock_mhz": card["sm_clock_lock_mhz"]} if "sm_clock_lock_mhz" in card else {}
                gpu = {"ruler_gpu": card["gpu"]} if card.get("gpu") else {}
                held = {"ruler_clock_held_mhz": card["sm_mhz_held"]} if card.get("sm_mhz_held") else {}  # what the
                return {**report["calibration"], **clock, **gpu, **held}  # card held under that lock (ruler.validate)
    return {}


def fingerprint(root):
    """What every session measures of its own card (the ruler's fingerprint, scripts/session.sh): {graph_node_us,
    bandwidth, gpu, nsys, clock (the SM clock lock)}, from launch_latency.json, bandwidth.json, host.json and
    clock.json; missing figures None, an unrecorded clock omitted (distinct from explicitly unlocked)."""
    d = Path(root) / "ruler"
    read = lambda name: json.loads((d / name).read_text()) if (d / name).exists() else {}  # noqa: E731
    lat, host, clock = read("launch_latency.json"), read("host.json"), read("clock.json")
    return {"graph_node_us": lat.get("graph_node_us"), "bandwidth": bandwidth([root]),
            "gpu": (host.get("gpu") or {}).get("name"), "nsys": (host.get("tools") or {}).get("nsys"),
            **({"clock": clock["sm_clock_lock_mhz"]} if "sm_clock_lock_mhz" in clock else {})}


def same_tracer(mine, theirs):
    """Whether two fingerprints are one tracer: a launch cost, the trace's lengthening and what a PDL launch leaves
    exposed are the tracer's at one SM clock, so the same GPU and Nsight version (or not recorded), and the same lock
    where both recorded one (explicit unlocked is evidence, not an unknown clock)."""
    same = all(not (mine.get(k) and theirs.get(k) and mine[k] != theirs[k]) for k in ("gpu", "nsys"))
    if "clock" in mine and "clock" in theirs:
        same &= mine["clock"] == theirs["clock"]
    return same


def lend(cal, mine, theirs):
    """The calibration for a session whose own ruler did not pass, from another's (cal) and the two fingerprints:
    - the session's own empty graph node is always its node floor, never dropped: the launch term stays modelled;
    - node tracing's inflation is the tracer's on this kind of card, so it is lent wherever the GPU, Nsight version and
      SM clock lock are the same (or not recorded), and with it what a PDL launch leaves exposed (pdl_release_ns, the
      card's at that clock, and pdl_wide_exposed_ns_per_node, the widest node's); calibration_lent says whether it
      was, since without it node-traced kernel time keeps the
      trace's lengthening, and the anatomy's split of the step cannot be read (patterns leaves it out);
    - the spin graph's end of the node-cost range and the DRAM counter scale are this rental's card's, lent only when
      its own empty node and bandwidth are within card_match of the lender's; otherwise the range is the own node.
    A session that measured nothing of its own (the validation runs, read with a ruler run apart on the same box)
    borrows cal whole, and calibrations marks it unchecked: every study session runs its own ruler, so it never
    borrows that way. {} when there is nothing at all."""
    if not mine["graph_node_us"] and not mine["bandwidth"]:
        return {**cal, "calibration_lent": True} if cal else {}
    own = mine["graph_node_us"]
    out = {"graph_node_floor_us": own, "graph_node_floor_range_us": [own, own]} if own else {}
    tracer = same_tracer(mine, theirs)
    if cal and tracer and "node_trace_inflation_ns" in cal:
        out["node_trace_inflation_ns"] = cal["node_trace_inflation_ns"]
        out |= {k: cal[k] for k in ("ruler_clock_mhz", "ruler_gpu", "pdl_release_ns", "pdl_wide_exposed_ns_per_node",
                                    "pdl_wide_grid") if k in cal}
    near = lambda a, b: bool(a and b) and abs(a / b - 1) <= TOLERANCES["card_match"]  # noqa: E731
    if cal and tracer and near(own, theirs["graph_node_us"]) and near(mine["bandwidth"], theirs["bandwidth"]):
        out |= {k: cal[k] for k in ("dram_counter_scale",) if k in cal}
        out["graph_node_floor_range_us"] = sorted([own, cal.get("graph_node_spin_us", own)])
    return {**out, "calibration_lent": "node_trace_inflation_ns" in out} if out else {}


def calibrations(roots, calibrate=()):
    """(per session {session: (bandwidth, calibration, ruler_from)}, the fallback (session, bandwidth)): each session
    is read by its own card's numbers (a rented card differs from the next; every study session runs the full ruler,
    scripts/session.sh). A session without its own bandwidth reads the calibrating session's (S1), else the first
    that has one (bandwidth_from names it); one whose ruler did not pass keeps its own empty node and borrows only as
    lend() allows, ruler_from naming the lender where node tracing's inflation was lent, with " (unchecked)" where the
    session measured nothing to match it by."""
    every = list(map(Path, [*roots, *calibrate]))
    found = {r.name: (bandwidth([r]), ruler_calibration([r]), fingerprint(r)) for r in every}
    order = [Path(r).name for r in [*calibrate, *roots]]
    bw_from = next((n for n in order if found[n][0]), None)
    lender = next((n for n in order if found[n][1]), None)
    own = {}
    for name, (bw, cal, mine) in found.items():
        if cal:
            if "pdl_release_ns" not in cal and lender and lender != name and same_tracer(mine, found[lender][2]) \
                    and "pdl_release_ns" in found[lender][1]:
                # an own ruler that passed without the PDL chain's release (the chain skipped or incomplete, or its
                # plateau unread) would price every overlapped kernel with the bare model, 2.7x apart on llama.cpp,
                # and nothing would say so: the release is the card's at that lock, so the calibrating session's
                # stands in, named (pdl_release_from; the cell's launch_release_from)
                theirs = found[lender][1]
                cal = {**cal, "pdl_release_ns": theirs["pdl_release_ns"], "pdl_release_from": lender,
                       **{k: theirs[k] for k in ("pdl_wide_exposed_ns_per_node", "pdl_wide_grid") if k in theirs}}
            own[name] = (bw, cal, name)
            continue
        lent = lend(found[lender][1] if lender else {}, mine, found[lender][2] if lender else mine)
        if "pdl_release_ns" in lent:
            lent["pdl_release_from"] = lender
        unchecked = not (mine["graph_node_us"] or mine["bandwidth"])
        lender_named = (lender + (" (unchecked)" if unchecked else "")) if lent.get("calibration_lent") else None
        own[name] = (bw, lent, lender_named)
    return own, (bw_from, found[bw_from][0] if bw_from else None)
