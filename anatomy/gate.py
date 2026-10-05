"""The gates as code: what a session must show before the next is booked, read from tidy's and the ruler's outputs and
the grid, one line per item (pass, fail or untested), non-zero when anything fails.

    python -m anatomy.gate smoke --session ~/decode-anatomy-runs/SMOKE-x --grid configs/grid/smoke.yaml
    python -m anatomy.gate g1    --session ~/decode-anatomy-runs/S1-x    --grid configs/grid/v1.yaml
    python -m anatomy.gate g2    --session ~/decode-anatomy-runs/S2-x    --grid configs/grid/v1.yaml --calibrate <S1>

The gate first builds its inputs with the command it prints (anatomy.tidy on the session, calibrated by itself or
--calibrate, then anatomy.report.patterns), into <session>/gate. It reads cells.parquet, noise.csv, sentinels.csv,
agreement.csv, tolerances-suggested.yaml and patterns.json there, the ruler's report, the host's files and each unit's
meta, status and clock log. It never reads a trace: an hour of reading becomes a few minutes, and
items a reader could not judge from tidy's outputs (G1's slot gate) are judged from them now.

Items:
- smoke: every unit the grid plans for the session there and finished with its capture's evidence for every batch
  (required evidence); every unit ok at the deployed commit; the session's own cells accepted; the hostload loops'
  share of their cores and the busy-time test run on the hostload cells; one export per traced window; the anchors
  mark the steps; the DRAM counters read and calibrated on every planned cell; the clock locked (the host's file,
  every unit's provenance, each unit's clock log where nothing slowed the clock); the ruler's report ok with its PDL
  chain run and the counter scale present; no llama-server slot problem; the greedy probe recorded for every engine,
  no probe error; every image's --help in the provenance; nothing secret-looking left.
- g1: the required evidence, the exports, the DRAM counters, the ruler and the clock as above; the noise block
  complete (every cell the grid's noise block names, from every process, no reading excluded, or the exceptions with
  their reasons) and no engine above its band (tolerances-suggested.yaml's g1_failures); the session's own cells
  accepted; the busy-time test run and passed; llama-server's slots clean on the noise block's servers and the cells,
  the traced waves included; the A/A rates against chance and each host fit's interval method, printed.
- g2: every planned sentinel with a valid start and end reading (a redo round's pair too) and every closing sentinel
  within its drift; every unit of the session with a record or an explained failure, the recorded workload the
  grid's; the flagged units' redo round run.
What the gate cannot judge is checked by hand: the nvbandwidth control, the upload time per unit, the hours.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

from anatomy import grid as G
from anatomy import tidy
from anatomy.decompose import suggested_overlap_model
from anatomy.metrics import TOLERANCES, rejection
from anatomy.report import patterns
from servebench import env
from servebench.nsys.capture import KINDS
from servebench.nsys.scrub import leaks
from servebench.results import is_sentinel, read_jsonl, read_meta, read_status, unit_dir

PASS, FAIL, UNTESTED = "pass", "fail", "untested"
SHARE_MIN = 0.8      # a hostload loop's share of its core: the spinners took 85-92% in validation
BLOCK_SESSION = {"smoke": "SMOKE", "g1": "S1", "g2": "S2"}


class Gate:
    def __init__(self, kind):
        self.kind, self.items = kind, []

    def item(self, name, status, detail=""):
        self.items.append((name, status, detail))
        return status

    def ok(self):
        return all(s != FAIL for _, s, _ in self.items)

    def report(self, out=None):
        out = out or sys.stdout
        width = max(len(n) for n, _, _ in self.items)
        for name, status, detail in self.items:
            print(f"{name:<{width}}  {status:<8}  {detail}", file=out)
        failed = [n for n, s, _ in self.items if s == FAIL]
        print(f"gate {self.kind}: {'passed' if not failed else 'FAILED: ' + ', '.join(failed)}", file=out)


def few(xs, n=4):
    """A list in a detail: its first few, and how many more."""
    xs = list(xs)
    return ", ".join(map(str, xs[:n])) + (f" (+{len(xs) - n} more)" if len(xs) > n else "")


def build(session, out, calibrate=None, bandwidth=None):
    """tidy and patterns into out: the gate's inputs, with the command it ran. Returns the tidy error, or None."""
    out.mkdir(parents=True, exist_ok=True)
    argv = [str(session), "--calibrate", str(calibrate or session), "--out", str(out)]
    if bandwidth:
        argv += ["--bandwidth", str(bandwidth)]
    print(f"gate: python -m anatomy.tidy {' '.join(argv)}")
    try:
        tidy.main(argv)
    except SystemExit as e:
        return str(e)
    cells = out / "cells.parquet"
    if cells.exists():
        print(f"gate: python -m anatomy.report.patterns {cells} --json {out / 'patterns.json'}")
        try:
            patterns.main([str(cells), "--json", str(out / "patterns.json")])
        except SystemExit as e:
            return f"patterns: {e}"
    return None


def read_csv(path):
    return list(csv.DictReader(path.open())) if path.exists() else []


def read_yaml(path):
    return yaml.safe_load(path.read_text()) if path.exists() else None


def units(session):
    """(key, dir, unit dict, status) for every unit of the session, finished or not."""
    return [(u["key"], d, u, status) for _s, d, u, status, _gpu in tidy.units([session])]


def cell_key(r):
    """A cell's key for matching rows to the grid's units, its context included: a long-context cell at batch 1 is
    not bring-up's."""
    return (r["engine"], r["arm"], r["level"], r["model"], int(r["batch"]), int(r.get("prompt_len") or 896),
            int(r.get("gen_len") or 288))


def recorded_differs(d, u):
    """Whether the unit a directory recorded (meta.json's unit) is another workload than the grid's u, its block
    aside: a finished unit of another batch set or context under the same key is not this one's evidence."""
    recorded = (read_meta(d) or {}).get("unit") or {}
    return any(recorded.get(k) != v for k, v in u.to_dict().items() if k != "block")


def expected_cells(grid, block_session, blocks=None):
    """{cell key: unit key} for the session's own blocks (not its noise or sentinels)."""
    out = {}
    for u in G.expand(grid, sessions={block_session}, blocks=blocks):
        for b in u.batches:
            out[(u.engine, u.arm, u.level, u.model, b, u.prompt_len, u.gen_len)] = u.key
    return out


# items ---------------------------------------------------------------------------------------------------------

def units_ok(g, session, commit):
    found = units(session)
    if not found:
        return g.item("units ok", FAIL, "no unit under the session")
    bad = [f"{k}: {(s or {}).get('error') or 'never finished'}" for k, _d, _u, s in found if not (s or {}).get("ok")]
    g.item("units ok", FAIL if bad else PASS, f"{len(found) - len(bad)} of {len(found)} ok; " + few(bad) if bad
           else f"{len(found)} units")
    if commit:
        off = [k for k, d, _u, _s in found if ((read_meta(d) or {}).get("git") or "").split("-dirty")[0] != commit]
        g.item("deployed commit", FAIL if off else PASS,
               f"{len(off)} units ran another commit than {commit}: {few(off)}" if off else f"every unit at {commit}")
    else:
        g.item("deployed commit", UNTESTED, "no --commit to compare meta.json's git with")


def required_evidence(g, grid, block_session, session):
    """Every planned unit and batch must exist and succeed for smoke/G1, including each required capture."""
    wanted, problems = G.session_units(grid, block_session), []
    for u in wanted:
        d = unit_dir(session.name, u, root=session.parent)
        status = read_status(d) or {}
        if not status.get("ok"):
            problems.append(f"{u.key}: {status.get('error') or 'missing successful attempt'}")
            continue
        if recorded_differs(d, u):
            problems.append(f"{u.key}: recorded workload differs from the grid")
            continue
        if u.capture in KINDS:
            evidence = tidy.windows(d, u.capture, u.level, u.batches)
        elif u.capture == "runner":
            evidence = tidy.runner_timing(d, u.engine)
        else:
            evidence = tidy.timing_cells(tidy.reestimated(d, read_jsonl(d / "driver.jsonl")))
        missing = set(u.batches) - evidence.keys()
        if missing:
            problems.append(f"{u.key}: no {u.capture} evidence for batches {sorted(missing)}")
    g.item("required evidence", FAIL if problems or not wanted else PASS,
           few(problems) if problems else f"{len(wanted)} planned units and all their batches" if wanted
           else f"the grid plans no unit for session {block_session}")


def contention(g, session, rows):
    loads = [(k, d) for k, d, u, _s in units(session) if u["capture"] == "hostload"]
    if not loads:
        return g.item("hostload loops", UNTESTED, "no hostload unit")
    weak = []
    for k, d in loads:
        rec = (read_meta(d) or {}).get("contention") or {}
        records = [rec] if "cpu_share" in rec else list(rec.values())
        means = [sum(r["cpu_share"]) / len(r["cpu_share"]) for r in records if r.get("cpu_share")]
        if not means or min(means) < SHARE_MIN:  # each server's loops on average: a loop's share runs through model
            weak.append(f"{k}: {'no loop measured' if not means else f'loops at {min(means):.0%} of their cores'}")
    g.item("hostload loops", FAIL if weak else PASS,  # load and graph capture too, so one loop's minimum is no judge
           few(weak) if weak else f"{len(loads)} units, every server's loops above {SHARE_MIN:.0%} of their cores")
    cells = [r for r in rows if "hostload" in (r.get("sources") or {})]
    if not cells:
        return g.item("busy-time test", FAIL, "no cell with a hostload trace")
    failed = [tidy_id(r) for r in cells if r.get("busy_invariant") is False]
    held = [r for r in cells if r.get("busy_invariant") is True]
    untested = [tidy_id(r) for r in cells if r.get("busy_invariant") is None]  # a GPU-bound cell cannot stretch
    if failed:
        g.item("busy-time test", FAIL, f"busy time moved under load: {few(failed)}")
    elif not held:
        g.item("busy-time test", FAIL, f"contention stretched no cell's step enough to test anything: {few(untested)}")
    else:
        more = f"; {len(untested)} GPU-bound cells stretched too little to test: {few(untested)}" if untested else ""
        g.item("busy-time test", PASS, f"held on {len(held)} cells" + more)


def tidy_id(r):
    return f"{r['engine']}/{r['arm']}/{r['level']}/{str(r['model']).split('/')[-1]}/b{r['batch']}"


def exports(g, session):
    traced = [(k, d, u) for k, d, u, _s in units(session) if u["capture"] in KINDS]
    if not traced:
        return g.item("trace exports", UNTESTED, "no traced unit")
    short = []
    for k, d, u in traced:  # paired with their batches by tidy's one rule (L1 exports are numbered, L2's per batch)
        windows = sum(1 for r in read_jsonl(d / "driver.jsonl") if r.get("kind") == "profile")
        got = len(tidy.windows(d, u["capture"], u["level"], u["batches"]))
        if got != windows or not windows:
            short.append(f"{k}: {got} exports for {windows} windows")
    g.item("trace exports", FAIL if short else PASS, few(short) if short else f"{len(traced)} units, one per window")


def anchors(g, rows):
    traced = [r for r in rows if r.get("anchor_ok") is not None or r.get("node_anchor_ok") is not None]
    if not traced:
        return g.item("step anchors", UNTESTED, "no traced cell")
    off = [tidy_id(r) for r in traced if r.get("anchor_ok") is False or r.get("node_anchor_ok") is False]
    g.item("step anchors", FAIL if off else PASS, few(off) if off else f"mark the steps on {len(traced)} cells")


def dram(g, session, rows):
    captures = [u for _k, _d, u, _s in units(session) if u["capture"] == "metrics"]
    if not captures:
        return g.item("DRAM counters", UNTESTED, "no metrics unit")
    expected = {cell_key({**u, "batch": b}) for u in captures for b in u["batches"]}
    read = {cell_key(r) for r in rows if r.get("metrics_valid") is True and r.get("dram_over_model") is not None}
    missing = expected - read
    g.item("DRAM counters", FAIL if missing else PASS,
           f"missing or invalid counters for {few('/'.join(map(str, k[:5])) for k in sorted(missing))}"
           if missing else f"read and calibrated on {len(read)} cells")


def clock(g, session, grid):
    mhz = grid.get("clock_mhz")
    if not mhz:
        return g.item("clock lock", UNTESTED, "the grid locks no clock")
    host = session / "host" / "clock-lock.txt"
    state = host.read_text().strip() if host.exists() else "no host/clock-lock.txt"
    problems = [] if state == "locked" else [state]
    # the reference is the clock the ruler held under the same lock, a reading of the card (tidy.clock_reference, the
    # cells' own): a card may hold less than asked (the H100 PCIe VM held 1620 MHz of 1755, idle and busy alike). The
    # study's grid fails on that, since its cells would run at a clock the page does not name: set clock_mhz to what
    # the card holds, or rent another box; a validation grid notes it
    ruler_held = tidy.clock_reference([session])[session.name]
    reference = ruler_held or mhz
    off = ruler_held is not None and abs(ruler_held / mhz - 1) > TOLERANCES["clock_lock"]
    if off and not grid.get("validation"):
        problems.append(f"the card holds {ruler_held:.0f} MHz of the {mhz} asked (the ruler's clock log): set "
                        "clock_mhz to what it holds, or another box")
    unlocked_mark, logs, slowed = [], [], []
    for k, d, u, _s in units(session):
        unlocked = is_sentinel(u.get("block")) and u["block"].endswith("-unlocked")
        want = None if unlocked else mhz
        if (read_meta(d) or {}).get("sm_clock_lock_mhz") != want:
            unlocked_mark.append(k)
        if not unlocked:  # one rule for a unit's clock, the cells' (tidy.unit_conditions): the lock is judged where
            # nothing slowed the clock; a unit slowed in every busy sample (the power cap) is noted, never failed
            conditions = tidy.unit_conditions(d, tidy.own_processes(u), ruler_held)
            if conditions["sm_mhz"] is None:
                logs.append(f"{k}: no usable clock samples")
            elif conditions["sm_mhz_unthrottled"] is None:
                slowed.append(k)
            elif conditions.get("clock_problem"):
                logs.append(f"{k}: {conditions['clock_problem']}")
    if unlocked_mark:
        problems.append(f"provenance without the lock on {few(unlocked_mark)}")
    if logs:
        problems.append(f"clock logs off {reference:.0f} MHz: {few(logs)}")
    if not any((d / "clocks.csv").exists() for _k, d, _u, _s in units(session)):
        problems.append("no unit has a clocks.csv")
    detail = (f"the lock asked {mhz} MHz and the card held {reference:.0f} (the ruler's clock log and every unit's)"
              if off else f"{mhz} MHz throughout")
    detail += (f" where nothing slowed the clock; slowed (the power cap) in every busy sample, the lock not judged, on "
               f"{few(slowed)}" if slowed else "")
    g.item("clock lock", FAIL if problems else PASS, "; ".join(problems) if problems else detail)


def ruler(g, session):
    path = session / "ruler" / "ruler_report.json"
    if not path.exists():
        return g.item("ruler", FAIL, "no ruler/ruler_report.json")
    rep = json.loads(path.read_text())
    cal = rep.get("calibration") or {}
    problems = [] if rep.get("ok") else [f"report not ok: {few(rep.get('problems') or ['windows off'])}"]
    if "dram_counter_scale" not in cal:
        problems.append("no dram_counter_scale (the counters were not read by the ruler)")
    pdl = cal.get("pdl")
    if isinstance(pdl, str):
        problems.append(f"PDL chain {pdl}")
    elif not cal.get("pdl_chain_ok"):
        problems.append("PDL chain: the dependence did not hold (pdl_chain_ok false)")
    _model, evidence = suggested_overlap_model(cal)  # what the chain says about the launch model's overlap choice
    windows = cal.get("pdl_windows") or []
    worst = max((abs(w["pdl_model_floored_error_ns"]) for w in windows if "pdl_model_floored_error_ns" in w),
                default=None)  # on the plateau the floored model fits by construction: the window below it is the test
    release = (f"{cal['pdl_release_ns']:.0f} ns a node (tail {cal.get('pdl_release_tail_ns')} ns)"
               if cal.get("pdl_release_ns") is not None else f"none ({cal.get('pdl_release_note', 'no chain')})")
    sweep = cal.get("pdl_wide_windows") or []
    wide = ("; nodes of " + "/".join(str(w["grid"]) for w in sweep) + " blocks leave "
            + "/".join(f"{w['pdl_wide_exposed_ns_per_node']:.0f}" for w in sweep) + " ns" if sweep else "")
    ns = lambda v: "none" if v is None else f"{v:.0f}"  # noqa: E731
    detail = (f"PDL release {release}{wide}; model error bare {ns(cal.get('pdl_model_error_ns'))} ns, floored at worst "
              f"{ns(worst)} ns over {len(windows)} windows; overlap model {evidence['verdict']}; node floor "
              f"{cal.get('graph_node_floor_us') or 0:.3f} us")
    g.item("ruler", FAIL if problems else PASS, "; ".join(problems) + "; " + detail if problems else detail)


SLOT_MESSAGES = ("llama-server ran batch", "llama-server reused its graph", "could not cut the log", "a server of",
                 "llama-server's prefill stalled")  # anatomy.llama_log's messages, the last from llama_prefill


def slots(g, session, rows, noise_rows, planned=None):
    """llama-server's slots: every server unit's log holds the launch lines the check reads (a log without them is
    nothing to judge, never a pass), and no cell (its timing or any of its traced waves: slots_problem) or noise
    reading (its timing's reasons, its occupancy's) carries one of llama_slots' problems. planned: the engines the
    grid plans for this session; a plan without llama.cpp (the study) passes the item empty, whatever
    llama.cpp units another grid left in the session directory."""
    from anatomy.llama_log import LAUNCH
    if planned is not None and "llamacpp" not in planned:
        return g.item("llama-server slots", PASS, "no llama.cpp in the plan: nothing to judge")
    servers = [(k, d) for k, d, u, s in units(session)
               if (u["engine"], u["level"]) == ("llamacpp", "L2") and s is not None]
    if not servers:
        return g.item("llama-server slots", UNTESTED, "no llama.cpp server unit")
    blind = [k for k, d in servers if not ((d / "container.log").exists()
                                           and LAUNCH.search((d / "container.log").read_text(errors="replace")))]
    bad = [f"{tidy_id(r)}: {str(r['slots_problem'])[:90]}" for r in rows if r.get("slots_problem")]
    bad += [f"noise {r['level']} b{r['batch']}: {r[field]}" for r in noise_rows
            for field in ("reasons", "occupied_reasons")  # its timing's reasons, and its occupancy readings'
            if r.get("engine") == "llamacpp" and any(m in (r.get(field) or "") for m in SLOT_MESSAGES)]
    problems = ([f"no launch lines in the log of {few(blind)}: nothing to judge"] if blind else []) + \
               ([f"problems: {few(bad)}"] if bad else [])
    g.item("llama-server slots", FAIL if problems else PASS,
           "; ".join(problems) if problems else f"every wave of {len(servers)} server units in one run of slots")


def agreement(g, session, out):
    """The greedy probe: every finished timing unit that has a probe (the engine API's; a server's only where its
    release names a greedy_endpoint, llama-server's /completion) recorded its prompts' tokens, none an error, and no
    comparison agreement.csv made is flagged (an arm that changed tokens, an engine that parted from every other)."""
    def probes(u):
        r = tidy.release_for(u)
        return u["level"] != "L2" or bool(r and r.extra.get("greedy_endpoint"))
    timed = [(k, d, u) for k, d, u, s in units(session)
             if u["capture"] in tidy.TIMING and (s or {}).get("ok") and probes(u)]
    if not timed:
        return g.item("greedy probe", UNTESTED, "no timing unit with a probe finished")
    missing, errors = [], []
    for k, d, _u in timed:
        recs = [r for r in read_jsonl(d / "driver.jsonl") if r.get("kind") == "greedy"]
        if not recs:
            missing.append(k)
        elif bad := [r for r in recs if "error" in r]:
            errors.append(f"{k}: {str(bad[0]['error'])[:80]}")
    rows = read_csv(out / "agreement.csv")
    flagged = [f"{r['kind']} {r['a']} / {r['b']}" for r in rows if r["kind"] != "probe-error"
               and r.get("flagged") in ("True", True)]
    problems = ([f"no probe recorded by {few(missing)}"] if missing else []) + \
               ([f"probe errors: {few(errors)}"] if errors else []) + \
               ([f"flagged: {few(flagged)}"] if flagged else [])
    g.item("greedy probe", FAIL if problems else PASS,
           "; ".join(problems) if problems
           else f"recorded by {len(timed)} units, {len(rows)} comparisons, none flagged")


def help_captured(g, session, grid):
    files = list((session / "host" / "help").rglob("help.txt"))
    want = len(grid["versions_map"])
    g.item("--help captured", PASS if len(files) >= want else FAIL,
           f"{len(files)} of {want} images' --help in host/help")


def scrub(g, session):
    left = leaks(session)
    g.item("scrub", FAIL if left else PASS, f"{len(left)} files still secret-looking: {few(left)}" if left
           else "nothing secret-looking")


def noise(g, grid, block_session, out):
    rows = read_csv(out / "noise.csv")
    block = grid.get("noise") or {}
    if block.get("session") != block_session:
        return g.item("noise block", FAIL, f"the grid has no noise block for session {block_session}: nothing "
                                            "measures the study's noise")
    want = 2 * int(block.get("processes", 3))
    expected, traced = {}, set()
    for u in G.noise(grid, block_session, "start"):
        for b in u.batches:
            key = (u.engine, u.arm, u.level, u.model, b)
            expected[key] = u.capture
            if u.capture == "graph":
                traced.add(key)
    by = {cell_key(r)[:5]: r for r in rows}  # noise rows carry no context: the noise block runs at the study's
    problems = []
    for key in sorted(expected, key=str):
        r = by.get(key)
        name = "/".join(map(str, key))
        if r is None:
            problems.append(f"{name}: no row")
            continue
        if int(r["processes"]) != want or int(r["invalid"]) != 0:
            problems.append(f"{name}: {r['processes']} of {want} processes, {r['invalid']} excluded"
                            + (f" ({r['reasons']})" if r.get("reasons") else ""))
        if key in traced and (int(r.get("occupied_processes") or 0) != want or int(r.get("occupied_invalid") or 0)):
            problems.append(f"{name}: {r.get('occupied_processes') or 0} of {want} occupancy readings, "
                            f"{r.get('occupied_invalid') or 0} excluded"
                            + (f" ({r['occupied_reasons']})" if r.get("occupied_reasons") else ""))
    g.item("noise block", FAIL if problems else PASS,
           few(problems, 6) if problems else f"{len(expected)} cells from {want} processes each"
           + (f", {len(traced)} with their occupancy" if traced else ""))
    suggested = read_yaml(out / "tolerances-suggested.yaml") or {}
    failures = suggested.get("g1_failures") or []
    if not suggested:
        return g.item("engines within bands", UNTESTED, "no tolerances-suggested.yaml")
    per, bands = suggested.get("per_engine") or [], suggested.get("bands") or {}

    def setter(stat):  # the engine and level whose reading set the global value (the largest inside the band)
        hi = (bands.get(stat) or [None, None])[1]
        inside = [p for p in per if p.get(stat) is not None and (hi is None or p[stat] <= hi)]
        best = max(inside, key=lambda p: p[stat], default=None)
        return f"{stat} {suggested.get(stat)}" + (f" ({best['engine']} {best['level']})" if best else "")
    g.item("engines within bands", FAIL if failures else PASS,
           few(f"{f['engine']} {f['level']} {f['statistic']} {f['value']} above {f['most']}" for f in failures)
           if failures else "; ".join(setter(k) for k in ("step_noise", "agree", "range_ok", "sentinel_drift")
                                      if k in suggested))


def own_cells(g, grid, block_session, rows):
    expected = expected_cells(grid, block_session)
    if not expected:
        return g.item("cells accepted", FAIL, f"the grid has no blocks for session {block_session}")
    by = {}
    for r in rows:
        by.setdefault(cell_key(r), r)
    rejected, missing = [], []
    for key in sorted(expected, key=str):
        r = by.get(key)
        if r is None:
            missing.append("/".join(map(str, key[:5])))
        elif rejection(r):
            rejected.append(f"{'/'.join(map(str, key[:5]))}: {rejection(r)[:90]}")
    problems = ([f"no cell for {few(missing)}"] if missing else []) + \
               ([f"rejected: {few(rejected)}"] if rejected else [])
    g.item("cells accepted", FAIL if problems else PASS,
           "; ".join(problems) if problems else f"{len(expected)} cells of the session's blocks")


def statistics(g, out):
    """Printed, never a gate: the A/A rates against what the rule itself gives on a calibrated null (derive.Noise.aa's
    own simulation, not derive.chance's pooled degrees of freedom), and each host fit's interval method."""
    path = out / "patterns.json"
    if not path.exists():
        return g.item("A/A rates (info)", UNTESTED, "no patterns.json")
    rep = json.loads(path.read_text())
    stats, fits = rep.get("stats") or {}, rep.get("fits") or []
    aa = stats.get("aa") or {}
    parts = []
    for name in ("processes", "sentinels"):
        x = aa.get(name)
        if x:
            parts.append(f"{name} {x['beyond']}/{x['pairs']} beyond ({x['rate']:.1%})"
                         + (f", {x['expected']:.1%} expected under the rule's own null" if x.get("expected") is not None
                            else ""))
    g.item("A/A rates (info)", PASS if parts else UNTESTED, "; ".join(parts) if parts else "no A/A pairs yet")
    methods = [f"{f.get('engine')}: {f.get('interval_method')}" for f in fits if f.get("interval_method")]
    g.item("host fits (info)", PASS if methods else UNTESTED, few(methods) if methods else "no host fit identified")


def sentinels(g, out, session, grid=None):
    rows = [r for r in read_csv(out / "sentinels.csv") if r.get("session") == session.name]  # not the calibrating S1's
    closing = [r for r in rows if r["position"] == "end"]
    if not closing and not (grid or {}).get("sentinels"):
        return g.item("sentinel drift", UNTESTED, "no sentinel planned or recorded for this session")
    off = []
    for u in G.sentinels(grid, "start") if grid else ():
        for b in u.batches:
            for position in ("start", "end"):
                found = [r for r in rows if (r.get("engine"), r.get("arm"), r.get("level"), r.get("model"),
                                            int(r["batch"]), r.get("position"), r.get("redo")) ==
                         (u.engine, u.arm, u.level, u.model, b, position, "")]
                if not found or not all(r.get("valid") in ("True", True) for r in found):
                    off.append(f"{u.engine}/{u.level}/{u.model}/b{b}: missing or invalid {position} sentinel")
    for r in closing:
        if r.get("valid") not in ("True", True) or r.get("drift_ok") not in ("True", True):
            detail = r.get("problem") or (f"{float(r['drift']):+.2%}" if r.get("drift") else "no valid opening reading")
            off.append(f"{r['engine']} {r.get('level')} {str(r['model']).split('/')[-1]} b{r['batch']}: {detail}")
    # a redo round's sentinels: every locked sentinel read at the round's start and end, both valid. Its unlocked
    # sentinel (session.sh runs it with the round) has no pair by design: it is reported against the session's locked
    # opening reading and never gated, as the main round's is (S2-20261003's redo round first showed it here)
    locked = [r for r in rows if r.get("redo") and r["position"] != "unlocked"]
    redo = {(r["engine"], r.get("level"), r["arm"], r["model"], r["batch"], r["redo"]) for r in locked}
    for key in redo:
        pair = [r for r in locked
                if (r["engine"], r.get("level"), r["arm"], r["model"], r["batch"], r.get("redo")) == key]
        if {r["position"] for r in pair} != {"start", "end"} or any(r.get("valid") not in ("True", True) for r in pair):
            off.append(f"{'/'.join(map(str, key))}: missing or invalid redo sentinel pair")
    g.item("sentinel drift", FAIL if off else PASS,
           few(off) if off else f"{len(closing)} closing readings within drift")


def coverage(g, grid, block_session, session):
    wanted = G.session_units(grid, block_session)
    if not wanted:
        return g.item("coverage", UNTESTED, "the grid has no units for this session")
    missing = []
    for u in wanted:
        d = unit_dir(session.name, u, root=session.parent)
        status = read_status(d)
        if status is None or (not status.get("ok") and not status.get("error")):
            missing.append(u.key)
        elif status.get("ok") and recorded_differs(d, u):
            missing.append(f"{u.key}: recorded workload differs from the grid")
    g.item("coverage", FAIL if missing else PASS,
           f"{len(missing)} of {len(wanted)} units without a record or an explained failure: {few(missing)}"
           if missing else f"{len(wanted)} units, each with a record or an explained failure")


def redo_round(g, session):
    reruns = session / "qc" / "reruns.csv"
    rows = read_csv(reruns)
    if not rows:
        return g.item("redo round", UNTESTED, "no qc/reruns.csv (no unit was flagged, or QC did not run)")
    since = reruns.stat().st_mtime  # a redo wrote the unit's provenance after the list (anatomy.grid.redone's rule)
    flagged = [r["unit"] for r in rows]
    undone = [k for k in flagged if ((read_meta(session / k) or {}).get("written") or 0) <= since]
    g.item("redo round", FAIL if undone else PASS,
           f"{len(undone)} flagged units never rerun: {few(undone)}" if undone
           else f"{len(flagged)} flagged units rerun")


# the gates --------------------------------------------------------------------------------------------------

def run(kind, session, grid_path, block_session=None, calibrate=None, bandwidth=None, out=None, commit=None):
    session, grid = Path(session), G.load(grid_path)
    if block_session is None:  # the gate's usual session, or a grid of one session's (the smoke grid's SMOKE)
        present = {b.get("session") for b in grid["blocks"].values()} | {(grid.get("noise") or {}).get("session")}
        present.discard(None)
        default = BLOCK_SESSION[kind]
        block_session = default if default in present or len(present) != 1 else next(iter(present))
    out = Path(out) if out else session / "gate"
    g = Gate(kind)
    error = build(session, out, calibrate, bandwidth)
    rows = tidy.load_cells(out / "cells.parquet") if (out / "cells.parquet").exists() else []
    g.item("inputs built", FAIL if error else PASS, error or f"{len(rows)} cells in {out}")
    noise_rows = read_csv(out / "noise.csv")
    planned = {u.engine for u in G.session_units(grid, block_session)}  # the grid's engines for this session
    if kind == "smoke":
        required_evidence(g, grid, block_session, session)
        own_cells(g, grid, block_session, rows)
        units_ok(g, session, commit)
        contention(g, session, rows)
        exports(g, session)
        anchors(g, rows)
        dram(g, session, rows)
        clock(g, session, grid)
        ruler(g, session)
        slots(g, session, rows, noise_rows, planned)
        agreement(g, session, out)
        help_captured(g, session, grid)
        scrub(g, session)
    elif kind == "g1":
        required_evidence(g, grid, block_session, session)
        exports(g, session)
        dram(g, session, rows)
        ruler(g, session)
        clock(g, session, grid)
        noise(g, grid, block_session, out)
        own_cells(g, grid, block_session, rows)
        contention(g, session, rows)
        slots(g, session, rows, noise_rows, planned)
        statistics(g, out)
    elif kind == "g2":
        sentinels(g, out, session, grid)
        coverage(g, grid, block_session, session)
        redo_round(g, session)
    return g


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=sorted(BLOCK_SESSION))
    p.add_argument("--session", type=Path, required=True, help="the session's raw directory")
    p.add_argument("--grid", type=Path, default=Path("configs/grid/v1.yaml"))
    p.add_argument("--block-session", help=f"the grid session the units came from (default per gate: {BLOCK_SESSION})")
    p.add_argument("--calibrate", type=Path, help="the session whose ruler and noise block calibrate (default: itself)")
    p.add_argument("--bandwidth", type=float, help="a bandwidth in B/s where no ruler ran (the CPU rehearsal)")
    p.add_argument("--out", type=Path, help="where tidy's and patterns' outputs go (default: <session>/gate)")
    p.add_argument("--commit", help="the commit deployed to the box (default: this checkout's); every unit must "
                                    "have run it")
    a = p.parse_args(argv)
    commit = a.commit or (env.git_sha(G.ROOT) or "").split("-dirty")[0] or None
    g = run(a.kind, a.session, a.grid, a.block_session, a.calibrate, a.bandwidth, a.out, commit)
    g.report()
    return 0 if g.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
