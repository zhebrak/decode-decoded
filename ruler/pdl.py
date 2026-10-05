"""A ground truth for the launch term's overlap model: a chain of graph nodes of known work, with and without
programmatic dependent launch (PDL), traced node by node and timed untraced.

    nsys profile <node-mode args> --capture-range=cudaProfilerApi --capture-range-end=repeat -o /out/pdl-node \
        python3 /work/ruler/pdl.py --mode node --out /out
    python3 /work/ruler/pdl.py --mode none --out /out         (ruler.run runs both)
    python -m ruler.validate ~/decode-anatomy-runs/<session>/ruler      # host side: pdl_calibration

The model under test (anatomy.decompose.exposed_launch_ms, servebench.nsys.parse.overlaps): a kernel that starts before
its same-stream predecessor ends overlaps it by (predecessor end - its start), and is charged max(0, floor - overlap),
floor being an empty graph node's cost on the card (graph_node_floor_us). llama.cpp's decode launches nearly every
kernel with PDL (392 of 397 kernels a step overlap their predecessor, deciles 0.48-4.4 us), so the model prices most of
its launch term. Before its first hardware run it had no ground truth: under PDL an early dependent waits at its
grid-dependency barrier, so what stays exposed after the predecessor ends may be the dependency release, not
floor - overlap; and node-mode tracing lengthens every kernel by node_trace_inflation_ns (105-120 ns on the H100s),
which inflates a measured overlap. The first hardware run (H100 PCIe, session PCIE-20261003) read 1,081 ns a node
exposed without PDL against a 1,129 ns empty node, and with PDL a plateau of 535 ns at tails of 1, 2 and 4 us (674 at
0.5 us): a dependent launch hides about half of a node's cost and no more, so the model now charges an overlapped
kernel at least that release (pdl_release_ns, measured per card; anatomy.decompose.exposed_launch_ms).

The chain: `nodes` Triton kernels of one thread each, captured in one CUDA graph, each doing `work_us` of
%globaltimer spin (clock-independent, as synthetic_busy.py's) and then storing x + 1 to a shared counter after loading
it, so the chain is a true data dependence: a PDL release that let a node run early would show as a wrong counter
(chain_ok in the truth). Every node measures its own work on the globaltimer (the spin overshoots its target by a
read or two), so the truth is not the requested figure. Two variants, one capture window each:
- nopdl: ordinary stream-ordered launches, the control: its exposed time per node is what an empty node's cost
  becomes beside real work;
- pdl, one window per tail_ns in --tails: the node launches with launch_pdl=True, waits at gdc_wait() before touching
  the counter, spins work_us - tail_ns, calls gdc_launch_dependents(), and spins the last tail_ns: its dependent starts
  about tail_ns before it ends (the traced overlap, plus the launch's own latency), the knob that sweeps the overlap
  across the deciles llama.cpp's trace shows (0.48-4.4 us); the tails run from 128 ns (four steps of a 32 ns timer),
  since only overlaps under about 0.7 us tell the raw and the corrected overlap model apart;
- pdl-wide, one window at WIDE_TAIL_NS per grid in --wide (8, 64, twice the SMs, 1024 blocks a node by default), the
  first block alone writing: how the release read from one-block nodes grows towards a real kernel's.
The untraced run (--mode none) replays each window's graph back to back between two CUDA events: the device time per
node, less the node's own measured work, is the truth (exposed_ns_per_node), the time a node adds beyond its work with
nothing queued behind it. The traced run (--mode node) writes the same windows' truth beside nsys's exports; each
window's export is numbered from 1 in window order. The one launch of a graph per replay adds a gap the nodes divide
(1-2.5 us over `nodes`, tens of ns each at the default 200).

Triton's PDL:
- the launch keyword: kernel[grid](..., launch_pdl=True) (triton third_party/nvidia/backend/driver.py, `launch_pdl`
  passed to the launcher from the compiled kernel's metadata; vLLM 0.30.0 passes it the same way, e.g.
  vllm/models/kimi_k3/nvidia/ops/attn_res.py:247);
- in-kernel: tl.extra.cuda.gdc_wait() (griddepcontrol.wait: blocks until the prior kernel's memory operations are
  visible) and tl.extra.cuda.gdc_launch_dependents() (griddepcontrol.launch_dependents: a hint that the next kernel may
  launch once every block has called it or finished), third_party/nvidia/language/cuda/gdc.py; both are safe to execute
  when PDL is off;
- hardware: griddepcontrol is Hopper (sm_90) and later; vLLM gates its own PDL on compute capability >= 9
  (vllm/platforms/cuda.py:767, is_arch_support_pdl);
- CUDA graphs: PDL launches captured in a stream become programmatic edges; a Triton PDL reproducer under
  torch.cuda.CUDAGraph on sm_90 (triton-lang/triton issue 11905, Triton 3.7.1) replays them, so capture works there;
- the version: vLLM 0.30.0 pins torch 2.13.0 (requirements/cuda.txt), whose wheel pins its own Triton; which one the
  pinned image carries could not be checked from here, so the run records triton.__version__ in the truth.
A Triton without the calls, a GPU below sm_90, or a failed trial launch writes {"skipped": reason} instead.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

NS = 1000
MODES = ("node", "none")


def build_kernel():
    """The chain's Triton kernel, or (None, reason) when this Triton lacks PDL's calls."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None, "triton not installed"
    cuda = getattr(getattr(tl, "extra", None), "cuda", None)
    if not (hasattr(cuda, "gdc_wait") and hasattr(cuda, "gdc_launch_dependents")):
        return None, f"triton {triton.__version__} has no tl.extra.cuda.gdc_wait / gdc_launch_dependents"

    @triton.jit
    def _node(counter, dur, idx, work_ns, tail_ns, PDL: tl.constexpr):
        pid = tl.program_id(0)  # one block in the chain's windows; many in the wide one, every block spinning, the
        # first alone writing the counter and the duration, so the chain stays one data dependence
        if PDL:
            tl.extra.cuda.gdc_wait()  # before the first read of what the predecessor wrote
        x = tl.load(counter)
        t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                       pack=1)
        now = t0
        while now - t0 < work_ns - tail_ns:
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                            pack=1)
        if PDL:
            tl.extra.cuda.gdc_launch_dependents()
        while now - t0 < work_ns:
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                            pack=1)
        if pid == 0:
            tl.store(counter, x + 1)
            tl.store(dur + idx, now - t0)

    return _node, None


def build_tick():
    """A Triton kernel that reads %globaltimer TICK_READS times in a row and stores the smallest positive step it saw,
    or (None, reason). The spin's tails (0.5-4 us) and the work subtracted from the truth are timed by that register,
    whose update rate is the implementation's: some GPUs are reported to tick about every microsecond unless a profiler
    is attached, which would give the traced and untraced runs different tails."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None, "triton not installed"

    @triton.jit
    def _tick(out, seed, N: tl.constexpr):
        x = tl.load(seed)
        prev = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                         pack=1)
        best = prev * 0 + 4611686018427387904  # larger than any step; the loop narrows it
        for _ in range(N):
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64,
                                            is_pure=False, pack=1)
            d = now - prev
            best = tl.where((d > 0) & (d < best), d, best)
            prev = now
        tl.store(out, best)

    return _tick, None


TICK_READS = 4096
TICKS_PER_TAIL = 4  # a tail must span this many timer steps for its window to count (the timer's own quantisation)


def timer_step_ns(torch, seed):
    """The smallest positive %globaltimer step one thread sees (build_tick), ns; (None, why) when it cannot run."""
    tick, why = build_tick()
    if tick is None:
        return None, why
    try:
        out = torch.zeros(1, dtype=torch.int64, device="cuda")
        tick[(1,)](out, seed, N=TICK_READS)
        torch.cuda.synchronize()
        return int(out.item()), None
    except Exception as e:  # noqa: BLE001  whatever the compiler or driver raises is the reason
        return None, f"{type(e).__name__}: {e}"[:200]


def chain(node, counter, dur, nodes, work_ns, tail_ns, pdl, grid=1):
    """Launch the chain on the current stream (a capture's, or eager); grid: blocks per node."""
    kw = {"launch_pdl": True} if pdl else {}
    for i in range(nodes):
        node[(grid,)](counter, dur, i, work_ns, tail_ns, PDL=pdl, **kw)


WIDE_TAIL_NS = 2000  # the wide windows' tail: on the plateau, where the one-block chain's release is read


def wide_grids(spec, sms):
    """The wide windows' grids from --wide: integers, or <n>sm for n times the card's SMs; sorted, without repeats or
    the one-block grid the plain windows already run."""
    grids = set()
    for t in spec.split(","):
        t = t.strip()
        if t:
            grids.add(int(t[:-2]) * sms if t.endswith("sm") else int(t))
    return sorted(g for g in grids if g > 1)


def wide_summary(wide):
    """The wide windows sorted by grid (pdl_wide_windows) and, for the largest grid, the scalars the gate prints and
    the cells may take: pdl_wide_grid, pdl_wide_exposed_ns_per_node, pdl_wide_overlap_ns_traced, pdl_wide_chain_ok."""
    if not wide:
        return {}
    wide = sorted(wide, key=lambda w: w["grid"])
    largest = wide[-1]
    return {"pdl_wide_windows": wide, "pdl_wide_grid": largest["grid"],
            **{k: v for k, v in largest.items() if k != "grid"}}


def windows_spec(tails, wide=()):
    """(variant, tail_ns, grid) in window order: the control first, then each PDL tail with one block a node, then one
    PDL window at WIDE_TAIL_NS per grid in `wide`: how the release the one-block chain measures grows with the blocks
    a node runs, towards a real kernel's (llama.cpp's GEMVs run thousands)."""
    return [("nopdl", 0, 1), *(("pdl", t, 1) for t in tails), *(("pdl-wide", WIDE_TAIL_NS, g) for g in wide)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True,
                   help="node: the replays run inside cudaProfilerApi capture windows (ruler.run's nsys node-mode "
                        "command); none: no profiler, the untraced timing the truth takes")
    p.add_argument("--nodes", type=int, default=200)
    p.add_argument("--work-us", type=float, default=10.0, help="spin per node")
    p.add_argument("--tails", default="128,250,375,500,625,750,1000,2000,4000",
                   help="ns each PDL node runs after releasing its dependent: 128 is TICKS_PER_TAIL steps of a 32 ns "
                        "%%globaltimer, the shortest the chain accepts; below about 0.7 us of overlap the raw and the "
                        "corrected model charge differently, where llama.cpp's shortest overlaps (0.48 us) sit; "
                        "375-750 resolve the bump both H100s showed at the 500 ns tail (808 ns exposed on the SXM "
                        "against a 488 ns plateau)")
    p.add_argument("--wide", default="8,64,2sm,1024",
                   help="grids of the wide PDL windows (windows_spec), blocks per node, comma-separated; <n>sm is n "
                        "times the card's SMs; empty for none. The release grows with the grid (one block 489 ns, "
                        "264 blocks 652 on the SXM), and real kernels run many")
    p.add_argument("--replays", type=int, default=50, help="graph replays per window (traced), or per timing run")
    p.add_argument("--passes", type=int, default=1,
                   help="how many times every window is timed: the untraced run three (ruler.run), so the truth's "
                        "spread between runs is known (pdl_calibration: pdl_truth_spread_ns); the traced run once, "
                        "its exports paired with the windows by order")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    target = a.out / f"pdl-{a.mode}.json"

    def skip(reason):
        target.write_text(json.dumps({"mode": a.mode, "skipped": reason}) + "\n")
        print(f"pdl: skipped: {reason}")

    import torch

    if not torch.cuda.is_available():
        return skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability(0)
    if major < 9:
        return skip(f"sm_{major}{minor}: programmatic dependent launch needs sm_90 or later")
    node, why = build_kernel()
    if node is None:
        return skip(why)
    import triton

    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    dur = torch.zeros(a.nodes, dtype=torch.int64, device="cuda")
    work_ns = int(a.work_us * NS)
    try:  # a trial launch of each variant: a Triton that takes the calls may still refuse the keyword
        for pdl in (False, True):
            chain(node, counter, dur, 2, work_ns, 0, pdl)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001  whatever the compiler or driver raises is the reason
        return skip(f"trial launch failed: {type(e).__name__}: {e}"[:300])

    step, step_why = timer_step_ns(torch, counter)  # in this run's own conditions: traced or not
    grids = wide_grids(a.wide, torch.cuda.get_device_properties(0).multi_processor_count)
    passes = []
    for _ in range(max(1, a.passes)):  # every window once per pass, each pass its own captures and timings
        out = []
        for variant, tail_ns, grid in windows_spec([int(t) for t in a.tails.split(",")], grids):
            pdl = variant != "nopdl"
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                chain(node, counter, dur, a.nodes, work_ns, tail_ns, pdl, grid)  # compiled and warm before the capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                chain(node, counter, dur, a.nodes, work_ns, tail_ns, pdl, grid)
            counter.zero_()
            graph.replay()
            torch.cuda.synchronize()
            chain_ok = int(counter.item()) == a.nodes  # a release that let a node run early reads a stale counter
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if a.mode == "node":
                torch.cuda.cudart().cudaProfilerStart()
            u0 = time.time()
            e0.record()
            for _ in range(a.replays):
                graph.replay()
            e1.record()
            e1.synchronize()
            u1 = time.time()
            if a.mode == "node":
                torch.cuda.cudart().cudaProfilerStop()
            out.append({"variant": variant, "tail_ns": tail_ns, "grid": grid, "chain_ok": chain_ok,
                        "node_device_ns": e0.elapsed_time(e1) * 1e6 / (a.replays * a.nodes),
                        "work_ns": statistics.median(dur.tolist()), "unix": [u0, u1]})
        passes.append(out)
    rec = {"mode": a.mode, "device": torch.cuda.get_device_name(0), "capability": [major, minor],
           "triton": triton.__version__, "torch": torch.__version__, "nodes": a.nodes, "work_us": a.work_us,
           "replays": a.replays, "passes": len(passes), "timer_step_ns": step,
           **({"timer_step_why": step_why} if step_why else {}),
           "windows": passes[0], **({"repeats": passes[1:]} if len(passes) > 1 else {})}  # the later passes' windows
    target.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def pdl_calibration(folder, floor, inflation):
    """The PDL chain's ground truth against the overlap model, priced with the ruler's own calibration (floor: the empty
    graph node, us; inflation: node tracing's lengthening of a kernel, ns, None when the ruler measured none:
    ruler.validate.calibration, the same figures the cells are read with), {} when the step did not run, {"pdl":
    "skipped: <why>"} when the card or Triton cannot. Otherwise, per window and as scalars for the reference window (the
    PDL window whose traced overlap is the median of them):
    - pdl_exposed_ns_per_node: untraced (pdl-none.json), the PDL chain's device time per node less the node's own
      measured work: what a PDL launch leaves exposed beyond its work, the truth the model prices;
    - nopdl_exposed_ns_per_node: the same for the control chain without PDL: what a node adds without PDL;
    - pdl_overlap_ns_traced: median overlap (servebench.nsys.parse.overlaps) of the kernels in that window's
      node-mode export, raw, node tracing's lengthening of each kernel included; pdl_overlapped_share: the share of its
      kernels that overlap their predecessor;
    - pdl_model_ns: what exposed_launch_ms charges a node, max(0, floor - traced overlap), with floor the empty node
      (graph_node_floor_us, in ns: pdl_floor_ns); pdl_model_corrected_ns: the same with node_trace_inflation_ns taken
      off the overlap first (pdl_inflation_ns; absent when the ruler has no busy windows to measure it from);
    - pdl_model_error_ns, pdl_model_corrected_error_ns: model - truth. Positive: the model over-charges launches;
    - pdl_release_ns (pdl_release_tail_ns: its window): what a PDL launch leaves exposed however early it started, the
      smallest exposed time over the PDL windows whose dependence held (535 ns on the H100 PCIe, where the bare model
      charged 0 past a 1.1 us overlap), read only when the two longest tails agree within the timer's step and the
      truth's spread (else pdl_release_note says why there is none); pdl_model_floored_ns and
      pdl_model_floored_corrected_ns, with their _error_ns: the models charging at least that
      (anatomy.decompose.exposed_launch_ms release_ns, the function itself), what the cells are priced with;
    - pdl_wide_windows: one entry per grid in `--wide` (8, 64, twice the SMs and 1024 blocks a node by default) at
      WIDE_TAIL_NS, each with grid, pdl_wide_exposed_ns_per_node, pdl_wide_overlap_ns_traced and pdl_wide_chain_ok:
      how the release grows with the blocks a node runs (489 ns for one block, 652 for 264 on the H100 SXM), towards
      a real kernel's; the largest grid's figures also as pdl_wide_grid, pdl_wide_exposed_ns_per_node,
      pdl_wide_overlap_ns_traced and pdl_wide_chain_ok. Reported beside the release, never as it;
    - pdl_tail_ns: the reference window's tail (its knob: how long each node runs after releasing its dependent);
    - pdl_chain_ok: every window's counter chain came out right (the dependence held under PDL);
    - pdl_windows: the figures above for every PDL window, in tail order; a window whose tail spans fewer than
      TICKS_PER_TAIL steps of the coarser run's %globaltimer (pdl_timer_step_ns; None where it could not be read) is
      left out, and with none left the calibration is refused;
    - pdl_truth_passes, pdl_truth_spread_ns: how many times the untraced run timed every window (pdl.py --passes; its
      `repeats` hold the later passes), and the widest range of a window's exposed time between them (each window's
      own in pdl_exposed_spread_ns; absent with one pass). The truth is the passes' median. With the timer's step, the
      spread is what a choice between the raw and the corrected model must clear in every window
      (anatomy.decompose.suggested_overlap_model).
    The traced windows are matched to the untraced ones by (variant, tail_ns), the exports to the node run's windows by
    order (nsys numbers them from 1)."""
    from servebench.nsys.parse import Trace, numbered_exports, overlaps

    folder = Path(folder)
    traced, untraced = read(folder / "pdl-node.json"), read(folder / "pdl-none.json")
    if not (traced or untraced):
        return {}
    for rec in (untraced, traced):
        if rec and rec.get("skipped"):
            return {"pdl": f"skipped: {rec['skipped']}"}
    if not (traced and untraced):
        return {"pdl": f"incomplete: {'pdl-node.json' if not traced else 'pdl-none.json'} is missing"}
    if not floor:
        return {"pdl": "incomplete: no graph_node_us (launch_latency.json) to price the model with"}
    floor_ns = floor * NS
    key_of = lambda w: (w["variant"], w["tail_ns"], w.get("grid", 1))  # noqa: E731
    truth = {key_of(w): w for w in untraced["windows"]}
    repeats = [{key_of(w): w for w in rep} for rep in untraced.get("repeats") or []]
    exports = numbered_exports(folder, "pdl-node")
    if len(exports) != len(traced["windows"]):  # paired by order: one missing or extra moves every tail onto another
        return {"pdl": f"incomplete: {len(exports)} node exports for {len(traced['windows'])} traced windows"}
    steps = [rec.get("timer_step_ns") for rec in (traced, untraced)]
    tick = max(steps) if None not in steps else None  # the coarser of the two runs' timers
    fine = lambda tail: tick is None or tail >= TICKS_PER_TAIL * tick  # noqa: E731
    from anatomy.decompose import exposed_launch_ms  # here, host side: the GPU half runs without anatomy on its path

    def charge(overlap, cut=0.0, release=0.0):
        """What the cells' launch term charges one kernel with this overlap, ns: the function itself, so the chain
        tests the model the cells use and not a copy of it (a copy clamped the inflated overlap differently)."""
        return exposed_launch_ms(1, floor, [overlap], 1, cut, release) * 1e6
    control, rows, wide = {}, [], []
    for i, w in enumerate(traced["windows"]):
        key = key_of(w)
        u = truth.get(key)
        if u is None:
            continue
        timings = [u, *(r[key] for r in repeats if key in r)]  # the untraced passes that timed this window
        exposures = [t["node_device_ns"] - t["work_ns"] for t in timings]
        exposed = statistics.median(exposures)
        chain_ok = bool(w.get("chain_ok", True)) and all(bool(t.get("chain_ok", True)) for t in timings)
        if w["variant"] == "nopdl":
            control = {"nopdl_exposed_ns_per_node": exposed}
            continue
        if not fine(w["tail_ns"]):  # the timer cannot resolve this tail: it and the work it times are steps apart
            continue
        with Trace(exports[i]) as t:
            kernels = t.kernels()
        under = overlaps(kernels)
        if not under:
            continue
        overlap = statistics.median(under)
        if w["variant"] == "pdl-wide":  # nodes of many blocks, as a real kernel is: how the release grows with them
            # (reported beside the one-block release, never as it)
            wide.append({"grid": w.get("grid"), "pdl_wide_exposed_ns_per_node": exposed,
                         "pdl_wide_overlap_ns_traced": overlap, "pdl_wide_chain_ok": chain_ok})
            continue
        row = {"tail_ns": w["tail_ns"], "pdl_exposed_ns_per_node": exposed,
               "pdl_exposed_spread_ns": max(exposures) - min(exposures), "pdl_overlap_ns_traced": overlap,
               "pdl_overlapped_share": len(under) / len(kernels), "pdl_model_ns": charge(overlap), "chain_ok": chain_ok}
        row["pdl_model_error_ns"] = row["pdl_model_ns"] - exposed
        if inflation is not None:
            row["pdl_model_corrected_ns"] = charge(overlap, inflation)
            row["pdl_model_corrected_error_ns"] = row["pdl_model_corrected_ns"] - exposed
        rows.append(row)
    tails = [w["tail_ns"] for w in traced["windows"] if w["variant"] == "pdl"]
    if not rows and tick is not None and not any(map(fine, tails)):
        return {"pdl": f"refused: %globaltimer steps {tick} ns, too coarse for every tail (a tail needs "
                       f"{TICKS_PER_TAIL} steps)", "pdl_timer_step_ns": tick}
    if not rows:
        return {"pdl": "incomplete: no PDL window with a trace and a timing"}
    rows.sort(key=lambda r: r["pdl_overlap_ns_traced"])
    spread = max(r["pdl_exposed_spread_ns"] for r in rows) if repeats else None
    # what a PDL launch leaves exposed however early it started: the plateau the exposed time falls to as the overlap
    # grows (535 ns on the H100 PCIe at tails of 1, 2 and 4 us alike, 674 at 0.5 us), the dependency release the bare
    # model, max(0, floor - overlap), charges nothing for. It is read only from windows whose dependence held, and only
    # when the two longest tails agree within what the chain resolves (the timer's step, the truth's spread between
    # passes): a plateau, never a point. The floored model charges a PDL kernel at least that
    # (anatomy.decompose.exposed_launch_ms release_ns); on the plateau it fits by construction, so its test is the
    # windows whose overlap is shorter than the node cost less the release
    held, release, note = [r for r in rows if r["chain_ok"]], None, None
    if len(held) < 2:
        note = "one PDL window whose dependence held: no plateau to read" if held else \
            "no PDL window whose dependence held"
    else:
        a, b = (r["pdl_exposed_ns_per_node"] for r in sorted(held, key=lambda r: r["tail_ns"])[-2:])
        if abs(a - b) <= max(tick or 0.0, spread or 0.0):
            release = min(held, key=lambda r: r["pdl_exposed_ns_per_node"])
        else:
            note = f"the two longest tails disagree: {a:.0f} against {b:.0f} ns exposed"
    if release:
        for r in rows:
            r["pdl_model_floored_ns"] = charge(r["pdl_overlap_ns_traced"], 0.0, release["pdl_exposed_ns_per_node"])
            r["pdl_model_floored_error_ns"] = r["pdl_model_floored_ns"] - r["pdl_exposed_ns_per_node"]
            if "pdl_model_corrected_ns" in r:
                r["pdl_model_floored_corrected_ns"] = charge(r["pdl_overlap_ns_traced"], inflation,
                                                            release["pdl_exposed_ns_per_node"])
                r["pdl_model_floored_corrected_error_ns"] = (r["pdl_model_floored_corrected_ns"]
                                                            - r["pdl_exposed_ns_per_node"])
    ref = rows[len(rows) // 2]
    out = {**control, **{k: v for k, v in ref.items() if k not in ("tail_ns", "chain_ok")},
           "pdl_tail_ns": ref["tail_ns"], "pdl_floor_ns": floor_ns,
           **({"pdl_release_ns": release["pdl_exposed_ns_per_node"], "pdl_release_tail_ns": release["tail_ns"]}
              if release else {"pdl_release_note": note}), **wide_summary(wide),
           "pdl_chain_ok": all(r["chain_ok"] for r in rows), "pdl_windows": rows, "pdl_timer_step_ns": tick,
           "pdl_truth_passes": 1 + len(repeats), **({"pdl_truth_spread_ns": spread} if repeats else {})}
    if inflation is not None:
        out["pdl_inflation_ns"] = inflation
    return out


if __name__ == "__main__":
    main()
