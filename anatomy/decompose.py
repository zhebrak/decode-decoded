"""The anatomy of one decode step: where the time between the speed-of-light floor and the measured step goes, in ms
per step.

    floor      = bytes per step / achievable bandwidth           what the step would take at the memory limit
    copy       = copies and sets outside any kernel              the GPU moving memory, not computing (node trace)
    launch     = kernels per step x graph-node floor + gaps      what launching the step's kernels costs the GPU
    kernel_exc = kernel time - floor - launch floor              kernels slower than their byte floor
    host_stall = (1 - occupied) x step                           the GPU idle with nothing queued: waiting on the host

occupied is the graph-mode trace's GPU busy time per step over the unprofiled step (anatomy.tidy), so host_stall is
step - busy: tracing costs the host a few us per traced eager kernel, which stretches the profiled step wherever the
host sets the pace, but leaves the GPU's work unchanged (validation, A100: SGLang B1 traced +4.0% longer, busy time per
step within 1% of the unprofiled step; GPU-bound cells' profiled steps within 0.3%).

Kernel time comes from a node-mode trace, corrected for the trace's own inflation (node_trace_inflation per kernel,
measured by the ruler per card: 340 ns on the A100, 120 ns on the H100 NVL). Real traces showed that CUPTI puts a
kernel's launch cost inside its duration, not between kernels (64 ns gaps between graph nodes), so the gaps alone
("bubbles", graph-mode span minus kernel time) read near zero and launch cost would hide inside "kernel excess". The
launch term therefore counts the kernels and charges each what a graph node costs on the card (graph_node_floor: an
empty node, 1.64 us on the A100 and 1.09 on the NVL), less what of it a programmatic dependent launch hides under the
kernel before it, never less than what such a launch leaves exposed however early it started (the ruler's PDL chain:
535 ns a node on the H100 PCIe, about half the node cost; exposed_launch_ms), plus whatever gaps the traces show
(bubbles_ms, reported apart: real engine
graphs on Hopper show 0.1-0.5 us a node the ruler's do not, unexplained). anatomy.tidy reports the node-cost part's
share over the range down to the ruler's untraced spin-graph overhead per node, with and without node tracing's
inflation taken off each overlap (launch_floor_share_range: a lower bound on launch cost). Without the ruler's
calibration the launch term is just the gaps, and the result says so (launch_model False).

Copies and memsets that no kernel covers are their own term, taken out of the gaps: busy time that is neither a kernel
nor a launch (llama.cpp's logits copy to the host at 256 slots is 1.5% of its step), which the gaps would otherwise
have booked as launch cost.

The five sum to the step by construction. A term below zero beyond TOLERANCE of the step means the traces and the
timing disagree, and the cell is flagged, never clipped. floor and launch are modelled (from bytes, and from kernels
and the ruler's node cost); copy and host stall are measured; kernel excess is what the kernels' measured time leaves.
"""
import math

TOLERANCE = 0.03
TERMS = ("floor", "copy", "kernel_exc", "launch", "host_stall")


def exposed_launch_ms(kernels, node_floor_us, overlaps_ns=(), steps=1, inflation_ns=0.0, release_ns=0.0):
    """What a step's kernels cost to launch on the GPU, ms: each of the step's `kernels` charged one graph node's cost
    on the card, less what of it runs under its same-stream predecessor (a programmatic dependent launch:
    servebench.nsys.parse.overlaps), and never less than release_ns: what such a launch leaves exposed however early
    it started, the ruler's PDL chain's pdl_release_ns (535 ns a node on the H100 PCIe, about half the node cost;
    without it, as before the chain ran on hardware, an overlap past the node cost was charged nothing, 440-535 ns a
    node short of the truth in every window). overlaps_ns is every overlapped kernel's overlap in a window of `steps`
    steps; each is first shortened by node tracing's lengthening of the predecessor (inflation_ns), then clamped kernel
    by kernel: clamping a mean is not the mean of the clamps (overlaps of 0.1 and 1.9 us under a 1 us node leave 0.9 us
    exposed, not the 0 their 1.0 us mean would; llama.cpp's batch 1, 394 of 397 kernels overlapping unevenly, read 2.6
    against 63.5 us a step in validation)."""
    f, least = node_floor_us * 1e3, max(0.0, release_ns or 0.0)  # a release below zero is no release
    under = [max(0.0, o - (inflation_ns or 0.0)) for o in overlaps_ns or ()]
    hidden_per_step = min(len(under) / steps, kernels)
    return ((kernels - hidden_per_step) * f + sum(max(least, f - o) for o in under) / steps) / 1e6


def decompose(step_ms, occupied, kernel_ms, floor_ms, kernels_per_step=None, node_floor_us=None, inflation_ns=None,
              overlaps_ns=None, steps=None, copy_ms=None, overlap_model="corrected", release_ns=None):
    """{floor, copy, kernel_exc, launch, host_stall (ms) and shares of the step, bubbles_ms (the measured gaps),
    launch_floor_ms (exposed_launch_ms), launch_release_ns (what it charged an overlapped kernel at least),
    kernel_ms_corrected, consistent, launch_model}; None when an input is missing."""
    if overlap_model not in ("raw", "corrected"):
        raise ValueError(f"unknown overlap model: {overlap_model}")
    if None in (step_ms, occupied, kernel_ms, floor_ms) or not step_ms:
        return None
    n = kernels_per_step or 0
    kern = kernel_ms - (n * inflation_ns / 1e6 if inflation_ns and n else 0.0)
    overlap_inflation = inflation_ns if overlap_model == "corrected" else 0.0
    launch_floor = exposed_launch_ms(n, node_floor_us, overlaps_ns, steps or 1, overlap_inflation, release_ns or 0.0) \
        if node_floor_us and n else 0.0
    copy = copy_ms or 0.0
    host_stall = (1 - occupied) * step_ms
    bubbles = occupied * step_ms - kern - copy
    parts = {"floor": floor_ms, "copy": copy, "kernel_exc": kern - floor_ms - launch_floor,
             "launch": launch_floor + bubbles, "host_stall": host_stall}
    out = {f"{k}_ms": v for k, v in parts.items()}
    out |= {f"{k}_share": v / step_ms for k, v in parts.items()}
    out |= {"bubbles_ms": bubbles, "launch_floor_ms": launch_floor, "launch_release_ns": release_ns,
            "kernel_ms_corrected": kern, "launch_model": bool(launch_floor), "overlap_model": overlap_model,
            "consistent": all(v >= -TOLERANCE * step_ms for v in parts.values())}
    return out


def suggested_overlap_model(cal):
    """(model, evidence) from S1's PDL chain (ruler.pdl), the one statement of the rule: of the windows the two floored
    models charge differently (an overlap shorter than the node cost less the release; on the plateau both charge the
    release, and such windows say nothing), "corrected" when every one's absolute error sits under the raw one's by
    more than the chain resolves, "raw" when every one says the opposite as clearly, None when they disagree, a gap is
    within the resolution (the larger of the %globaltimer's step and the truth's spread between the untraced passes),
    or no window tells them apart. One window's point error decided this once, and the choice moves llama.cpp's launch
    term at batch 1 by 40% (3.4% against 4.7% of the step in validation); undecided, the checked-in value stands as
    a default and every cell's launch_floor_share_range carries both models. evidence: the windows, how many tell the
    models apart (informative), how many corrected wins, the smallest gap, the resolution and the verdict, for
    tolerances-suggested.yaml."""
    finite = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731

    def errors(w):  # the model the cells are priced with: floored by the PDL release where the chain measured one
        floored = "pdl_model_floored_error_ns" in w
        return (w.get("pdl_model_floored_error_ns" if floored else "pdl_model_error_ns"),
                w.get("pdl_model_floored_corrected_error_ns" if floored else "pdl_model_corrected_error_ns"))
    windows = [(raw, corr) for w in cal.get("pdl_windows") or [] for raw, corr in (errors(w),)
               if finite(raw) and finite(corr)]
    informative = [(raw, corr) for raw, corr in windows if raw != corr]  # charged alike by both: nothing to choose
    tick, spread = cal.get("pdl_timer_step_ns"), cal.get("pdl_truth_spread_ns")
    evidence = {"windows": len(windows), "informative": len(informative), "timer_step_ns": tick,
                "truth_spread_ns": spread, "truth_passes": cal.get("pdl_truth_passes")}
    if not windows:
        return None, {**evidence, "verdict": "undecided: no PDL window with both errors"}
    if not informative:
        return None, {**evidence, "verdict": "undecided: the two models charge every window alike"}
    gaps = [abs(raw) - abs(corr) for raw, corr in informative]  # > 0: corrected closer to the truth
    evidence |= {"corrected_wins": sum(g > 0 for g in gaps), "min_gap_ns": round(min(map(abs, gaps)), 1)}
    if tick is None or spread is None:
        return None, {**evidence, "verdict": "undecided: " + ("the %globaltimer step was not read" if tick is None else
                                                              "one untraced pass, the truth's spread unknown")}
    resolution = max(tick, spread)
    evidence["resolution_ns"] = resolution
    if all(g > resolution for g in gaps):
        return "corrected", {**evidence, "verdict": f"corrected: every window by more than {resolution:g} ns"}
    if all(g < -resolution for g in gaps):
        return "raw", {**evidence, "verdict": f"raw: every window by more than {resolution:g} ns"}
    return None, {**evidence, "verdict": "undecided: the windows disagree, or a gap is within the resolution"}
