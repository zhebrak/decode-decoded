"""The page's pictures of GPU activity, cut from node-mode traces (every kernel inside the CUDA graphs has a row):
the opening picture, a few decode steps of one cell per engine (strips, strips.json), and two steps of every cell the
cell explorer shows (cell_strips, cell-strips.json, compact, fetched only when a reader opens a cell). Written by
anatomy.tidy beside the cell table, so the page never needs the raw traces.

Node tracing is not free: it lengthens each kernel a little (the ruler's node_trace_inflation) and so each step. The
strip is the trace as captured, and records its own step beside the unprofiled one, so the page can say by how much
the picture is stretched instead of hiding it.
"""
from servebench.nsys.parse import KERNEL_CLASSES, Trace, classify, step_starts
from servebench.results import resolve, row_cell_id

MODEL, BATCH, STEPS, SKIP = "Qwen/Qwen3-0.6B", 1, 3, 8   # SKIP: steps into the window, clear of its start
CELL_STEPS = 2  # a cell's strip in the explorer: two steps show a step and the gap into the next
TYPICAL = 0.15  # a typical step is within this of the capture's median step length (typical_window)
CLASSES = [*(c for c, _ in KERNEL_CLASSES), "other"]  # cell-strips.json names a kernel's class by its index here


def typical_window(starts, steps=STEPS, margin=SKIP, within=TYPICAL):
    """The first step of the `steps` consecutive whole steps that read most like the capture's median step: every one
    within `within` of the median length, the window's total closest to steps x median, at least `margin` steps from
    either end of the capture; None when no window qualifies. The page's opening strip drew the trace as captured,
    SKIP steps in, and on the study's card two of its nine steps were hiccups of the traced run (vLLM's first 4.1 ms
    where its others were 2.3, TensorRT-LLM's third 3.6 where its others were 2.6): a picture of pauses, read as a
    picture of speed."""
    lengths = [b - a for a, b in zip(starts, starts[1:])]
    if len(lengths) < steps:
        return None
    med = sorted(lengths)[len(lengths) // 2]
    best = None
    for i in range(margin, len(lengths) - steps - margin + 1):
        window = lengths[i:i + steps]
        if all(abs(x / med - 1) <= within for x in window):
            off = abs(sum(window) - steps * med)
            if best is None or off < best[0]:
                best = (off, i)
    return best[1] if best else None


def strip(path, anchor="graph", steps=STEPS, skip=SKIP, typical=False):
    """`steps` whole decode steps of one node-mode trace, from the anchor `skip` steps in (a negative skip counts back
    from the capture's last anchor, as tidy places a cell's window: skip=-back is the window back steps before the
    end), each kernel as a row with its class, in microseconds from the first step's start; [] when the trace holds
    too few whole steps. With `typical`, the steps are the typical_window's, where one qualifies, and skip's otherwise:
    the opening picture shows steps like the capture's median, not whatever pause the trace held at a fixed place."""
    with Trace(path) as t:
        starts = step_starts(t, anchor)
        if skip < 0:
            skip = len(starts) + skip
        if typical and (at := typical_window(starts, steps)) is not None:
            skip = at
        if skip < 0 or len(starts) < skip + steps + 1:
            return []
        t0 = starts[skip]
        rows = []
        for s in range(steps):
            a, b = starts[skip + s], starts[skip + s + 1]
            rows.append({"kind": "step", "start": (a - t0) / 1e3, "step": s})
            rows += [{"kind": "kernel", "start": (k.start - t0) / 1e3, "end": (k.end - t0) / 1e3, "step": s,
                      "class": classify(k.name)} for k in t.kernels() if a <= k.start < b]
        rows.append({"kind": "step", "start": (starts[skip + steps] - t0) / 1e3, "step": steps})
        return rows


def cut_of(r):
    """(anchor, skip) for a row's strip: the anchor its traces were read with and the window tidy placed (counted back
    from the capture's last anchor, moved by trace_offset_steps), so the strip shows the step the cell reports; the
    graph anchor SKIP steps in for a row without them (older tables)."""
    anchor = r.get("anchor") or "graph"
    back = (r.get("window_from_end") or 0) - (r.get("trace_offset_steps") or 0)
    return anchor, (-int(back) if back and back > 0 else SKIP)


def strips(rows, roots):
    """{engine: {cell, step_ms, traced_step_ms, rows}} for each engine's accepted defaults cell at its headline level,
    MODEL at BATCH, from the node trace tidy recorded under the cell's sources (found under the session roots): three
    typical steps (typical_window), traced_step_ms their mean."""
    from anatomy.metrics import headline_rows

    out = {}
    for r in headline_rows(rows):
        path = resolve((r.get("sources") or {}).get("node"), roots)
        if r.get("model") != MODEL or r.get("batch") != BATCH or not path or not path.exists():
            continue
        anchor, skip = cut_of(r)
        cut = strip(path, anchor, skip=skip, typical=True)
        ticks = [x["start"] for x in cut if x["kind"] == "step"]
        if len(ticks) > 1:
            out[r["engine"]] = {"cell": row_cell_id(r), "step_ms": r.get("step_ms"),
                                "traced_step_ms": (ticks[-1] - ticks[0]) / 1e3 / STEPS, "rows": cut}
    return out


def cell_strips(rows, roots):
    """{classes: CLASSES, cells: {cell id: {step_ms, traced_step_ms, ticks, k}}} for every cell the explorer shows (a
    defaults cell at its engine's headline level, accepted or not) that has a node trace: CELL_STEPS whole steps, the
    step starts (ticks) and each kernel as [start, end, class index], in microseconds from the first step to 0.01. One
    cell's strip is a few kilobytes, so the whole study's fits one file the page fetches only when a reader opens a
    cell."""
    from anatomy.metrics import headline_rows

    out = {}
    for r in headline_rows(rows, only_accepted=False):
        path = resolve((r.get("sources") or {}).get("node"), roots)
        if not path or not path.exists():
            continue
        anchor, skip = cut_of(r)
        cut = strip(path, anchor, steps=CELL_STEPS, skip=skip)
        ticks = [round(x["start"], 2) for x in cut if x["kind"] == "step"]
        if len(ticks) > 1:
            kernels = [[round(x["start"], 2), round(x["end"], 2), CLASSES.index(x["class"])]
                       for x in cut if x["kind"] == "kernel"]
            out[row_cell_id(r)] = {"step_ms": r.get("step_ms"),
                                   "traced_step_ms": (ticks[-1] - ticks[0]) / 1e3 / CELL_STEPS, "ticks": ticks,
                                   "k": kernels}
    return {"classes": CLASSES, "cells": out}
