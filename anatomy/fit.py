"""The model behind the calculator: per engine and arm, how long the host takes per decode step, and how
that combines with the GPU's time, so the crossover can be predicted on another card. Pure functions over cell rows.

Per cell the measurements give G = occupied x step, the time the GPU had work queued (kernels and their launch cost),
and the host stall S = step - G. Two ways an engine can combine host and GPU work:
- overlapped (async scheduling, overlap scheduler): the host prepares step n+1 while the GPU runs step n, so
  step = max(H, G) and S = max(0, H - G). A cell with S > 0 shows H = G + S; a cell with S ~ 0 only says H <= G.
- serial: the host works, then the GPU, so step = H + G and S = H in every cell.
H(B) = h0 + h1 B per (engine, arm, level): a fixed cost per step and a cost per sequence (input preparation,
sampling, output processing all scale with the batch). Fitted by least squares on the cells that carry H: the
host-bound cells for the overlapped form, all cells for the serial one. The form whose predicted step times fit the
measured ones better is kept, by root-mean-square relative error: the two forms agree on GPU-bound cells and differ on
the host-bound ones, so a median over all cells would be decided by the cells that cannot tell them apart.
A host model needs MIN_HOST_CELLS host-bound cells at two batch widths or more; with fewer the host cost is reported as
not identified, never as a line through its own noise. h0 and h1 carry a bootstrap interval (INTERVAL) that keeps
each model's cells together: one process measures every batch of a model, so its cells share that process's offset
(intervals: whole models where enough carry the fit, else cells with a process offset from the measured noise). The
line is fitted unconstrained: a per-sequence cost below zero whose interval reaches 0 is fixed at 0 and said so
(h1_fixed_at_zero), one whose interval lies below 0, or a fixed cost below 0, makes the fit unphysical (a reason, and
no scenario drawn from it), never a clamp that quietly moves the slope into h0.

Validation: leave one model size out, choose the form and fit on the rest, predict the held-out model's step times
from its measured G (the choice made inside each fold, so the error is not conditioned on a form picked with the
held-out model). Residuals are reported apart for host-bound and GPU-bound cells.
Prediction for another card: same host (the stated limit: a different CPU is untested); of the GPU time, the
memory-bound part (the decomposition's floor + kernel excess) scales by the bandwidth ratio when the cell is not
compute-leaning, and the launch cost does not (it is per kernel, not per byte).
"""
import math
import random
from statistics import median

STALL_MIN = 0.05  # a cell is host-bound when its host stall is at least this share of the step
MIN_HOST_CELLS = 4  # fewer host-bound cells than this, or one batch width only: the host cost is not identified
INTERVAL = (0.05, 0.95)  # the bootstrap interval on h0 and h1
RESAMPLES = 400
MIN_BOOT_MODELS = 5  # models carrying the fit's points that resampling whole models needs: with fewer it has too few
                     # distinct draws (one host-bound model gives one, and an interval of zero width)


def _lstsq(points):
    """(h0, h1) minimising sum (h0 + h1 b - y)^2 over (b, y); h1 alone through the mean when b does not vary."""
    n = len(points)
    if n == 0:
        return None
    mb = sum(b for b, _ in points) / n
    my = sum(y for _, y in points) / n
    var = sum((b - mb) ** 2 for b, _ in points)
    h1 = sum((b - mb) * (y - my) for b, y in points) / var if var else 0.0
    return my - h1 * mb, h1


def gpu_ms(r):
    return r["occupied"] * r["step_ms"]


def usable(rows):
    """Accepted cells with an occupancy: a flagged cell (stalls, disagreeing estimators, profiler overhead, broken
    anchors) never shapes the host model."""
    from anatomy.metrics import accepted

    return [r for r in rows if r.get("step_ms") and r.get("occupied") is not None and accepted(r)]


def points(rows, form):
    """(batch, host time) for every cell that carries it: every cell's stall for the serial form, the host-bound
    cells' step for the overlapped one."""
    rows = usable(rows)
    if form == "serial":
        return [(r["batch"], r["step_ms"] - gpu_ms(r)) for r in rows]
    return [(r["batch"], r["step_ms"]) for r in rows if host_bound(r)]


def fit_form(rows, form):
    pts = points(rows, form)
    return _lstsq(pts), len(pts)


def predict_step(h, g, batch, form):
    host = h[0] + h[1] * batch
    return host + g if form == "serial" else max(host, g)


def rel_errors(rows, h, form):
    return [abs(predict_step(h, gpu_ms(r), r["batch"], form) / r["step_ms"] - 1) for r in usable(rows)]


def fit(rows, offset_sd=None):
    """{form, h0_ms, h1_ms, h0_interval, h1_interval, n_fitted (the cells the line went through: every cell for the
    serial form, the host-bound ones for the overlapped), n_host_cells (stall at least STALL_MIN), median_error,
    rms_error, alternative, lomo} for one engine's cells (one arm, level and prompt), or None when there is nothing to
    fit. When too few cells waited on the
    host (MIN_HOST_CELLS, two batch widths), the host cost is not identified, only bounded (H <= G in the cells that did
    not wait): the result says so (form "not host-bound" or "too few host-bound cells", h0/h1 None,
    host_upper_bound_ms per batch), rather than reading 2% slack as a serial host cost."""
    use = usable(rows)
    waited = [r for r in use if host_bound(r)]
    if use and (len(waited) < MIN_HOST_CELLS or len({r["batch"] for r in waited}) < 2):
        bound = {}
        for r in use:
            if not host_bound(r):
                bound[r["batch"]] = min(bound.get(r["batch"], float("inf")), gpu_ms(r))
        return {"form": "too few host-bound cells" if waited else "not host-bound", "h0_ms": None, "h1_ms": None,
                "h0_interval": None, "h1_interval": None, "n_host_cells": len(waited), "median_error": None,
                "rms_error": None, "alternative": None, "host_upper_bound_ms": dict(sorted(bound.items())),
                "lomo": {"per_model": {}, "median": None, "cells": 0}}
    out = forms(rows)
    if not out:
        return None
    best = min(out.values(), key=lambda f: f["rms_error"] if f["rms_error"] is not None else 1e9)
    other = next((f for f in out.values() if f is not best), None)
    best = {**best, "n_host_cells": len(waited), **intervals(rows, best["form"], offset_sd)}
    return {**best, **physical(rows, best), "alternative": other, "lomo": leave_one_model_out(rows),
            "residuals": residuals(rows, (best["h0_ms"], best["h1_ms"]), best["form"])}


def forms(rows):
    """{form: its fit} for both forms (overlapped, serial), each with its errors over every usable cell."""
    out = {}
    for form in ("overlapped", "serial"):
        h, n = fit_form(rows, form)
        if h is None:
            continue
        errs = rel_errors(rows, h, form)
        out[form] = {"form": form, "h0_ms": h[0], "h1_ms": h[1], "n_fitted": n,
                     "median_error": median(errs) if errs else None,
                     "rms_error": math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else None}
    return out


def physical(rows, f):
    """{h0_ms, h1_ms, h1_fixed_at_zero, unphysical}: a per-sequence cost below 0 whose interval reaches 0 is not
    resolved, so the line is refitted through the mean with h1 = 0 (and says so); one whose whole interval is below 0,
    or a fixed cost below 0, is no host model: unphysical names why, and the scenario leaves the fit out."""
    h0, h1 = f["h0_ms"], f["h1_ms"]
    lo, hi = f.get("h1_interval") or (h1, 0.0)  # no interval: the sign is not resolved, as one that reaches 0
    out = {"h0_ms": h0, "h1_ms": h1, "h1_fixed_at_zero": False, "unphysical": None}
    if h1 < 0 and hi < 0:
        return out | {"unphysical": f"the host's time falls as the batch grows (per sequence {h1 * 1e3:.1f} us, "
                                    f"interval {lo * 1e3:.1f} to {hi * 1e3:.1f})"}
    if h1 < 0:
        pts = points(rows, f["form"])
        out |= {"h0_ms": sum(y for _, y in pts) / len(pts), "h1_ms": 0.0, "h1_fixed_at_zero": True}
    if out["h0_ms"] < 0:
        out["unphysical"] = f"a fixed host cost below zero ({out['h0_ms']:.3f} ms)"
    return out


def residuals(rows, h, form):
    """Median |relative error| of the fitted step on the host-bound and on the GPU-bound cells, apart: a good fit to
    the second says little about the host."""
    use = usable(rows)
    out = {}
    for regime, cells in (("host-bound", [r for r in use if host_bound(r)]),
                          ("gpu-bound", [r for r in use if not host_bound(r)])):
        errs = rel_errors(cells, h, form)
        out[regime] = {"median_error": median(errs) if errs else None, "cells": len(errs)}
    return out


def intervals(rows, form, offset_sd=None):
    """{h0_interval, h1_interval, interval_method}: INTERVAL quantiles of h0 and h1 refitted RESAMPLES times, a fixed
    seed so the page rebuilds identically. One process measured every batch of a model, so its cells share that
    process's offset and are not independent:
    - with MIN_BOOT_MODELS models carrying the fit's points or more, whole models are resampled ("models");
    - with fewer, each model's cells are resampled and its process given an offset drawn from the between-process
      noise (offset_sd, relative: the noise block's), which moves its steps and so its host time, the GPU's busy time
      held ("cells and process offsets");
    - with neither, the interval is not identified (None), and physical() then never judges h1's sign from it."""
    rows, rng, fits = usable(rows), random.Random(0), []
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    carrying = {m for m, rs in by_model.items() if any(form == "serial" or host_bound(r) for r in rs)}
    if len(carrying) >= MIN_BOOT_MODELS:
        groups, method = list(by_model.values()), "models"
        draw = lambda: [r for _ in groups for r in rng.choice(groups)]  # noqa: E731
    elif offset_sd:
        method = "cells and process offsets"

        def draw():
            out = []
            for rs in by_model.values():
                k = 1 + rng.gauss(0, offset_sd)  # the process's offset: its step moves, the GPU's busy time does not
                out += [{**r, "step_ms": r["step_ms"] * k, "occupied": r["occupied"] / k}
                        for r in (rng.choice(rs) for _ in rs)]
            return out
    else:
        return {"h0_interval": None, "h1_interval": None, "interval_method": None}
    for _ in range(RESAMPLES):
        h, n = fit_form(draw(), form)
        if h is not None and n:
            fits.append(h)
    if not fits:
        return {"h0_interval": None, "h1_interval": None, "interval_method": None}
    q = lambda xs: [sorted(xs)[int(p * (len(xs) - 1))] for p in INTERVAL]  # noqa: E731
    return {"h0_interval": q([h[0] for h in fits]), "h1_interval": q([h[1] for h in fits]), "interval_method": method}


def host_bound(r):
    return (r["step_ms"] - gpu_ms(r)) >= STALL_MIN * r["step_ms"]


def leave_one_model_out(rows):
    """Median |relative error| of predicted step time on each held-out model's host-bound cells, with the form chosen
    and H fitted on the other models, and how many cells that rests on. Host-bound cells only: on a GPU-bound one the
    overlapped form predicts max(H, G) with G measured, so its error is at most its small stall and says nothing about
    H."""
    models = sorted({r["model"] for r in usable(rows)})
    errs, n_cells = {}, 0
    for m in models:
        rest = forms([r for r in rows if r["model"] != m])
        if not rest:
            continue
        f = min(rest.values(), key=lambda x: x["rms_error"] if x["rms_error"] is not None else 1e9)
        if not f["n_fitted"]:
            continue
        e = rel_errors([r for r in usable(rows) if r["model"] == m and host_bound(r)], (f["h0_ms"], f["h1_ms"]),
                       f["form"])
        if e:
            errs[m], n_cells = median(e), n_cells + len(e)
    return {"per_model": errs, "median": median(errs.values()) if errs else None, "cells": n_cells}


def predict(rows, f, bw_ratio):
    """The scenario on a card with bw_ratio = its bandwidth / this card's, the same host: each measured cell's step
    and GPU busy share, a band from the corners of h0's and h1's intervals (occupied_lo, occupied_hi), and whether its
    batch lies outside the batches the host line was fitted on (extrapolated). Every listed card has at least this
    card's memory, so every measured cell fits on it. [] for a fit with no host model, an unphysical one, or one whose
    interval is not identified (a scenario is never drawn without its band)."""
    if f.get("h0_ms") is None or f.get("unphysical") or not (f.get("h0_interval") and f.get("h1_interval")):
        return []
    fitted = [b for b, _ in points(rows, f["form"])]
    corners = [(f["h0_ms"], f["h1_ms"])] + [(a, b) for a in f["h0_interval"] for b in f["h1_interval"]]
    out = []
    for r in usable(rows):
        floor, exc = r.get("floor_ms"), r.get("kernel_exc_ms")
        if floor is None or exc is None:  # no kernel time: nothing to scale, so no prediction (never the H100 echoed)
            continue
        # the bytes scale with bandwidth; launches and copies (to the host, over the link) stay as measured
        g = gpu_ms(r) if r.get("compute_leaning") else \
            (floor + exc) / bw_ratio + max(r.get("launch_ms") or 0.0, 0.0) + (r.get("copy_ms") or 0.0)
        steps = [predict_step(h, g, r["batch"], f["form"]) for h in corners]
        out.append({"engine": r["engine"], "model": r["model"], "batch": r["batch"], "step_ms": steps[0],
                    "occupied": g / steps[0], "occupied_lo": g / max(steps), "occupied_hi": g / min(steps),
                    "extrapolated": bool(fitted) and not min(fitted) <= r["batch"] <= max(fitted)})
    return out
