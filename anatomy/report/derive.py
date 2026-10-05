"""The page's derived tables, from the cell rows (anatomy.tidy, or anatomy.report.synthetic for the layout preview):
pure functions, tested, so every number on the page, in the principles and in patterns.json is one these computed.

Noise (Noise.sigma). A step time's run-to-run standard deviation between processes, relative to the step, for a cell
like the one compared: from the noise block's cells (noise.csv beside the cell table: each cell's processes at both
ends of the first session, so drift over a session counts) of the same engine, level and arm, at the nearest model
size, as a line in log variance against log2(batch) fitted over the batches measured, weighted by their degrees of
freedom and corrected for the log's bias (in validation SGLang's spread was 0.07% at batch 1 and 0.62% at batch 64, so
one figure per engine would be wrong at both, and one per batch rests on five degrees of freedom); held flat beyond the
batches measured. An arm without noise cells of its own takes the largest its engine shows at that level and batch.
Never below configs/tolerances.yaml's noise_floor (what six processes cannot sample: thermal state over hours,
warm-up); without noise.csv, step_noise. Repeats inside one process need not bound the spread between processes (vLLM
at batch 1 repeated within 0.13% in each process while separate processes sat 0.87% apart), so a cell's own repeat
range is not its uncertainty. Two cells from different sessions also carry each engine's
drift between those sessions, from the means of its valid sentinel readings in each (sentinels.csv), added once per
engine in quadrature; an engine with no sentinel in one of them is charged the sentinel_drift tolerance. And the noise
is checked on each session's own host (Noise.check): where an engine's sentinel cell there spreads more than the first
session's noise allows, that engine's noise in that session is scaled up to what was seen.

A comparison of two step times is `beyond` noise when |ratio - 1| > K x their combined noise (K: tolerances'
beyond_k). With an estimated noise and an engine picked as the fastest of several, more than the normal 0.27% of
no-difference comparisons cross that line, so chance() counts them by simulation, for the group's size and the noise
estimate's degrees of freedom, and the page states the total.

Tables:
- crossovers: per (engine, model), the batch from which the GPU stays at least 90% occupied (a dip below the line
  beyond the cell's own noise at a larger batch moves it up), interpolated in log2(batch) between the two measured
  batches that bracket it, for the chart, and stated as that bracket; never extrapolated (frontier: the ones with a
  batch);
- engine_gaps: like with like, at each model, batch and level (the engine API, L1, or the server, L2), every engine's
  step over the fastest engine's there;
- mitigations: one mitigation off against every mitigation on, same engine, release, model, batch and prompt;
- stack: the runner (L0) and the server (L2) against the engine's headline level for the same cell;
  both read within one session only: a pair from two sessions (exploratory) carries a drift term in place of the
  control, and is listed in patterns.json, never in a principle or a chart;
- parts: where one cell's step goes, as shares, with the launch term's range;
- headline_fits: the host-cost model per engine (anatomy.fit), the calculator's input;
- tables: all of them, built once.
The occupancy's noise and whether a host stall is resolved come from the same noise (occupancy_noise), not from a
global tolerance, so one rule decides every "within noise".
"""
import csv
import functools
import json
import math
import random
from pathlib import Path
from statistics import median

from anatomy import fit
from anatomy.metrics import BEYOND_K as K
from anatomy.metrics import HEADLINE, NOISE_FLOOR, SERIAL_ARMS, TOLERANCES, accepted, defaults, headline_rows, rejection
from anatomy.report.charts import arm_name
from servebench import ROOT
from servebench.physics import load_shape
from servebench.results import model_name, row_cell_id

FLOOR = NOISE_FLOOR
OCCUPIED = 0.9  # the frontier's line: below it, the host holds the step back



class Noise:
    """sigma(row) -> (relative SD, its degrees of freedom or None when assumed) and drift(engine, session, session),
    from noise.csv's and sentinels.csv's rows (the module docstring has the rules)."""

    def __init__(self, cells=(), sentinels=()):
        pooled, occ = {}, {}
        for r in cells:
            k = (r["engine"], r["level"], r["arm"], r["model"], int(r["batch"]))
            n = int(r.get("processes") or 2)
            if r.get("sd") not in (None, ""):
                pooled.setdefault(k, []).append((float(r["sd"]), n))
            if r.get("occupied_sd") not in (None, ""):  # the noise block's graph captures: occupancy's own spread
                occ.setdefault(k, []).append((float(r["occupied_sd"]), n))
        # sessions pooled: the root mean square of their SDs, their degrees of freedom summed
        rms = lambda v: (math.sqrt(sum(s * s for s, _ in v) / len(v)), sum(n - 1 for _, n in v))  # noqa: E731
        self.cells = {k: rms(v) for k, v in pooled.items()}
        self.processes = [({"engine": r["engine"], "level": r["level"], "arm": r["arm"], "model": r["model"],
                            "batch": int(r["batch"]), "session": r.get("session")},
                           [float(x) for x in str(r.get("process_steps") or "").split()]) for r in cells]
        self.occupancy = {k: rms(v) for k, v in occ.items()}
        self.readings = {}  # {(engine, level, arm, model, batch): {session: [valid locked-clock steps]}}
        self.ends = {}  # same key plus session: the first round's start/end
        for r in sentinels:
            # Older sidecars dropped the level: regenerate them from raw metadata rather than guessing L1 or L2.
            if r.get("level") and r.get("position") != "unlocked" and str(r.get("valid")) == "True" \
                    and r.get("step_ms"):
                k = (r["engine"], r["level"], r["arm"], r["model"], int(r["batch"]))
                self.readings.setdefault(k, {}).setdefault(r["session"], []).append(float(r["step_ms"]))
                if r.get("position") in ("start", "end") and not r.get("redo"):
                    self.ends.setdefault((*k, r["session"]), {})[r["position"]] = float(r["step_ms"])
        self.sentinels = {k: {s: sum(v) / len(v) for s, v in by.items()} for k, by in self.readings.items()}
        self.scale = {}  # {(engine, session): factor} where check found that session's host noisier than S1's

    @classmethod
    def beside(cls, cells_path):
        here = Path(cells_path)
        return cls(read_csv(here.with_name("noise.csv")), read_csv(here.with_name("sentinels.csv")))

    def line(self, table, r):
        """(SD, its degrees of freedom) for a cell like r from one of the noise tables (self.cells, steps' relative SDs;
        self.occupancy, occupancy's absolute ones), or None where the table has nothing at its engine and level: the
        cell's own arm, else the largest any arm of its engine shows; at the nearest model, the batch line through its
        batches (batch_line), times the session's scale (check: a noisier host widens every noise of that engine)."""
        own = {k: v for k, v in table.items() if k[:2] == (r["engine"], r["level"])}
        if not own:
            return None
        arm = r.get("arm", "defaults")
        arms = [arm] if any(k[2] == arm for k in own) else sorted({k[2] for k in own})  # none of its own: the largest
        size = math.log(r.get("params") or load_shape(r["model"]).params)
        found = []
        for a in arms:
            cells = {k: v for k, v in own.items() if k[2] == a}
            model = min({k[3] for k in cells}, key=lambda m: abs(math.log(load_shape(m).params) - size))
            found.append(batch_line([(k[4], *v) for k, v in cells.items() if k[3] == model], r["batch"]))
        s, df = max(found)
        return s * self.scale.get((r["engine"], r.get("session")), 1.0), df

    def aa(self):
        """The comparisons' rule (beyond K x the pair's combined noise) on pairs that differ by nothing, each judged by
        noise estimated without it; judged by noise from the same readings, the largest of n processes' pairwise
        differences sits at most s x sqrt(2(n-1)) apart, under the K x sqrt 2 x s line by construction:
        - processes: in each noise-block cell with five processes or more, every two judged at the SD of that cell's
          other processes (leave two out), never below FLOOR;
        - sentinels: each session's end reading of a sentinel cell against its start, at K x sqrt 2 x the first
          session's noise for that cell (the noise model as measured, before check() scales it by these readings).
        {processes, sentinels: {pairs, beyond, rate}, or None without the readings}, or None without either: the
        false-positive rate measured on the study's own null pairs, beside the one chance() assumes (about 0.27% at
        K = 3). Pairs inside one cell are not independent, so the rate is read, not tested."""
        def rate(pairs, beyond):
            return {"pairs": pairs, "beyond": beyond, "rate": beyond / pairs} if pairs else None

        pairs = beyond = 0
        for _cell, steps in self.processes:
            if len(steps) < 5:
                continue
            for i, a in enumerate(steps):
                for j in range(i + 1, len(steps)):
                    rest = [x for k, x in enumerate(steps) if k not in (i, j)]
                    mean = sum(rest) / len(rest)
                    sd = math.sqrt(sum((x - mean) ** 2 for x in rest) / (len(rest) - 1)) / mean
                    pairs += 1
                    beyond += abs(a / steps[j] - 1) > K * math.sqrt(2) * max(sd, FLOOR)
        processes = rate(pairs, beyond)
        if processes:  # the same rule on simulated cells with each cell's own SD: what a calibrated null gives,
            # judged at an SD from n - 2 processes (not derive.chance's pooled degrees of freedom)
            rng, sim_pairs, sim_beyond = random.Random(0), 0, 0
            for _cell, steps in self.processes:
                n = len(steps)
                if n < 5:
                    continue
                mean = sum(steps) / n
                sd = math.sqrt(sum((x - mean) ** 2 for x in steps) / (n - 1)) / mean
                for _ in range(200):
                    xs = [rng.gauss(1.0, sd) for _ in range(n)]
                    for i in range(n):
                        for j in range(i + 1, n):
                            rest = [x for k, x in enumerate(xs) if k not in (i, j)]
                            m = sum(rest) / len(rest)
                            s_rest = math.sqrt(sum((x - m) ** 2 for x in rest) / (len(rest) - 1)) / m
                            sim_pairs += 1
                            sim_beyond += abs(xs[i] / xs[j] - 1) > K * math.sqrt(2) * max(s_rest, FLOOR)
            processes["expected"] = sim_beyond / sim_pairs if sim_pairs else None
        scale, self.scale, pairs, beyond = self.scale, {}, 0, 0
        try:
            for (engine, level, arm, model, batch, session), at in sorted(self.ends.items()):
                if "start" in at and "end" in at:
                    ref = {"engine": engine, "arm": arm, "level": level, "model": model, "batch": batch,
                           "session": session}
                    pairs += 1
                    beyond += abs(at["end"] / at["start"] - 1) > K * math.sqrt(2) * self.sigma(ref)[0]
        finally:
            self.scale = scale
        sentinels = rate(pairs, beyond)
        return {"processes": processes, "sentinels": sentinels} if processes or sentinels else None

    def own_arm(self, table, r):
        """Whether the table measured r's own arm at its engine and level."""
        return any(k[:3] == (r["engine"], r["level"], r.get("arm", "defaults")) for k in table)

    def sigma(self, r):
        """A cell's noise, never below the floor, nor below how far its rerun moved it from the run it superseded
        (rerun_ratio: two readings of one cell are its spread, whatever the noise block says)."""
        found = self.line(self.cells, r)
        moved = abs((r.get("rerun_ratio") or 1.0) - 1)
        if found is None:
            return max(TOLERANCES["step_noise"], FLOOR, moved), None
        return max(found[0], FLOOR, moved), found[1]

    def check(self, head):
        """Whether the noise measured in the first session holds on each session's host: per engine and session, the
        spread of its sentinel cell there (the valid start and end readings, and the grid's own cell of the same
        engine, model and batch in head) against sigma for that cell. A spread above the 95th percentile its degrees
        of freedom allow scales every sigma of that engine in that session up to it (self.scale). Returns the checks,
        for patterns.json."""
        self.scale = {}  # every check starts from S1's noise, including repeated report builds with this instance
        out, scale = [], {}
        for (engine, level, arm, model, batch), by in sorted(self.readings.items()):
            for session, xs in sorted(by.items()):
                cell = next((r for r in head if (r["engine"], r["level"], r["arm"], r["model"], r["batch"],
                                                r.get("session")) == (engine, level, arm, model, batch, session)), None)
                steps = xs + ([cell["step_ms"]] if cell else [])
                if len(steps) < 3:
                    continue
                mean = sum(steps) / len(steps)
                sd = math.sqrt(sum((x - mean) ** 2 for x in steps) / (len(steps) - 1)) / mean
                ref = cell or {"engine": engine, "arm": arm, "level": level, "model": model, "batch": batch}
                s, n = self.sigma(ref)[0], len(steps) - 1
                limit = s * math.sqrt(chi2_95(n) / n)
                out.append({"engine": engine, "level": level, "session": session, "model": model, "batch": batch,
                            "readings": len(steps), "sd": sd, "sigma": s, "limit": limit, "holds": sd <= limit})
                if sd > limit:
                    scale[(engine, session)] = max(scale.get((engine, session), 1.0), sd / s)
        self.scale = scale
        return out

    def drift(self, engine, a, b):
        """The largest relative change of the engine's sentinel cells between sessions a and b, each the mean of its
        valid locked-clock readings; None without a reading in both."""
        ds = [abs(by[a] / by[b] - 1) for k, by in self.sentinels.items() if k[0] == engine and a in by and b in by]
        return max(ds) if ds else None

    def combined(self, a, b):
        """(The relative noise on a's step over b's, its degrees of freedom): each one's sigma, and when they come
        from different sessions each engine's drift between them once."""
        (sa, da), (sb, db) = self.sigma(a), self.sigma(b)
        terms = [sa, sb]
        if a.get("session") != b.get("session"):
            for e in dict.fromkeys((a["engine"], b["engine"])):
                d = self.drift(e, a.get("session"), b.get("session"))
                terms.append(TOLERANCES["sentinel_drift"] if d is None else d)
        dfs = [d for d in (da, db) if d is not None]
        return math.sqrt(sum(t * t for t in terms)), min(dfs) if dfs else None


def chi2_95(n):
    """The 95th percentile of chi-square with n degrees of freedom (Wilson-Hilferty beyond the table)."""
    table = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.07, 6: 12.59}
    return table.get(n) or n * (1 - 2 / (9 * n) + 1.645 * math.sqrt(2 / (9 * n))) ** 3


def batch_line(points, batch):
    """(sigma at batch, degrees of freedom) from [(batch, sd, df)]: log variance as a line in log2(batch), least squares
    weighted by df, each point's log variance corrected for its bias (E log s^2 = log sigma^2 - 1/df - 1/(3 df^2), near
    enough); one batch alone is its own sd. batch is held inside the range measured. The degrees of freedom are the
    fitted value's own, where it is read: a log variance carries variance 2/df, so the line's value at x carries
    2/W + 2 (x - mean)^2 / sum(df (xi - mean)^2), fewest at the ends of the range (three batches of 5 give 15 at the
    middle, about 6 at either end)."""
    pts = [(math.log2(b), 2 * math.log(sd) + 1 / df + 1 / (3 * df * df), df) for b, sd, df in points if sd > 0 and df]
    if not pts:
        return 0.0, None
    if len({x for x, _, _ in pts}) == 1:
        return math.exp(sum(y * w for _, y, w in pts) / sum(w for _, _, w in pts) / 2), sum(w for _, _, w in pts)
    w = sum(d for _, _, d in pts)
    mx, my = sum(x * d for x, _, d in pts) / w, sum(y * d for _, y, d in pts) / w
    slope = sum(d * (x - mx) * (y - my) for x, y, d in pts) / sum(d * (x - mx) ** 2 for x, _, d in pts)
    sxx = sum(d * (x - mx) ** 2 for x, _, d in pts)
    x = min(max(math.log2(batch), min(p[0] for p in pts)), max(p[0] for p in pts))
    return math.exp((my + slope * (x - mx)) / 2), max(1, round(2 / (2 / w + 2 * (x - mx) ** 2 / sxx)))


def load(cells):
    """(rows, (trace rows, stretch), noise): the one reader of the cell table and the files anatomy.tidy writes beside
    it (strips.json, noise.csv, sentinels.csv), for the page, the post's figures and the patterns."""
    from anatomy.tidy import load_cells

    return load_cells(cells), strips(cells), Noise.beside(cells)


def agreement(path):
    """The greedy-token check's summary from anatomy.tidy's agreement.csv: {identity: [settings that gave their
    defaults' tokens, of those compared], pairs: engine pairs compared, median_first_divergence: [least, most],
    flagged: the engines that parted from every other at once}; None without the file."""
    rows = read_csv(path)
    if not rows:
        return None
    yes = lambda r: str(r.get("flagged")) == "True"  # noqa: E731  (CSV: the bool as text)
    ident = [r for r in rows if r["kind"] == "identity"]
    meds = [float(r["median_first_divergence"]) for r in rows
            if r["kind"] == "pair" and r.get("median_first_divergence") not in (None, "")]
    return {"identity": [sum(not yes(r) for r in ident), len(ident)],
            "pairs": sum(r["kind"] == "pair" for r in rows),
            "median_first_divergence": [min(meds), max(meds)] if meds else None,
            "flagged": sorted({r["odd"] for r in rows if r["kind"] == "pair" and yes(r) and r.get("odd")}),
            "probe_errors": sum(r["kind"] == "probe-error" for r in rows)}


def strips(cells):
    """(trace rows for the opening picture, how much longer the traced steps ran than untraced ones) from the
    strips.json anatomy.tidy writes beside the cell table; ([], None) without one."""
    path = Path(cells).with_name("strips.json")
    if not path.exists():
        return [], None
    cut = json.loads(path.read_text())
    rows = [{**x, "engine": e} for e, s in cut.items() for x in s["rows"]]
    stretch = sorted(s["traced_step_ms"] / s["step_ms"] - 1 for s in cut.values() if s.get("step_ms"))
    return rows, stretch[len(stretch) // 2] if stretch else None


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def compare(a, b, noise):
    """a's step over b's: {ratio, sigma (its relative noise, one SD), df (of that estimate; None when assumed),
    beyond (|ratio - 1| > K sigma)}."""
    ratio, (s, df) = a["step_ms"] / b["step_ms"], noise.combined(a, b)
    return {"ratio": ratio, "sigma": s, "df": df, "beyond": abs(ratio - 1) > K * s}


@functools.cache
def chance(k, df, draws=20000):
    """How many of a group's k - 1 comparisons against its fastest member cross K x their noise when the k are truly
    equal: each member's noise estimated on its own with df degrees of freedom (None: known), and each comparison
    judged on both members' estimates, as compare() judges it; by simulation, a fixed seed. The noise floor only
    raises an estimate, so this is an upper bound on what the floored test gives."""
    rng = random.Random(1000 * k + (df or 0))
    est = (lambda: math.sqrt(sum(rng.gauss(0, 1) ** 2 for _ in range(df)) / df)) if df else (lambda: 1.0)
    hits = 0
    for _ in range(draws):
        xs = [(rng.gauss(0, 1), est()) for _ in range(k)]
        lo, s_lo = min(xs)
        hits += sum(x - lo > K * math.hypot(s, s_lo) for x, s in xs if x != lo)
    return hits / draws



def occupancy_noise(r, noise):
    """occupied's noise, absolute (in occupancy's units): where the noise block traced the cell's own engine, level and
    arm, occupancy's own spread between processes there (Noise.line: nearest model, the batch line, the session's
    scale), never below the floor; otherwise the larger of the spread any arm of its engine shows and the cell's step
    noise carried onto its occupancy (one process's trace fraction; sqrt 2 of it when occupied is busy over another
    process's untraced step). A host-bound arm never inherits a GPU-bound defaults cell's small spread: in the pod's
    data the defaults were about 98% busy, graphs-off 14-19%. None without an occupancy."""
    if r.get("occupied") is None:
        return None
    floor = FLOOR * r["occupied"]
    two = r.get("occupied_source") == "trace over the untraced step"
    from_steps = noise.sigma(r)[0] * r["occupied"] * (math.sqrt(2) if two else 1.0)
    found = noise.line(noise.occupancy, r)
    if found is not None and noise.own_arm(noise.occupancy, r):
        return max(found[0], floor)
    return max(found[0] if found else 0.0, from_steps, floor)


def tracing_bias(r):
    """How far a trace's own fraction can read low: where occupied is the trace's fraction, tracing's stretch of the
    step (the profiler overhead) inside it, which the occupancy estimator did not take out (anatomy.tidy.occupancy);
    0 where occupied is busy over the untraced step, which has none."""
    return max(r.get("profiler_overhead") or 0.0, 0.0) if r.get("occupied_source") == "trace" else 0.0


def waiting(r, noise):
    """Whether the GPU waited on the host in this cell: occupied under the line by more than K x its noise plus what
    tracing can take off it (tracing_bias)."""
    return r.get("occupied") is not None and \
        r["occupied"] < OCCUPIED - K * occupancy_noise(r, noise) - tracing_bias(r)


def crossover_bound(points, level=OCCUPIED):
    """points: [(batch, occupied[, its noise])] for every batch attempted, occupied None where no accepted measurement
    exists (a rejected cell, no trace). The batch from which occupied stays at or above level: after the last batch
    below it by more than K x that batch's noise, the first at or above it, log2-interpolated from the batch before
    ("exact"); where that batch is unknown the crossover is only bounded, (the batch, "upper"), never interpolated
    across a gap; (None, None) when it never gets there and stays."""
    pts = sorted((p[0], p[1], p[2] if len(p) > 2 and p[2] else 0.0) for p in points)
    below = [i for i, (_, o, n) in enumerate(pts) if o is not None and o < level - K * n]
    start = below[-1] + 1 if below else 0
    first = next((i for i in range(start, len(pts)) if pts[i][1] is not None and pts[i][1] >= level), None)
    if first is None:
        return None, None
    b, o, _ = pts[first]
    if first == 0:
        return float(b), "exact"
    pb, po, _ = pts[first - 1]
    if po is None:  # the batch below was not measured: the crossing is somewhere below b
        return float(b), "upper"
    t = (level - po) / (o - po)
    return 2 ** (math.log2(pb) + t * (math.log2(b) - math.log2(pb))), "exact"


def crossovers(rows, noise, level=OCCUPIED):
    """Per headline engine and model, over every batch attempted: {crossover_batch, bound ("exact", "upper", or None
    when no accepted batch reaches the line and stays), crossover_text, rejected (cells), measured (accepted cells)}.
    The text never claims more than the cells resolve: a crossing between two measured batches reads "4-8", not an
    interpolated 6.2. A rejected cell is unknown, not below the line (in validation, counting a rejected batch-1 cell
    as below put SGLang at batch 8)."""
    by = {}
    for r in headline_rows(rows, only_accepted=False):
        by.setdefault((r["engine"], r["model"], r["params"]), []).append(r)
    out = []
    for (engine, model, params), rs in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        pts = [(r["batch"], r.get("occupied") if accepted(r) else None, occupancy_noise(r, noise) if accepted(r)
                else None) for r in rs]
        b, bound = crossover_bound(pts, level)
        batches = sorted(p[0] for p in pts)
        if b is None:
            text = None
        elif bound == "upper":
            text = f"at most {b:.0f}"
        elif b in batches:
            text = f"{b:.0f}"
        else:
            text = f"{max(x for x in batches if x < b)}–{min(x for x in batches if x > b)}"
        out.append({"engine": engine, "model": model, "params": params, "crossover_batch": b, "bound": bound,
                    "measured": sum(map(accepted, rs)), "rejected": sum(not accepted(r) for r in rs),
                    "crossover_text": text})
    return out


def engine_gaps(rows, noise):
    """Like with like: at each model, batch and level (L1 against L1, L2 against L2, so llama.cpp's server never meets
    an engine API), every engine's step over the fastest engine's there. One row per engine and cell; the fastest's own
    has ratio 1 and no comparison (sigma, df, beyond None); `clear` says whether every other engine there is beyond
    noise of the fastest (else the fastest is tied)."""
    by = {}
    for r in defaults(rows):
        if r["level"] in ("L1", "L2"):
            by.setdefault((r["model"], r["batch"], r["level"]), []).append(r)
    out = []
    for (model, batch, level), rs in sorted(by.items(), key=lambda kv: (kv[1][0]["params"], kv[0][1], kv[0][2])):
        if len(rs) < 2:
            continue
        best = min(rs, key=lambda r: r["step_ms"])
        same = {"ratio": 1.0, "sigma": None, "df": None, "beyond": None}
        found = [(r, same if r is best else compare(r, best, noise)) for r in sorted(rs, key=lambda r: r["step_ms"])]
        clear = all(c["beyond"] for r, c in found if r is not best)
        for r, c in found:
            out.append({"engine": r["engine"], "model": model, "params": r["params"], "batch": batch, "level": level,
                        "step_ms": r["step_ms"], "fastest": best["engine"], "fastest_ms": best["step_ms"],
                        "clear": clear, **c, "cells": [row_cell_id(best)] + ([] if r is best else [row_cell_id(r)])})
    return out


def mitigations(rows, noise):
    """Each accepted arm cell against the accepted defaults cell of the same engine, version, model, batch and prompt,
    at the engine's headline level: occupied and step with every mitigation on and with that arm's one changed, and
    what the arm's captures measure of each side: the GPU's busy time per step (the graph trace's, busy_ms) and the
    host stall, step - busy. An arm
    that changes only the host's schedule (SERIAL_ARMS: async scheduling, the overlap scheduler) must leave the GPU's
    busy time alone, so its step's change is host time (gpu_work_held, judged as busy_invariance is, at the larger of
    the busy_invariant tolerance and K x the busy time's noise): a check that the anatomy attributes, not only adds
    up. The other arms change the GPU's work by design, and are reported without that check."""
    head = headline_rows(rows, arms=None)
    base = {(r["engine"], r.get("version"), r["model"], r["batch"]): r for r in head if r["arm"] == "defaults"}
    out = []
    for r in head:
        b = base.get((r["engine"], r.get("version"), r["model"], r["batch"]))
        if r["arm"] == "defaults" or b is None:
            continue
        # the trace's own busy time per step, never occupied x step: where occupied is the trace's fraction (a
        # stretched step), that product carries the tracing overhead, -1% to -3% on a defaults side, against an
        # exact serial arm: enough to flip gpu_work_held at its 2% margin
        busy = [x.get("busy_ms") if x.get("occupied") is not None else None for x in (b, r)]
        held = None
        if r["arm"] in SERIAL_ARMS and None not in busy:
            noise_ = math.hypot(*(occupancy_noise(x, noise) / x["occupied"] for x in (b, r) if x["occupied"]))
            held = abs(busy[1] / busy[0] - 1) <= max(TOLERANCES["busy_invariant"], K * noise_)
        out.append({"engine": r["engine"], "model": r["model"], "params": r.get("params"), "batch": r["batch"],
                    "arm": r["arm"], "arm_label": arm_name(r["arm"]), "level": r["level"],
                    "occupied_on": b.get("occupied"), "occupied_off": r.get("occupied"), "step_on": b["step_ms"],
                    "step_off": r["step_ms"], "busy_on": busy[0], "busy_off": busy[1],
                    "stall_on": b["step_ms"] - busy[0] if busy[0] is not None else None,
                    "stall_off": r["step_ms"] - busy[1] if busy[1] is not None else None,
                    "gpu_work_held": held, **compare(r, b, noise), **pairing(b, r)})
    return out


def pairing(a, b):
    """A paired comparison's cells and sessions, and whether it is exploratory: a mitigation or a level of the stack
    is a controlled change only within one session (one host, one rental); across sessions a drift term stands in for
    the control, so such a pair is listed but never feeds a principle (tables)."""
    sessions = [a.get("session"), b.get("session")]
    return {"cells": [row_cell_id(a), row_cell_id(b)], "sessions": sessions, "exploratory": sessions[0] != sessions[1]}


def stack(rows, noise):
    """The same cell at another level of the stack against the engine's headline level (the engine API, L1; for
    llama.cpp, which has no engine API, its server): the runner (L0) and the server (L2)."""
    at = {(r["engine"], r["model"], r["batch"], r["level"]): r for r in defaults(rows)}
    out = []
    for (engine, model, batch, level), r in sorted(at.items(), key=lambda kv: (kv[1]["params"], kv[0])):
        head = at.get((engine, model, batch, HEADLINE.get(engine)))
        if head is None or head is r:
            continue
        out.append({"engine": engine, "model": model, "params": r["params"], "batch": batch, "level": level,
                    "headline_level": head["level"], "step_ms": r["step_ms"], "headline_ms": head["step_ms"],
                    **compare(r, head, noise), **pairing(head, r)})
    return out


def parts(r, noise):
    """Where one cell's step goes, as shares of it: {floor, copy, excess (lo, hi), launch (lo, hi), gaps, stall,
    stall_resolved, largest}. The launch term is the node-cost range (the ruler bounds what a graph node costs, and
    the overlap model is read with and without tracing's inflation) with the gaps inside graphs apart, kernel excess
    the range that leaves; `largest` names the part beyond the floor that is the largest at both ends of that range,
    else every part that is at one end. The stall is resolved only beyond K x the occupancy's noise plus tracing's
    bias. None for a cell without a decomposition or whose terms do not add up (consistent False)."""
    if r.get("floor_share") is None or r.get("consistent") is False:
        return None
    step = r["step_ms"]
    primary = (r.get("launch_floor_ms") or 0) / step
    node = tuple(r.get("launch_floor_share_range") or (primary, primary))
    excess = tuple(r["kernel_exc_share"] + primary - x for x in node)
    gaps, stall, copy = (r.get("bubbles_ms") or 0) / step, r["host_stall_share"], r.get("copy_share") or 0.0
    ends = [{"kernel excess": excess[i], "launching kernels": node[i], "gaps inside graphs": gaps,
             "copies": copy, "host stall": stall} for i in (0, 1)]
    return {"floor": r["floor_share"], "copy": copy, "excess": excess, "launch": node, "gaps": gaps, "stall": stall,
            "stall_resolved": abs(stall) > K * (occupancy_noise(r, noise) or 0) + tracing_bias(r),
            "largest": sorted({max(e, key=e.get) for e in ends})}


# SERIAL_ARMS (anatomy.metrics): the host runs before the GPU every step, so its time shows in every cell


def headline_fits(rows, arms=("defaults",), noise=None):
    """The calculator's model: one host-cost fit per engine (anatomy.fit), on its headline cells, its interval's
    process offset the median between-process noise of those cells (noise.sigma; none without a noise model). With
    arms=SERIAL_ARMS, the fit on the arms that switch overlap off, where the host's time per step is visible in every
    cell rather than only where it holds the GPU up (a cross-check reported in patterns.json, not the calculator's
    input: overlap changes the host's own work too)."""
    head = [r for r in headline_rows(rows, arms=None) if r["arm"] in arms]
    out = {}
    for engine in HEADLINE:
        mine = [r for r in head if r["engine"] == engine]
        offset = median(noise.sigma(r)[0] for r in mine) if noise is not None and mine else None
        f = fit.fit(mine, offset_sd=offset)
        if f is not None:
            out[engine] = {k: f.get(k) for k in ("form", "h0_ms", "h1_ms", "h0_interval", "h1_interval",
                                                 "interval_method",
                                                 "median_error", "n_fitted", "n_host_cells", "host_upper_bound_ms",
                                                 "h1_fixed_at_zero", "unphysical", "residuals")}
            out[engine] |= {"cells": sum(r["engine"] == engine for r in head),
                            "lomo_median_error": f["lomo"]["median"], "lomo_cells": f["lomo"]["cells"]}
    return out


def finite(x):
    """x with every non-finite float replaced by None, recursively: the page data is strict JSON."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def tables(rows, noise=None):
    """Every derived table, built once: what the principles, the patterns, the page and the post's figures read."""
    noise = noise or Noise()
    head = headline_rows(rows)
    checks = noise.check(defaults(rows))  # first: compare each sentinel with its own level
    cross = crossovers(rows, noise)
    return {"rows": rows, "noise": noise, "noise_check": checks, "head": head, "crossovers": cross,
            "frontier": [c for c in cross if c["crossover_batch"] is not None],
            "gaps": engine_gaps(rows, noise), **paired(mitigations(rows, noise), stack(rows, noise)),
            "stack_rejected": rejected_servers(rows),
            "parts": {row_cell_id(r): p for r in head if (p := parts(r, noise))},
            "fits": headline_fits(rows, noise=noise), "serial_fits": headline_fits(rows, SERIAL_ARMS, noise=noise)}


def rejected_servers(rows):
    """The server cells the stack table cannot hold: an engine's server (L2, defaults) at the headline prompt whose
    cell was rejected, with its reason, so the stack claim names what it does not compare (S2-20261003: TensorRT-LLM's
    server at 32 and 64 streams on 0.6B and at 64 on 4B, sequences fallen behind the batch)."""
    return [{"engine": r["engine"], "model": r["model"], "batch": r["batch"], "rejection": rejection(r),
             "params": r.get("params") or load_shape(r["model"]).params}
            for r in defaults(rows, only_accepted=False) if r.get("level") == "L2" and not accepted(r)
            and HEADLINE.get(r["engine"]) == "L1"]


def paired(mits, stk):
    """{mitigations, stack: the same-session pairs the principles, patterns and charts read; exploratory: the
    cross-session ones (patterns.json; the mitigation chart draws its own in rows of their own, never a principle),
    each with its pair: "mitigation" or "stack"}."""
    return {"mitigations": [m for m in mits if not m["exploratory"]],
            "stack": [s for s in stk if not s["exploratory"]],
            "exploratory": [{**x, "pair": pair} for pair, xs in (("mitigation", mits), ("stack", stk))
                            for x in xs if x["exploratory"]]}


def about(rows):
    """What a table is, for every export made from it (page.json, patterns.json, the figures): its status (preview,
    validation, study: metrics.dataset_status) and why its rows are not the study's (not_study), the code that built
    it, the cards, engines, models and sessions it holds, and when it was measured; a detached file says it all without
    the page."""
    import datetime

    from anatomy.metrics import NOT_STUDY, current_versions, dataset_status, not_study
    from servebench.env import git_sha

    day = lambda t: datetime.datetime.fromtimestamp(t, datetime.UTC).strftime("%Y-%m-%d")  # noqa: E731
    written = sorted(r["written"] for r in rows if isinstance(r.get("written"), (int, float)))
    models = sorted({r["model"] for r in rows if r.get("model")}, key=lambda m: load_shape(m).params)
    why = {k for r in rows if not r.get("synthetic") for k in not_study(r)}
    return {"schema": "decode-anatomy/export/1", "dataset_status": dataset_status(rows),
            "not_study": [NOT_STUDY[k] for k in NOT_STUDY if k in why],
            "source_commit": git_sha(ROOT),
            "generated": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "cards": sorted({r["gpu"] for r in rows if r.get("gpu")}), "engines": current_versions(rows),
            "models": [model_name(m) for m in models], "batches": sorted({int(r["batch"]) for r in rows}),
            "sessions": sorted({r["session"] for r in rows if r.get("session")}),
            "measured": {"first": day(written[0]), "last": day(written[-1])} if written else None}
