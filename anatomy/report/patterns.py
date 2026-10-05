"""What the cell table shows, in words, from anatomy.report.derive's tables (built once, derive.tables), so a sentence
never says more than they do:
- principles: the page's opening, at most one per chart, each a short claim with the numbers behind it. A claim is
  only as strong as its evidence: a superlative only where it holds beyond noise, a trend only where it holds across
  every size measured, "within noise" stated as the most it could hide;
- patterns: every difference beyond noise, one sentence each, citing its cells (patterns.json, the post's source).

Kinds, in the page's order: crossover (where the GPU stops waiting on the host), anatomy (where the step goes),
engines (each engine against the fastest, like with like), mitigations (one switched off), stack (the server and the
runner against the engine API), host (the fitted host cost).

    python -m anatomy.report.patterns results/cells.parquet [--json results/patterns.json]
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from anatomy import fit
from anatomy.metrics import HEADLINE, defaults
from anatomy.report import derive
from anatomy.report.charts import ARMS, ENGINE_NAMES, ENGINES
from anatomy.report.derive import K
from servebench.results import row_cell_id

LEVEL_NAMES = {"L0": "the runner (L0)", "L1": "the engine API (L1)", "L2": "the server (L2)"}


def size(model):
    """"0.6B" from "Qwen/Qwen3-0.6B"; "1B" from "unsloth/Llama-3.2-1B-Instruct"."""
    m = re.search(r"-(\d+(?:\.\d+)?B)\b", model)
    return m.group(1) if m else model.split("/")[-1]


def pct(x):
    return f"{100 * x:.0f}%" if abs(x) >= 0.095 else f"{100 * x:.1f}%"


def span(xs, fmt=pct):
    lo, hi = min(xs), max(xs)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}–{fmt(hi)}"


def longer(ratio, s=None):
    """"12% longer (± 0.8%, one SD)", or "2.1x as long" once the ratio says it better."""
    if ratio >= 1.5:
        return f"{ratio:.1f}x as long"
    return f"{pct(abs(ratio - 1))} {'longer' if ratio > 1 else 'shorter'}{f' (± {pct(s)}, one SD)' if s else ''}"


def than(ratio, whose):
    """"41% longer than the fastest", "2.1x as long as the fastest"."""
    return f"{longer(ratio)} {'as' if ratio >= 1.5 else 'than'} {whose}"


def of(k, n, what="cells", its=False):
    """"3 of 5 cells", "all 4 cells", "both cells", "the one cell": k of n things, counted the way they read; with its,
    "3 of its 5 cells", "all 4 of its cells", "both its cells", "its one cell"."""
    one = what.removesuffix("s")
    if its:
        return f"{k} of its {n} {what}" if k != n else {1: f"its one {one}", 2: f"both its {what}"}.get(
            n, f"all {n} of its {what}")
    if k != n:
        return f"{k} of {n} {what}"
    return {1: f"the one {one}", 2: f"both {what}"}.get(n, f"all {n} {what}")


def names(engines):
    ns = [ENGINE_NAMES[e] for e in engines]
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def sizes(models_by_params):
    """"0.6B", "0.6B–4B" for the first and last of models sorted by size."""
    ms = [size(m) for m in models_by_params]
    return ms[0] if len(ms) == 1 else f"{ms[0]}–{ms[-1]}"


def lower(label):
    """A label mid-sentence: "Async scheduling off" -> "async scheduling off", "CUDA graphs off" kept."""
    return label if label[:2].isupper() else label[:1].lower() + label[1:]


def by_size(rows):
    return [m for _, m in sorted({(r["params"], r["model"]) for r in rows})]


def batches(xs):
    lo, hi = min(xs), max(xs)
    return f"batch {lo}" if lo == hi else f"batch {lo}–{hi}"


# principles: one per chart, each citing every cell it read (`cells`): a page built from study data checks them
# (check), and a finding's card links them


def ids(rows):
    return sorted({row_cell_id(r) for r in rows})


def cited(pairs):
    """Every cell a list of comparisons names."""
    return sorted({c for x in pairs for c in x["cells"]})


def check(report, rows):
    """Why the principles cannot stand on these rows ([] when they can): one that cites no cell, or cites a cell that
    is not an accepted row. page.build refuses a study page that fails it."""
    accepted = {row_cell_id(r) for r in rows if derive.accepted(r)}
    out = []
    for p in report["principles"]:
        if not p.get("cells"):
            out.append(f"{p['section']}: cites no cell")
        elif missing := [c for c in p["cells"] if c not in accepted]:
            out.append(f"{p['section']}: cites {len(missing)} cells that are not accepted rows, {missing[0]} first")
    return out


def crossover_principle(t):
    """Where the GPU waits on the host: headline cells under 90% occupied by more than K x their noise. A size bound
    is claimed only where every engine that waits was also measured, without waiting, at a larger model."""
    head = [r for r in t["head"] if r.get("occupied") is not None]
    if not head:
        return None
    wait = [r for r in head if derive.waiting(r, t["noise"])]
    if not wait:
        least = min(head, key=lambda r: r["occupied"])
        return {"section": "crossover", "claim": "The GPU barely waits on the host, even at batch 1.",
                "evidence": f"No cell is under 90% busy beyond its noise; the least busy is {pct(least['occupied'])} "
                            f"({ENGINE_NAMES[least['engine']]}, {size(least['model'])}, batch {least['batch']}).",
                "cells": ids(head)}
    engines = [e for e in HEADLINE if any(r["engine"] == e for r in wait)]
    top = max(r["params"] for r in wait)
    bounded = all(any(r["engine"] == e and r["params"] > top for r in head) for e in engines)
    per = []
    for e in engines:
        ws = [r for r in wait if r["engine"] == e]
        n = sum(r["engine"] == e for r in head)
        per.append(f"{ENGINE_NAMES[e]} in {of(len(ws), n, its=True)} ({sizes(by_size(ws))}, "
                   f"{batches([r['batch'] for r in ws])})")
    never = [e for e in HEADLINE if e not in engines and any(r["engine"] == e for r in head)]
    model = size(next(r["model"] for r in wait if r["params"] == top))
    return {"section": "crossover",
            "claim": f"Only models up to {model} leave the GPU waiting on the host." if bounded else
                     f"The GPU waits on the host in {of(len(wait), len(head))} measured.",
            "evidence": f"Under 90% busy beyond noise in {of(len(wait), len(head))}: " + "; ".join(per)
                        + (f". {names(never)} never" if never else "") + ".", "cells": ids(head)}


def anatomy_principle(t):
    """The bandwidth floor's share of a batch-1 step at the smallest and the largest model, and what fills the rest.
    A cell whose session could borrow no node-trace inflation (calibration_lent False, anatomy.calibration.lend) is
    left out: its
    kernel time keeps the trace's lengthening, so its split between kernel excess and launch is not the card's."""
    cells = [(r, t["parts"][row_cell_id(r)]) for r in t["head"]
             if r["batch"] == 1 and row_cell_id(r) in t["parts"] and r.get("calibration_lent") is not False]
    if not cells:
        return None
    models = by_size([r for r, _ in cells])
    small, large = models[0], models[-1]
    fs = [p["floor"] for r, p in cells if r["model"] == small]
    fl = [p["floor"] for r, p in cells if r["model"] == large]
    n_small = sum(r["model"] == small for r, _ in cells)
    winners = Counter(p["largest"][0] for r, p in cells if r["model"] == small and len(p["largest"]) == 1)
    part, n = winners.most_common(1)[0] if winners else (None, 0)
    key = {"kernel excess": "excess", "launching kernels": "launch"}.get(part)
    # "mostly X" only where X is at least half of what lies above the floor, at both ends of the launch range, for
    # every engine; else X is only the largest part
    most = part and n == n_small and key and all(min(p[key]) >= (1 - p["floor"]) / 2 for r, p in cells
                                                  if r["model"] == small)
    mostly = f", mostly {part}" if most else f", the largest part of it {part}" if part and n == n_small else ""
    if max(fs) < 0.5:
        claim = f"Most of a {size(small)} step is spent above the bandwidth floor{mostly}" + \
            (f"; most of a {size(large)} step is the floor." if small != large and min(fl) >= 0.5 else ".")
    elif min(p["floor"] for _, p in cells) >= 0.5:
        claim = "The bandwidth floor is most of every step at batch 1."
    else:
        claim = f"The bandwidth floor is {span(fs)} of a {size(small)} step at batch 1, depending on the engine."
    evidence = f"At batch 1 the floor is {span(fs)} of a {size(small)} step" + \
               (f" and {span(fl)} of a {size(large)} step" if small != large else "")
    if part:
        evidence += (f". Above the floor, {part} is the largest part for "
                     + ("every engine" if n == n_small else f"{n} of {n_small} engines") + f" at {size(small)}")
    return {"section": "anatomy", "claim": claim, "evidence": evidence + ".", "cells": ids(r for r, _ in cells)}


def engines_principle(t):
    """How far apart the engines are, like with like (the engine API; the server beside it). "Most on small models"
    only where the largest gap falls with size across every model, over the batches all of them were measured at;
    a fastest engine counted only where every other is beyond noise of it."""
    gaps = t["gaps"]
    level = "L1" if any(g["level"] == "L1" and g["sigma"] is not None for g in gaps) else "L2"
    cmp = [g for g in gaps if g["level"] == level and g["sigma"] is not None]
    if not cmp:
        return None
    real = [g for g in cmp if g["beyond"]]
    where = LEVEL_NAMES[level].split(" (")[0]
    if not real:
        return {"section": "engines", "claim": "The engines are within noise of one another.",
                "evidence": f"At {where}, all {len(cmp)} differences from the fastest are under {K:g} times the "
                            "run-to-run noise.", "cells": cited(cmp)}
    models = by_size(cmp)
    common = set.intersection(*({g["batch"] for g in cmp if g["model"] == m} for m in models))
    top = [max((g["ratio"] if g["beyond"] else 1.0 for g in cmp if g["model"] == m and g["batch"] in common),
               default=1.0) for m in models]
    worst = max(real, key=lambda g: g["ratio"])
    if len(models) > 1 and common and top[0] > 1 and all(a >= b for a, b in zip(top, top[1:])):
        claim = "Engines differ most on small models."
        evidence = (f"At {where} ({batches(common)}), the slowest step is up to {than(top[0], 'the fastest')} at "
                    f"{size(models[0])}, and " + ("within noise" if top[-1] == 1 else f"{longer(top[-1])}")
                    + f" at {size(models[-1])}")
    else:
        claim = f"The slowest engine's step is up to {than(worst['ratio'], 'the fastest')}."
        evidence = f"At {where}, {size(worst['model'])}, batch {worst['batch']}"
    cells = {(g["model"], g["batch"]): (g["fastest"], g["clear"]) for g in cmp}
    clear = Counter(e for e, c in cells.values() if c)
    if clear:
        e, n = clear.most_common(1)[0]
        tied = sum(not c for _, c in cells.values())
        evidence += (f". {ENGINE_NAMES[e]} is the fastest beyond noise in {of(n, len(cells))} compared"
                     + (f"; in {tied} the fastest is within noise of another engine" if tied else ""))
    served = [g for g in gaps if g["level"] == "L2" and g["sigma"] is not None and g["beyond"]]
    if level == "L1" and served:
        w = max(served, key=lambda g: g["ratio"])
        evidence += f". Among the servers, {ENGINE_NAMES[w['engine']]} is up to {than(w['ratio'], 'the fastest')}"
    return {"section": "engines", "claim": claim, "evidence": evidence + ".", "cells": cited(cmp)}


def mitigations_principle(t):
    """The largest step change any one mitigation makes. "Matters most" only where at least two were measured and,
    on the leader's own engine, its largest change is beyond noise of every other arm's."""
    ms = t["mitigations"]
    if not ms:
        return None
    effect = lambda m: abs(math.log(m["ratio"]))  # noqa: E731
    top = {}
    for m in ms:
        if m["beyond"] and (m["arm"] not in top or effect(m) > effect(top[m["arm"]])):
            top[m["arm"]] = m
    if not top:
        return {"section": "mitigations", "claim": "No mitigation changes the step beyond noise.",
                "evidence": f"{len(ms)} cells measured with one mitigation changed.", "cells": cited(ms)}
    ranked = sorted(top.values(), key=effect, reverse=True)
    lead = ranked[0]
    # "matters most" only where, on every engine that measured the leading arm, its largest effect beats every other
    # arm's on that engine by more than K x their combined noise
    def beats(engine):
        mine = max((m for m in ms if m["arm"] == lead["arm"] and m["engine"] == engine), key=effect)
        rivals = [max((m for m in ms if m["arm"] == a and m["engine"] == engine), key=effect)
                  for a in {m["arm"] for m in ms if m["engine"] == engine} - {lead["arm"]}]
        return bool(rivals) and all(effect(mine) - effect(r) > K * math.hypot(mine["sigma"], r["sigma"])
                                    for r in rivals)
    wins = all(beats(e) for e in {m["engine"] for m in ms if m["arm"] == lead["arm"]})
    off = ARMS.get(lead["arm"], (None, None, None))[2]
    if wins and off and lead["ratio"] > 1:
        claim = f"Of the mitigations, {off} matter{'' if off.endswith('s') or ' and ' in off else 's'} most."
    elif off and lead["ratio"] > 1:
        claim = f"Switching off {off} makes a step up to {longer(lead['ratio'])}."
    else:
        claim = f"{lead['arm_label']} changes the step most: up to {longer(lead['ratio'])}."
    rest = [f"{lower(m['arm_label'])}, up to {longer(m['ratio'])} ({ENGINE_NAMES[m['engine']]})" for m in ranked[1:3]]
    quiet = sorted({lower(m["arm_label"]) for m in ms} - {lower(m["arm_label"]) for m in ranked})
    where = f"{ENGINE_NAMES[lead['engine']]}, {size(lead['model'])}, batch {lead['batch']}"
    said = f"Largest at {where}" if longer(lead["ratio"]) in claim else \
        f"{lead['arm_label']}, the step is up to {longer(lead['ratio'])} ({where})"
    evidence = (said + (f"; then {'; '.join(rest)}" if rest else "")
                + (f". Within noise everywhere: {', '.join(quiet)}" if quiet else ""))
    return {"section": "mitigations", "claim": claim, "evidence": evidence + ".", "cells": cited(ms)}


def stack_principle(t):
    """The server against the engine API over every cell measured at both. Within noise is stated as the most it
    could hide: the largest difference plus K x its noise."""
    st = [s for s in t["stack"] if s["level"] == "L2" and s["headline_level"] == "L1"]
    if not st:
        return None
    real = [s for s in st if s["beyond"]]
    worst = max(st, key=lambda s: abs(s["ratio"] - 1))
    scope = f"{sizes(by_size(st))}, {batches([s['batch'] for s in st])}"
    gone = t.get("stack_rejected") or []  # the server cells the claim does not compare, beside it
    by_engine = {}
    for g in gone:
        by_engine.setdefault(g["engine"], []).append(g)
    behind = all("fell behind the batch" in str(g.get("rejection") or "") for g in gone)
    missing = ("" if not gone else f" Not compared: {of(len(gone), len(gone), 'server cells')} rejected, "
               + "; ".join(f"{ENGINE_NAMES[e]} at {batches([g['batch'] for g in gs]).removeprefix('batch ')} streams "
                           f"on {sizes(by_size(gs))}" for e, gs in by_engine.items())
               + (" (sequences fell behind the batch: a frontend that could not keep its streams even on this host)"
                  if behind else " (see their rejections)") + ".")
    if not real:
        # the bound is per engine: one engine's noisy server (TensorRT-LLM's on the study's 8-core host, its noise
        # four times the others') would otherwise set the whole stack's "at most" (17% where vLLM's and SGLang's
        # servers sat within 2%)
        bounds = {e: max(s["ratio"] - 1 + K * s["sigma"] for s in st if s["engine"] == e)
                  for e in sorted({s["engine"] for s in st}, key=list(ENGINES).index)}
        hidden = ", ".join(f"{pct(b)} for {ENGINE_NAMES[e]}" for e, b in bounds.items())
        return {"section": "stack", "claim": "The server stack costs nothing beyond noise over the engine API.",
                "evidence": f"Within noise in {of(len(st), len(st))} measured at both ({scope}); the most it could "
                            f"hide is {hidden}; the largest difference is {pct(worst['ratio'] - 1)} (± "
                            f"{pct(worst['sigma'])}, one SD; {ENGINE_NAMES[worst['engine']]}, {size(worst['model'])}, "
                            f"batch {worst['batch']}). HTTP, streaming, detokenising and the server's own scheduling "
                            "together." + missing,
                "cells": cited(st)}
    return {"section": "stack", "claim": f"The server stack adds up to {pct(worst['ratio'] - 1)} a step."
            if worst["ratio"] > 1 else f"The server stack's step differs by up to {pct(worst['ratio'] - 1)}.",
            "cells": cited(st),
            "evidence": f"Against the engine API, beyond noise in {of(len(real), len(st))} measured at both "
                        f"({scope}; {names(sorted({s['engine'] for s in real}, key=list(HEADLINE).index))}); the "
                        f"largest at {ENGINE_NAMES[worst['engine']]}, {size(worst['model'])}, batch {worst['batch']}."
                        + missing}


def host_principle(t):
    """The fitted host cost, named for the engines it was fitted for (anatomy.fit: enough cells with a host stall of
    STALL_MIN or more), with its bootstrap interval; a per-sequence cost only where its interval excludes zero; and the
    engines without a fit, with the most their host can cost (it never held the GPU up)."""
    # a fit whose interval could not be identified (fit.intervals: too few models, no noise to offset them by) is a
    # pattern, never a principle
    fits = {e: f for e, f in t["fits"].items()
            if f.get("h0_ms") is not None and not f.get("unphysical") and f.get("h0_interval")}
    if not fits:
        return None
    engines = [e for e in HEADLINE if e in fits]
    h0 = span([fits[e]["h0_ms"] for e in engines], lambda x: f"{x:.2f}")
    per_seq = [1e3 * fits[e]["h1_ms"] for e in engines if (fits[e].get("h1_interval") or [0])[0] > 0]
    h1 = f", plus {span(per_seq, lambda x: f'{x:.0f}')} µs a sequence" if per_seq else ""
    each = "; ".join(f"{ENGINE_NAMES[e]} {fits[e]['h0_ms']:.2f} ms a step ({fits[e]['h0_interval'][0]:.2f}–"
                     f"{fits[e]['h0_interval'][1]:.2f}, 90% interval) from {fits[e]['n_fitted']} cells, "
                     f"{fits[e]['n_host_cells']} with a host stall of {100 * fit.STALL_MIN:.0f}% or more"
                     for e in engines if fits[e].get("h0_interval"))
    errs = [fits[e]["lomo_median_error"] for e in engines if fits[e].get("lomo_median_error") is not None]
    bounded = [f"{ENGINE_NAMES[e]}'s under {min(f['host_upper_bound_ms'].values()):.2f} ms"
               for e, f in t["fits"].items() if f.get("h0_ms") is None and f.get("host_upper_bound_ms")]
    return {"section": "calculator", "claim": f"{names(engines)}'s host costs {h0} ms a step{h1}.",
            "cells": ids(r for r in t["head"] if r["engine"] in engines),
            "evidence": f"{each}" + (f"; median error {span(errs)} on models left out of the fit" if errs else "")
                        + (f". Too few host-bound cells to fit the others, only to bound them: {'; '.join(bounded)} a "
                           "step" if bounded else "") + ". The scenario on other GPUs starts from it."}


def principles(t):
    return [p for p in (crossover_principle(t), anatomy_principle(t), engines_principle(t), mitigations_principle(t),
                        stack_principle(t), host_principle(t)) if p]


# patterns: every difference, one sentence each


def crossover_patterns(t):
    """Per engine, every headline model's crossover, runs of models with the same one written as a range: a model
    whose accepted cells never stay at 90% says so, and one with no accepted cell at all says it was not measured."""
    by = {}
    for c in t["crossovers"]:
        text = (f"from batch {c['crossover_text']}" if c["crossover_text"] else "not within the batches measured"
                if c["measured"] else f"not measured ({c['rejected']} cells rejected)")
        by.setdefault(c["engine"], []).append((c["model"], text))
    out = []
    for engine in [e for e in HEADLINE if e in by]:
        runs = []
        for model, text in by[engine]:
            if runs and runs[-1][2] == text:
                runs[-1][1] = model
            else:
                runs.append([model, model, text])
        out.append({"kind": "crossover", "engine": engine, "effect": 0.0, "cells": [],
                    "text": f"{ENGINE_NAMES[engine]} keeps the GPU 90% busy: "
                            + "; ".join(f"{sizes([a, b] if a != b else [a])} {text}" for a, b, text in runs) + "."})
    return out


def engine_patterns(t):
    """Per model, batch and level with a difference beyond noise: the fastest engine, each other's gap to it, and the
    engines that tried the cell without an accepted measurement, so an absence is never silent."""
    out = []
    by = {}
    for g in t["gaps"]:
        by.setdefault((g["model"], g["batch"], g["level"]), []).append(g)
    tried = {}
    for r in defaults(t["rows"], only_accepted=False):
        tried.setdefault((r["model"], r["batch"], r["level"]), set()).add(r["engine"])
    for (model, batch, level), gs in by.items():
        slower = [g for g in gs if g["beyond"]]
        if not slower:
            continue
        best = next(g for g in gs if g["sigma"] is None)
        rejected = [e for e in HEADLINE if e in tried.get((model, batch, level), set()) - {g["engine"] for g in gs}]
        said = [f"{ENGINE_NAMES[g['engine']]} within noise of it" for g in gs if g["beyond"] is False] + \
               [f"{ENGINE_NAMES[g['engine']]}'s {longer(g['ratio'], g['sigma'])}" for g in slower] + \
               [f"{ENGINE_NAMES[e]}'s cell rejected" for e in rejected]
        out.append({"kind": "engines", "model": model, "batch": batch, "level": level,
                    "effect": max(g["ratio"] for g in slower) - 1, "cells": [c for g in slower for c in g["cells"]],
                    "text": f"{size(model)}, batch {batch}, {LEVEL_NAMES[level]}: {ENGINE_NAMES[best['engine']]} has "
                            f"the shortest step, {best['step_ms']:.2f} ms; {'; '.join(said)}."})
    return out


def anatomy_patterns(t):
    """Where the step goes, for each engine's smallest and largest model at batch 1 (derive.parts): a host stall inside
    the occupancy's own noise reads as within noise."""
    out = []
    for engine in HEADLINE:
        cells = sorted(((r, t["parts"][row_cell_id(r)]) for r in t["head"] if r["engine"] == engine
                        and r["batch"] == 1 and row_cell_id(r) in t["parts"]), key=lambda x: x[0]["params"])
        for r, p in cells[::max(1, len(cells) - 1)]:
            said = [("the bandwidth floor", pct(p["floor"])), ("kernel excess", span(p["excess"])),
                    ("launching kernels", span(p["launch"])), ("gaps inside graphs", pct(p["gaps"])),
                    *([("copies", pct(p["copy"]))] if p["copy"] >= 0.005 else []),
                    ("host stall", pct(p["stall"]) if p["stall_resolved"] else "within noise")]
            largest = (f"; beyond the floor, {p['largest'][0]} is the largest part" if len(p["largest"]) == 1 else
                       f"; beyond the floor, {' or '.join(p['largest'])} is the largest, depending on what a graph "
                       "node costs")
            out.append({"kind": "anatomy", "engine": engine, "model": r["model"], "batch": 1,
                        "effect": max(p["excess"] + p["launch"] + (p["gaps"], p["copy"], p["stall"])),
                        "cells": [row_cell_id(r)],
                        "text": f"{ENGINE_NAMES[engine]}, {size(r['model'])}, batch 1, {r['step_ms']:.2f} ms: "
                                + ", ".join(f"{name} {v}" for name, v in said) + largest + "."})
    return out


def mitigation_patterns(t):
    out = []
    for m in t["mitigations"]:
        if not m["beyond"]:
            continue
        occ = (f"; the GPU busy {pct(m['occupied_off'])} of it, against {pct(m['occupied_on'])}"
               if m["occupied_on"] is not None and m["occupied_off"] is not None else "")
        out.append({"kind": "mitigations", "engine": m["engine"], "model": m["model"], "batch": m["batch"],
                    "effect": m["ratio"] - 1, "cells": m["cells"],
                    "text": f"{ENGINE_NAMES[m['engine']]}, {size(m['model'])}, batch {m['batch']}, "
                            f"{ARMS.get(m['arm'], (None, 'with ' + m['arm']))[1]}: the step is "
                            f"{longer(m['ratio'], m['sigma'])}{occ}."})
    return out


def stack_patterns(t):
    out = []
    for engine in HEADLINE:
        for level in ("L0", "L2"):
            ss = [s for s in t["stack"] if s["engine"] == engine and s["level"] == level]
            if not ss:
                continue
            real = [s for s in ss if s["beyond"]]
            worst = max(ss, key=lambda s: abs(s["ratio"] - 1))
            out.append({"kind": "stack", "engine": engine, "effect": worst["ratio"] - 1,
                        "cells": [c for s in real for c in s["cells"]],
                        "text": f"{ENGINE_NAMES[engine]}, {LEVEL_NAMES[level]} against "
                                f"{LEVEL_NAMES[ss[0]['headline_level']]}: "
                                + (f"beyond noise in {of(len(real), len(ss))}" if real else
                                   f"within noise in {of(len(ss), len(ss))}")
                                + f"; the largest at {size(worst['model'])}, batch {worst['batch']}: "
                                f"{longer(worst['ratio'], worst['sigma'])}."})
    return out


def attribution_patterns(t):
    """Per arm that changes only the host's schedule (async scheduling, the overlap scheduler off), whether the GPU's
    busy time per step held while the step changed: what lets the step's change be read as host time. A cell where
    it moved says the arm changes more than the schedule there."""
    out = []
    for m in (m for m in t["mitigations"] if m.get("gpu_work_held") is not None):
        moved = m["busy_off"] / m["busy_on"] - 1
        out.append({"kind": "attribution", "engine": m["engine"], "model": m["model"], "batch": m["batch"],
                    "effect": abs(moved), "cells": m["cells"],
                    "text": f"{ENGINE_NAMES[m['engine']]}, {size(m['model'])}, batch {m['batch']}, "
                            f"{lower(m['arm_label'])}: the GPU's busy time per step "
                            + (f"held ({pct(moved)}) while the step changed by {pct(m['ratio'] - 1)}, so the change "
                               "is the host's" if m["gpu_work_held"] else
                               f"moved {pct(moved)}: the arm changes more than the host's schedule here") + "."})
    return out


def host_patterns(t):
    out = []
    for engine, f in t["fits"].items():
        if f.get("h0_ms") is None:
            continue
        if f.get("unphysical"):  # no host model: said, never drawn from
            out.append({"kind": "host", "engine": engine, "effect": 0.0, "cells": [],
                        "text": f"{ENGINE_NAMES[engine]}'s cells give no physical host model: {f['unphysical']}."})
            continue
        err, n = f.get("lomo_median_error"), f.get("lomo_cells") or 0
        held = f"; median error {pct(err)} on models left out of the fit ({n} cells)" if err is not None and n else ""
        if f.get("h0_interval") and f.get("h1_interval"):
            (lo, hi), (h1lo, h1hi) = f["h0_interval"], f["h1_interval"]
            band = (f" ({lo:.2f}–{hi:.2f})", f" ({1e3 * h1lo:.1f}–{1e3 * h1hi:.1f}; 90% intervals)")
        else:
            band = ("", " (their interval not identified: too few models to resample, and no noise to offset them)")
        out.append({"kind": "host", "engine": engine, "effect": f["h0_ms"], "cells": [],
                    "text": f"{ENGINE_NAMES[engine]}'s host spends {f['h0_ms']:.2f} ms a step{band[0]} plus "
                            f"{1e3 * f['h1_ms']:.1f} µs a sequence{band[1]}, {f['form']} with the GPU's work, fitted "
                            f"on {f['n_fitted']} cells, {f['n_host_cells']} of them host-bound{held}."})
    for engine, f in t.get("serial_fits", {}).items():  # overlap off: the host's time in every step
        if f.get("h0_ms") is None:
            continue
        band = f" ({f['h0_interval'][0]:.2f}–{f['h0_interval'][1]:.2f})" if f.get("h0_interval") else ""
        out.append({"kind": "host", "engine": engine, "effect": f["h0_ms"], "cells": [],
                    "text": f"{ENGINE_NAMES[engine]} with overlap switched off, where the host's time shows in every "
                            f"step: {f['h0_ms']:.2f} ms a step{band} plus {1e3 * f['h1_ms']:.1f} µs a "
                            f"sequence, {f['form']}, from {f['n_fitted']} cells."})
    return out


def patterns(t):
    """(every pattern, kinds in the page's order and the largest effects first within each; {comparisons made,
    expected_by_chance: how many of them would cross the line with no real difference, k})."""
    out = []
    for group in (crossover_patterns(t), anatomy_patterns(t), engine_patterns(t), mitigation_patterns(t),
                  attribution_patterns(t), stack_patterns(t), host_patterns(t)):
        out += sorted(group, key=lambda p: -abs(p["effect"]))
    # chance by family: each engine group against its fastest member, each mitigation and level pair on its own; the
    # least certain noise estimate in a family sets its degrees of freedom
    families = {}
    for g in t["gaps"]:
        if g["sigma"] is not None:
            families.setdefault((g["model"], g["batch"], g["level"]), []).append(g["df"])
    least = lambda dfs: min((d for d in dfs if d is not None), default=None)  # noqa: E731
    pairs = t["mitigations"] + t["stack"]
    chance = sum(derive.chance(len(dfs) + 1, least(dfs)) for dfs in families.values()) + \
        sum(derive.chance(2, x["df"]) for x in pairs)
    n = sum(len(dfs) for dfs in families.values()) + len(pairs)
    return out, {"comparisons": n, "expected_by_chance": chance, "k": K, "aa": t["noise"].aa()}


# summaries: one plain sentence under each chart, for the selection on screen (the page picks by its controls), so
# a reader who cannot see the chart, or skims past it, gets what it shows


def summaries(t):
    """{chart: sentence, or {control value: sentence}}: the occupancy charts, the frontier, the anatomy per batch and
    engine, the engine gaps per level and batch, the mitigations, the stack, and the map per measure and engine."""
    head = t["head"]
    out = {}
    for b in (1, 64):
        rows = [r for r in head if r["batch"] == b and r.get("occupied") is not None]
        if rows:
            lo, hi = min(rows, key=lambda r: r["occupied"]), max(rows, key=lambda r: r["occupied"])
            out[f"occ{b}"] = (f"At batch {b} the GPU is busy from {pct(lo['occupied'])} of the step "
                              f"({ENGINE_NAMES[lo['engine']]}, {size(lo['model'])}) to {pct(hi['occupied'])} "
                              f"({ENGINE_NAMES[hi['engine']]}, {size(hi['model'])}).")
    reached = [c for c in t["frontier"] if c.get("crossover_batch") is not None]
    if reached:
        by = {}
        for c in reached:
            by.setdefault(c["engine"], []).append(c)
        def reach(cs):
            lo, hi = (f(round(c["crossover_batch"]) for c in cs) for f in (min, max))
            return f"{lo}" if lo == hi else f"{lo} to {hi}"
        out["frontier"] = "The batch from which each engine keeps the GPU 90% busy: " + "; ".join(
            f"{ENGINE_NAMES[e]} {reach(cs)} ({f'all {len(cs)} models' if len(cs) > 1 else size(cs[0]['model'])})"
            for e, cs in sorted(by.items(), key=lambda kv: list(HEADLINE).index(kv[0]))) + "."
    anatomy = {}
    for r in head:
        p = t["parts"].get(row_cell_id(r))
        if p:
            anatomy.setdefault(str(r["batch"]), {}).setdefault(r["engine"], []).append((r, p))
    out["anatomy"] = {b: {e: f"At batch {b}, the floor is {span([p['floor'] for _, p in xs])} of the step "
                             + (f"across {len(xs)} models" if len(xs) > 1 else f"at {size(xs[0][0]['model'])}")
                             + "; above it, "
                             + ", ".join(sorted({p['largest'][0] for _, p in xs}))
                             + " is the largest part." for e, xs in by_e.items()}
                      for b, by_e in anatomy.items()}
    gaps = {}
    for g in t["gaps"]:
        gaps.setdefault(f"{g['level']}/{g['batch']}", []).append(g)
    out["engines"] = {k: (f"The slowest step here is {than(max(g['ratio'] for g in gs), 'the fastest')}; "
                          + (f"{sum(bool(g['beyond']) for g in gs)} of {len(gs)} gaps are beyond noise."
                             if any(g["sigma"] is not None for g in gs) else "no noise measured to judge them by."))
                      for k, gs in gaps.items()}
    if t["mitigations"]:
        top = max(t["mitigations"], key=lambda m: abs(math.log(m["ratio"])))
        out["mitigations"] = (f"{len(t['mitigations'])} cells with one mitigation changed; the largest change is "
                              f"{lower(top['arm_label'])} on {ENGINE_NAMES[top['engine']]}, {size(top['model'])}, "
                              f"batch {top['batch']}: {longer(top['ratio'])}.")
    if across := [x for x in t.get("exploratory", []) if x["pair"] == "mitigation"]:
        out["mitigations"] = (out.get("mitigations", "") + f" {len(across)} more set a mitigation's cell against "
                              "defaults measured in another session (the rows marked across sessions): the drift "
                              "between sessions is part of each of those differences, so none of them is a "
                              "principle.").strip()
    if t["stack"]:
        top = max(t["stack"], key=lambda s: abs(s["ratio"] - 1))
        level = LEVEL_NAMES[top["level"]].split(" (")[0]
        out["stack"] = (f"{len(t['stack'])} cells at another level; the largest difference from the headline level "
                        f"is {pct(top['ratio'] - 1)} ({ENGINE_NAMES[top['engine']]}, {level}, {size(top['model'])}, "
                        f"batch {top['batch']}).")
    measures = {"occupied": ("GPU busy", pct), "sol": ("speed of light", pct), "mbu": ("bandwidth used", pct),
                "step_ms": ("step time", lambda x: f"{x:.2f} ms")}
    out["map"] = {}
    gaps = Counter(r["engine"] for r in derive.headline_rows(t["rows"], only_accepted=False) if not derive.accepted(r))
    for key, (name, fmt) in measures.items():
        for e in HEADLINE:
            rows = [r for r in head if r["engine"] == e and r.get(key) is not None]
            rejected = (f" {gaps[e]} more {'was' if gaps[e] == 1 else 'were'} run and not accepted (outlined)."
                        if gaps[e] else "")
            if rows:
                lo, hi = min(rows, key=lambda r: r[key]), max(rows, key=lambda r: r[key])
                out["map"].setdefault(key, {})[e] = (
                    f"{ENGINE_NAMES[e]}'s {name} is {fmt(lo[key])} in its one cell ({size(lo['model'])}, batch "
                    f"{lo['batch']})." if len(rows) == 1 else
                    f"{ENGINE_NAMES[e]}'s {name} runs from {fmt(lo[key])} ({size(lo['model'])}, batch {lo['batch']}) "
                    f"to {fmt(hi[key])} ({size(hi['model'])}, batch {hi['batch']}) over {len(rows)} cells.") + rejected
    return out


def report(t):
    """patterns.json: what the data are (derive.about), the principles, every pattern, and the count behind them."""
    found, stats = patterns(t)
    return {"about": derive.about(t["rows"]), "principles": principles(t), "patterns": found, "stats": stats,
            "summaries": summaries(t),
            # pairs whose two cells came from different sessions: a drift term, not a control, so never a principle
            "exploratory": t.get("exploratory", [])}


def write(rep, path):
    """The one writer of patterns.json: strict JSON (no NaN), for the page's copy and results/'s alike."""
    Path(path).write_text(json.dumps(derive.finite(rep), indent=1, allow_nan=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cells", type=Path)
    p.add_argument("--json", type=Path, help="also write the report here")
    a = p.parse_args(argv)
    rows, _, noise = derive.load(a.cells)
    r = report(derive.tables(rows, noise))
    for x in r["principles"]:
        print(f"* {x['claim']}\n  {x['evidence']}")
    kind = None
    for x in r["patterns"]:
        if x["kind"] != kind:
            kind = x["kind"]
            print(f"\n{kind}")
        print(f"- {x['text']}")
    s = r["stats"]
    print(f"\n{s['comparisons']} comparisons at {K:g}x their noise: about {s['expected_by_chance']:.1f} of them would "
          "cross that line by chance.")
    if a.json:
        write(r, a.json)


if __name__ == "__main__":
    main()
