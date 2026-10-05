"""Equivalent useful work: the greedy-token probe every timing unit ends with, set engine against engine and each
arm against its defaults (agreement.csv). anatomy.tidy writes it beside the cell table."""
from statistics import median

from anatomy.metrics import HEADLINE, IDENTICAL_ARMS
from anatomy.tidy import TIMING, chosen_units
from servebench.results import read_jsonl

EARLY = 2  # an engine parting from every other within this many tokens, on most prompts, is flagged


def first_divergence(a, b):
    """The index of the first token two greedy outputs differ at (their length where one is the other's start)."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def agreement(roots):
    """agreement.csv's rows, from the timing units' greedy probes (servebench.drivers.common.probe, anatomy.served
    .greedy): the same prompts' 32 greedy tokens per engine, arm and level. One shape for every row: kind, session,
    model, a and b (engine-version/arm/level), prompts, identical, median_first_divergence, flagged (a bool), odd (the
    engine that parted, on a flagged pair).
    - identity: an arm that changes no arithmetic (IDENTICAL_ARMS) against its engine's defaults at the same level,
      model and session: flagged when any prompt's tokens differ (a finding to explain);
    - pair: two engines' defaults at their headline levels, per model and session: the median first token they part
      at. bf16 kernels may legitimately part after a few tokens; an engine that parts from every other within EARLY
      tokens on most prompts while those agree among themselves (wrong weights, config or conversion) is flagged, odd
      naming it. A model with two engines (the second family) gets its pair, never a flag: no majority;
    - probe-error: a unit whose probe failed, how many prompts (prompts) and the first error (odd)."""


    got, name, failed = {}, {}, []
    for session, d, u, _gpu, _others, _batches in chosen_units(roots):
        if u["capture"] in TIMING:
            recs = [r for r in read_jsonl(d / "driver.jsonl") if r.get("kind") == "greedy"]
            ids = {r["prompt"]: r["ids"] for r in recs if "ids" in r}
            key = (session, u["model"], u["engine"], u["arm"], u["level"])
            name[key] = f"{u['engine']}-{u['version']}/{u['arm']}/{u['level']}"
            if ids:
                got[key] = ids
            if errors := [r for r in recs if "error" in r]:  # a probe that failed says so, never an empty table
                failed.append({"kind": "probe-error", "session": session, "model": u["model"], "a": name[key],
                               "b": "", "prompts": len(errors), "identical": None, "median_first_divergence": None,
                               "flagged": True, "odd": errors[0]["error"][:200]})

    def row(kind, key_a, key_b, flagged=False, odd=""):
        x, y = got[key_a], got[key_b]
        shared = sorted(set(x) & set(y))
        parts = [first_divergence(x[i], y[i]) for i in shared]
        return {"kind": kind, "session": key_a[0], "model": key_a[1], "a": name[key_a], "b": name[key_b],
                "prompts": len(shared), "identical": sum(x[i] == y[i] for i in shared),
                "median_first_divergence": median(parts) if parts else None, "flagged": flagged, "odd": odd}

    out = []
    for key in sorted(got):
        base = (*key[:3], "defaults", key[4])
        if key[3] in IDENTICAL_ARMS and base in got:
            r = row("identity", base, key)
            out.append(r | {"flagged": r["identical"] < r["prompts"]})
    for session, model in sorted({k[:2] for k in got}):
        heads = {k[2]: k for k in got
                 if k[:2] == (session, model) and k[3] == "defaults" and k[4] == HEADLINE.get(k[2])}
        names = sorted(heads)

        def at(x, y, p, heads=heads):
            a, b = got[heads[x]], got[heads[y]]
            return first_divergence(a[p], b[p]) if p in a and p in b else 0
        odd = ""
        for e in names if len(names) > 2 else []:
            rest = [x for x in names if x != e]
            prompts = set.intersection(*(set(got[heads[x]]) for x in names))
            parted = [p for p in prompts if all(at(e, x, p) < EARLY for x in rest)
                      and all(at(x, y, p) >= EARLY for i, x in enumerate(rest) for y in rest[i + 1:])]
            if prompts and len(parted) > len(prompts) / 2:
                odd = e
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out.append(row("pair", heads[a], heads[b], odd in (a, b), odd if odd in (a, b) else ""))
    return out + failed
