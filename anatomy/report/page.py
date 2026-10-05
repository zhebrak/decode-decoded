"""The living page, decode.zhebrak.io: results/cells.parquet in, site/ out (index.html with its data inline, the data
files for download, CNAME for GitHub Pages). Without a cells table it renders the synthetic layout preview
(anatomy.report.synthetic) under a banner that says so.

    python -m anatomy.report.page --cells results/cells.parquet --out site
    python -m anatomy.report.page --out site                      # the layout preview, synthetic data
"""
import argparse
import json
import shutil
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from anatomy import fit
from anatomy.metrics import current_versions, rejection
from anatomy.report import charts, derive, patterns, synthetic
from servebench import ROOT
from servebench.results import model_name, row_cell_id

TEMPLATES = Path(__file__).parent / "templates"
# where the page sends a reader: its own address (the canonical URL and the CNAME for GitHub Pages), the code, the raw
# data and corrections. Given to build() (the CLI's --site, --repository, --dataset, --issues, --author); a link not
# given is left out of the page, and the raw-data links of the cell explorer with it
DEFAULT_LINKS = {"site": None, "repository": None, "dataset": None, "issues": None, "author": "https://zhebrak.io"}
LINKS = dict(DEFAULT_LINKS)
KEEP = ("engine", "version", "arm", "level", "model", "params", "batch", "prompt_len", "gen_len", "step_ms",
        "occupied", "kernel_ms", "mbu", "mbu_datasheet", "mfu", "sol", "floor_ms", "copy_ms", "kernel_exc_ms",
        "launch_ms", "bubbles_ms", "host_stall_ms", "consistent", "compute_leaning", "session", "valid")


def changelog(path=None):
    """The versions table: CHANGELOG.md's rows, one per measured version set and correction, newest last."""
    path = Path(path or ROOT / "CHANGELOG.md")
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return []
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    return [dict(zip(head, (c.strip() for c in ln.strip("|").split("|")))) for ln in lines[2:]]


def predictions(rows, fits):
    """The calculator's rows, computed here by anatomy.fit.predict (the page script only filters them): per engine
    with a host model, per card, every accepted cell with kernel time, its predicted step and occupancy."""
    head = derive.headline_rows(rows)
    out = []
    for engine, f in fits.items():
        if f.get("h0_ms") is None:
            continue
        cells = [r for r in head if r["engine"] == engine]
        for card, bw in charts.CARDS.items():
            for p in fit.predict(cells, f, bw / charts.CARDS["H100 SXM"]):
                out.append({**p, "card": card, "params": next(r["params"] for r in cells if r["model"] == p["model"])})
    return out


def page_data(t, report, about, trace=None, stretch=None):
    """What the page script reads, and nothing more (the full rows would be the largest payload, and mostly unread:
    the whole table is the cells.csv download, every pattern patterns.json). t: derive.tables; report:
    patterns.report(t); about: derive.about(rows), what the data is (its status, cards, models), the one carrier of
    it."""
    rows = t["rows"]
    v = charts.views(t, keep=lambda r: {k: r.get(k) for k in KEEP})
    return {"about": about, "engine_names": charts.ENGINE_NAMES,
            "headline": [{k: r.get(k) for k in KEEP} for r in t["head"]], "occupancy": v["occupancy"],
            "map_gaps": map_gaps(rows),
            "frontier": v["frontier"], "engine_gaps": v["gaps"], "anatomy": v["anatomy"],
            "anatomy_left_out": v["left_out"], "stack": t["stack"],
            "mitigations": t["mitigations"] + [x for x in t["exploratory"] if x["pair"] == "mitigation"],
            "predictions": predictions(rows, t["fits"]), "fits": t["fits"], "cards": charts.CARDS,
            "principles": report["principles"], "pattern_stats": report["stats"], "summaries": report["summaries"],
            "evidence": evidence(rows, report["principles"]), "explorer": explorer(rows),
            "trace": synthetic.trace_strip() if about["dataset_status"] == "preview" else (trace or []),
            "trace_stretch": stretch,
            "specs": charts.all_specs(),
            "engine_colours": {t: {e: pal[slot] for e, slot in charts.ENGINES.items()}
                               for t, pal in charts.PALETTES.items()},
            "changelog": changelog()}


CHECKS = (("estimators_agree", "The step's independent checks agree"),
          ("trace_consistent", "The trace fits the unprofiled step"),
          ("anchor_ok", "The graph trace's step anchors mark the steps"),
          ("node_anchor_ok", "The node trace's step anchors mark the steps"),
          ("busy_invariant", "Busy time held with the host's cores contended"),
          ("consistent", "The five parts add up within 3%"))


def outcome(r, key):
    """One check's outcome in words: passed, failed, not run, or, for busy invariance whose hostload capture ran but
    stretched the step less than tracing does, ran and tested nothing."""
    v = r.get(key)
    if v is None and key == "busy_invariant" and "hostload" in (r.get("sources") or {}):
        return "ran, but the contention stretched the step too little to test it"
    return {True: "passed", False: "failed"}.get(v, "not run")


def map_gaps(rows):
    """The map's cells that were run and not accepted (a failed or skipped batch, a check that failed), each with its
    reason: the map shows the grid's whole coverage, never only what passed."""
    return [{"engine": r["engine"], "model": r["model"], "params": r.get("params"), "batch": r["batch"], "gap": True,
             "rejection": r.get("rejection") or rejection(r) or "not accepted"}
            for r in derive.headline_rows(rows, only_accepted=False) if not derive.accepted(r)]


def raw_links(r):
    """{capture: the dataset folder of its raw files} for one row, from its own sources (session/unit/capture) under
    its version set; {} where the row does not say its version set (sessions before it was recorded)."""
    if not r.get("version_set") or not LINKS.get("dataset"):
        return {}
    return {c: f"{LINKS['dataset']}/tree/main/{r['version_set']}/{src}" for c, src in (r.get("sources") or {}).items()}


def explorer(rows):
    """The cell explorer's rows: every defaults cell at its engine's headline level (accepted or not, the reason
    with it), what the page shows of one cell: its step and busy share, the five parts, kernel time by class, each
    check passed, failed or not run, its captures' dataset folders, and the grid command that measures it again."""
    out = []
    for r in derive.headline_rows(rows, only_accepted=False):
        model = model_name(r["model"])
        out.append({"id": row_cell_id(r), "engine": r["engine"], "model": model, "batch": r["batch"],
                    "accepted": derive.accepted(r), "rejection": r.get("rejection") or "",
                    "step_ms": r.get("step_ms"), "occupied": r.get("occupied"),
                    "parts": {t: r.get(f"{t}_ms") for t in ("floor", "copy", "kernel_exc", "launch", "host_stall")},
                    "classes_ms": r.get("classes_ms") or {},
                    "checks": [[label, outcome(r, k)] for k, label in CHECKS],
                    "sources": {c: raw_links(r).get(c) or src for c, src in (r.get("sources") or {}).items()},
                    "measure": f"scripts/box.sh grid <session> S2 --block main --engines {r['engine']} --models "
                               f"{r['model']}  # every batch of {model} on {charts.ENGINE_NAMES[r['engine']]}"})
    return out


def evidence(rows, principles):
    """{cell id: the dataset folder holding its captures}, for every cell a principle cites; None where the row does
    not say its version set."""
    cited = {c for p in principles for c in p.get("cells") or []}
    out = {}
    for r in rows:
        if row_cell_id(r) in cited:
            link = next(iter(raw_links(r).values()), None)
            out[row_cell_id(r)] = link.rsplit("/", 1)[0] if link else None
    return out


def lede(about):
    """The page's scope in a sentence, from what the rows hold: the engines, the cards, the models and the batches."""
    cards = ", ".join(c.replace("NVIDIA ", "").replace("H100 80GB HBM3", "H100 SXM") for c in about["cards"]) or \
        "H100 SXM"
    models, batches = about["models"], about["batches"]
    family = f"{models[0]}" if len(models) == 1 else f"{models[0]} to {models[-1]}"
    batch = f"batch {batches[0]}" if len(batches) == 1 else f"batch {batches[0]} to {batches[-1]}"
    count = {1: "One", 2: "Two", 3: "Three", 4: "Four"}.get(len(about["engines"]), str(len(about["engines"])))
    return f"{count} serving engines on {'one ' if len(about['cards']) <= 1 else ''}{cards}, {family}, {batch}"


def card(t, about, out):
    """The social card (Open Graph, Twitter): the smallest model's anatomy by batch as a PNG, its status drawn in when
    the data are not the study's (static.render), at data/card.png; None where the page has no address or vl-convert
    is not installed (the report extra)."""
    if not LINKS.get("site"):
        return None
    try:
        from anatomy.report import static
    except ImportError:
        return None
    try:
        made = static.render(t, Path(out) / "data", themes=("light",), png=("anatomy-batch",), only=("anatomy-batch",))
    except ImportError:
        return None
    png = next((p for p in made if p.suffix == ".png"), None)
    for p in made:  # the page keeps the card only; the post's figures are make figures'
        if p != png:
            p.unlink()
    if png is None:
        return None
    png.rename(png.with_name("card.png"))
    return LINKS["site"] + "data/card.png"


def build(cells, out, links=None):
    LINKS.clear()
    LINKS.update({**DEFAULT_LINKS, **{k: v for k, v in (links or {}).items() if v}})
    out = Path(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    if cells:
        rows, (trace, stretch), noise = derive.load(cells)
        agreed = derive.agreement(Path(cells).with_name("agreement.csv"))
    else:
        rows, trace, stretch, noise, agreed = synthetic.cells(), [], None, derive.Noise(), None
    t = derive.tables(rows, noise)
    report = patterns.report(t)
    about = report["about"]
    if about["dataset_status"] == "study" and (problems := patterns.check(report, rows)):
        # a study page claims what its cells show: a principle without an accepted cell behind it never goes out
        raise SystemExit("the principles do not rest on accepted cells: " + "; ".join(problems))
    data = derive.finite({**page_data(t, report, about, trace, stretch), "agreement": agreed,
                          "cell_strips": bool(cells and Path(cells).with_name("cell-strips.json").exists())})
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    order = list(charts.ENGINES)
    table = sorted(t["head"], key=lambda r: (order.index(r["engine"]), r["params"], r["batch"]))
    html = env.get_template("page.html").render(data=data, table=table, engine_names=charts.ENGINE_NAMES,
                                                versions=current_versions(rows), lede=lede(about), links=LINKS,
                                                card=card(t, about, out))
    (out / "index.html").write_text(html)
    (out / "data" / "page.json").write_text(json.dumps(data, default=str, allow_nan=False))
    patterns.write(report, out / "data" / "patterns.json")
    strips_file = Path(cells).with_name("cell-strips.json") if cells else None
    if strips_file and strips_file.exists():  # the cell explorer's traces, fetched when a reader opens a cell
        shutil.copyfile(strips_file, out / "data" / "cell-strips.json")
    if cells and Path(cells).with_name("agreement.csv").exists():  # the greedy-token check, downloadable beside
        shutil.copyfile(Path(cells).with_name("agreement.csv"), out / "data" / "agreement.csv")
    if cells and Path(cells).with_suffix(".csv").exists():  # anatomy.tidy's own cells.csv: one writer of the format
        shutil.copyfile(Path(cells).with_suffix(".csv"), out / "data" / "cells.csv")
    else:
        import pandas as pd

        pd.DataFrame([{**r, "dataset_status": "preview"} for r in rows]).to_csv(out / "data" / "cells.csv", index=False)
    if LINKS["site"]:  # GitHub Pages reads the custom domain from this file
        from urllib.parse import urlparse

        (out / "CNAME").write_text(urlparse(LINKS["site"]).netloc + "\n")
    return out / "index.html"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path)
    p.add_argument("--out", type=Path, default=Path("site"))
    for k, default in DEFAULT_LINKS.items():
        p.add_argument(f"--{k}", help=f"the page's {k} link" + (f" (default {default})" if default else ""))
    a = p.parse_args(argv)
    print(build(a.cells, a.out, {k: getattr(a, k) for k in DEFAULT_LINKS}))


if __name__ == "__main__":
    main()
