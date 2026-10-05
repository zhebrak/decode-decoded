"""The post's four figures, rendered from the cell table with vl-convert in the page's palette and typeface
(anatomy.report.charts; Instrument Sans, OFL, vendored in fonts/), light and dark SVG each and a PNG of the light one:

    anatomy-batch    the smallest model's decode step share by batch, one stacked bar per engine
    anatomy-models   the step share by model size, two bars per engine (batch 1, then 32 or the largest the card holds)
    engines          each engine's step over the fastest engine's by model size, batch 1 solid and 16 dashed
    mitigations      each optimisation switched off: the step time ratio and the GPU busy share, by model and batch

    python -m anatomy.report.static --cells results/cells.parquet --out figures
    python -m anatomy.report.static --out /tmp/figures                      # the synthetic layout preview
    python -m anatomy.report.static ... --font "'Google Sans Code', monospace" --font-dir <dir with its .ttf files>
"""
import argparse
from pathlib import Path

from anatomy.report import charts, derive, synthetic
from anatomy.report.charts import ENGINE_NAMES, ENGINES, PARAM_AXIS, RATIO_AXIS, TERM_NAMES, TERMS, colour, q

FONTS = Path(__file__).parent / "fonts"
REGULAR = {1, 2, 4, 8, 16, 32, 64, 128, 256}  # the post's batches: the grid's "max" cells (14B at 211, the KV limit)
# stay on the page, where the explorer says what they are, and off the figures, where 211 on an axis costs a caption
INITIAL = {"vllm": "V", "sglang": "S", "trtllm": "T", "llamacpp": "L"}
INITIALS = " : ".join(f"datum.value === '{k}' ? '{v}'" for k, v in INITIAL.items()) + " : datum.value"
GROUPS = {"graphs-off": "CUDA graphs off", "async-off": "Scheduler off", "overlap-off": "Scheduler off"}
ROWS = [  # the mitigations figure's rows: (group, field, title)
    ("CUDA graphs off", "ratio", "CUDA graphs off: step time with them off, over the step time with everything on"),
    ("CUDA graphs off", "occupied_off", "CUDA graphs off: share of the step the GPU is busy"),
    ("Scheduler off", "ratio", ["Scheduler off: step time with it off, over the step time with everything on",
                                "(vLLM: async scheduling; SGLang and TensorRT-LLM: overlap scheduler)"]),
]
Y = {("CUDA graphs off", "ratio"): {"scale": {"type": "log", "domain": [0.9, 10], "nice": False, "zero": False},
                                    "axis": {**RATIO_AXIS, "values": [1, 2, 4, 8]}, "title": "Step time, off / on"},
     ("Scheduler off", "ratio"): {"scale": {"type": "log", "domain": [0.95, 1.7], "nice": False, "zero": False},
                                  "axis": {**RATIO_AXIS, "values": [1, 1.2, 1.4, 1.6]}, "title": "Step time, off / on"},
     "occupied_off": {"scale": {"domain": [0, 1]}, "axis": {"format": "%", "values": [0, 0.25, 0.5, 0.75, 1]},
                      "title": "GPU busy"}}


def regular(rows):
    """The rows at the post's batches; a row that names no batch stays."""
    return [r for r in rows if r.get("batch") is None or r["batch"] in REGULAR]


def short(model):
    return str(model).split("/")[-1]


def share(r, term):
    """A part's share of the step: the table's own share where it has one, else its milliseconds over the step."""
    s = r.get(f"{term}_share")
    if s is None and r.get(f"{term}_ms") is not None and r.get("step_ms"):
        s = r[f"{term}_ms"] / r["step_ms"]
    return max(float(s or 0.0), 0.0)


def anatomy_rows(t):
    """The cells with a decomposition (the page's own set, charts.views), at the regular batches, one record per
    part."""
    out = []
    for r in regular(charts.views(t)["anatomy"]):
        for i, term in enumerate(TERMS):
            out.append({"engine": r["engine"], "initial": INITIAL.get(r["engine"], r["engine"][:1].upper()),
                        "model": short(r["model"]), "params": r.get("params") or 0, "batch": int(r["batch"]),
                        "term": term, "term_name": TERM_NAMES[term], "term_order": i, "share": share(r, term),
                        "step_ms": r.get("step_ms")})
    return out


def models_of(rows):
    """The models in the rows, smallest first."""
    return [m for _, m in sorted({(r["params"], r["model"]) for r in rows})]


def figures(t):
    """{name: rows}: the four figures' data, from derive.tables through the page's own views."""
    parts = anatomy_rows(t)
    models = models_of(parts)
    out = {}
    if models:
        out["anatomy-batch"] = [r for r in parts if r["model"] == models[0]]
        pairs = []
        for m in models:
            batches = sorted({r["batch"] for r in parts if r["model"] == m})
            second = max((b for b in batches if b <= 32), default=None)
            if 1 in batches and second and second != 1:
                pairs += [{**r, "batch_slot": (1, second).index(r["batch"])}
                          for r in parts if r["model"] == m and r["batch"] in (1, second)]
        out["anatomy-models"] = pairs
    gaps = [g for g in charts.views(t)["gaps"] if g["level"] == "L1" and g["batch"] in (1, 16)]
    out["engines"] = [{k: g.get(k) for k in ("engine", "model", "batch", "params", "ratio", "beyond", "step_ms",
                                              "fastest")} for g in gaps]
    out["mitigations"] = [{"engine": r["engine"], "arm": r["arm"], "arm_label": r.get("arm_label", r["arm"]),
                           "group": GROUPS[r["arm"]], "model": short(r["model"]), "params": r.get("params") or 0,
                           "batch": r["batch"], "ratio": r["ratio"], "beyond": r.get("beyond"),
                           "occupied_on": r.get("occupied_on"), "occupied_off": r.get("occupied_off")}
                          for r in regular(t["mitigations"]) if not r.get("exploratory") and r["arm"] in GROUPS]
    return {k: v for k, v in out.items() if v}


def title(text, subtitle=None, p=None):
    out = {"text": text, "offset": 10}
    if subtitle:
        out |= {"subtitle": subtitle, "subtitleColor": p["secondary"], "subtitleFontSize": 12,
                "subtitleFont": charts.FONT}
    return out


def anatomy_batch_spec(p, data):
    model = data[0]["model"]
    return {
        "$schema": charts.SCHEMA, "config": charts.config(p), "data": {"values": data},
        "title": title(f"{model}: decode step share, by batch",
                       "share of the step; in each group V vLLM, S SGLang, T TensorRT-LLM", p),
        "facet": {"column": {"field": "batch", "type": "ordinal", "sort": sorted(REGULAR),
                             "header": {"title": "Batch", "labelOrient": "bottom", "titleOrient": "bottom",
                                        "labelColor": p["text"], "titleColor": p["secondary"], "labelFontSize": 12,
                                        "titleFontSize": 12, "labelPadding": 4}}},
        "spacing": 6,
        "spec": {"width": 62, "height": 280,
                 "mark": {"type": "bar", "width": {"band": 0.82}, "stroke": p["surface"], "strokeWidth": 1.5},
                 "encoding": {
                     "x": {"field": "engine", "type": "nominal", "sort": list(ENGINES), "title": None,
                           "axis": {"labelExpr": INITIALS, "labelAngle": 0, "domain": False, "ticks": False,
                                    "labelColor": p["secondary"], "labelPadding": 3}},
                     "y": {"field": "share", "type": "quantitative", "stack": "normalize", "title": "Share of the step",
                           "axis": {"format": "%", "tickCount": 5, "grid": True}},
                     "color": colour(p, "term", TERMS, TERM_NAMES, title=None, legend=True),
                     "order": {"field": "term_order"},
                     "tooltip": [{"field": "engine"}, {"field": "batch"}, {"field": "term_name", "title": "Part"},
                                 {"field": "share", "format": ".0%"}, {"field": "step_ms", "format": ".3f"}]}},
        "resolve": {"axis": {"y": "shared"}},
    }


def anatomy_models_spec(p, data):
    names = [m for _, m in sorted({(r["params"], r["model"]) for r in data})]
    seconds = sorted({r["batch"] for r in data if r["batch_slot"] == 1}, reverse=True)
    odd = sorted({r["model"] for r in data if r["batch_slot"] == 1 and r["batch"] != seconds[0]})
    note = f" ({', '.join(odd)}: the largest batch the card holds)" if odd else ""
    return {
        "$schema": charts.SCHEMA, "config": charts.config(p), "data": {"values": data},
        "title": title(f"Decode step share, by model size, at batch 1 and {seconds[0]}",
                       [f"share of the step; in each group V vLLM, S SGLang, T TensorRT-LLM, each at batch 1 then "
                        f"{seconds[0]}", note.strip()] if note else
                       f"share of the step; in each group V vLLM, S SGLang, T TensorRT-LLM, each at batch 1 then "
                       f"{seconds[0]}", p),
        "facet": {"column": {"field": "model", "type": "nominal", "sort": names,
                             "header": {"title": None, "labelOrient": "bottom", "labelColor": p["text"],
                                        "labelFontSize": 12, "labelPadding": 6}}},
        "spacing": 10,
        "spec": {"width": 96, "height": 280, "layer": [
            {"mark": {"type": "bar", "stroke": p["surface"], "strokeWidth": 1.5},
             "encoding": {
                 "x": {"field": "engine", "type": "nominal", "sort": list(ENGINES), "title": None,
                       "axis": {"labelExpr": INITIALS, "labelAngle": 0, "domain": False, "ticks": False,
                                "labelColor": p["secondary"], "labelPadding": 18}},
                 "xOffset": {"field": "batch_slot", "type": "ordinal"},
                 "y": {"field": "share", "type": "quantitative", "stack": "normalize", "title": "Share of the step",
                       "axis": {"format": "%", "tickCount": 5, "grid": True}},
                 "color": colour(p, "term", TERMS, TERM_NAMES, title=None, legend=True),
                 "order": {"field": "term_order"},
                 "tooltip": [{"field": "engine"}, {"field": "model"}, {"field": "batch"},
                             {"field": "term_name", "title": "Part"}, {"field": "share", "format": ".0%"},
                             {"field": "step_ms", "format": ".3f"}]}},
            {"transform": [{"filter": "datum.term === 'floor'"}],  # the batch under each bar
             "mark": {"type": "text", "baseline": "top", "dy": 4, "fontSize": 10, "color": p["muted"]},
             "encoding": {"x": {"field": "engine", "type": "nominal", "sort": list(ENGINES)},
                          "xOffset": {"field": "batch_slot", "type": "ordinal"},
                          "y": {"datum": 0}, "text": {"field": "batch"}}}]},
        "resolve": {"axis": {"y": "shared"}},
    }


def engines_spec(p, data):
    top = max(d["ratio"] for d in data)
    x = q("params", "Model", scale={"type": "log", "nice": False, "padding": 24}, axis=PARAM_AXIS)
    y = q("ratio", "Step, over the fastest engine's", scale={"domain": [0.97, top * 1.02], "zero": False,
                                                            "nice": False}, axis=RATIO_AXIS)
    col = colour(p, "engine", ENGINES, ENGINE_NAMES, title=None)
    dash = {"field": "batch", "type": "ordinal", "title": None,
            "scale": {"domain": [1, 16], "range": [[1, 0], [6, 4]]},
            "legend": {"labelExpr": "'Batch ' + datum.value", "symbolType": "M-2,0H2", "symbolSize": 100,
                       "symbolStrokeColor": p["secondary"], "symbolStrokeWidth": 2}}
    tip = [{"field": "engine_name", "title": "Engine"}, {"field": "batch", "title": "Batch"},
           {"field": "model_name", "title": "Model"}, {"field": "ratio", "title": "Over the fastest", "format": ".3f"},
           {"field": "step_ms", "title": "Step (ms)", "format": ".3f"}]
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"y": {"datum": 1}}}
    line = {"mark": {"type": "line", "strokeWidth": 2}, "encoding": {"x": x, "y": y, "color": col, "strokeDash": dash}}
    return {
        "$schema": charts.SCHEMA,
        "config": {**charts.config(p), "legend": {**charts.config(p)["legend"], "layout": {"top": {"margin": 28}}}},
        "data": {"values": data}, "width": 720, "height": 360,
        "title": title("Decode step over the fastest engine's, by model size, at batch 1 and 16"),
        "transform": [charts.engine_label(), charts.model_label()],
        "layer": [rule, line, *charts.points(p, x, y, {}, tip, col, size=60)],
    }


def mitigations_row(p, data, group, field, text, first, last):
    """One row of the mitigations figure: a column per model, batch along x, `field` along y on a scale shared across
    the row, a line per engine; hollow points within noise on the ratio rows."""
    models = [m for _, m in sorted({(r["params"], r["model"]) for r in data})]
    x = q("batch", "Batch" if last else None, scale={"type": "log", "base": 2, "domain": [0.8, 320], "nice": False},
          axis={"values": [1, 8, 64, 256], "labelExpr": "datum.value", "grid": True})
    yc = Y.get((group, field)) or Y[field]
    y = q(field, yc["title"], scale=yc["scale"], axis=yc["axis"])
    col = colour(p, "engine", ENGINES, ENGINE_NAMES, title=None, legend=first)
    tip = [{"field": "engine_name", "title": "Engine"}, {"field": "arm_label", "title": "Changed"},
           {"field": "model", "title": "Model"}, {"field": "batch", "title": "Batch"},
           {"field": "ratio", "title": "Step time, off / on", "format": ".2f"},
           {"field": "occupied_on", "title": "GPU busy, all on", "format": ".0%"},
           {"field": "occupied_off", "title": "GPU busy, off", "format": ".0%"}]
    line = {"mark": {"type": "line", "strokeWidth": 2}, "encoding": {"x": x, "y": y, "color": col}}
    if field == "ratio":
        rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"y": {"datum": 1}}}
        layers = [rule, line, *charts.points(p, x, y, {}, tip, col, size=56)]
    else:
        layers = [line, {"mark": {"type": "point", "filled": True, "size": 56, "opacity": 1, "strokeWidth": 2,
                                  "stroke": p["surface"]}, "encoding": {"x": x, "y": y, "color": col, "tooltip": tip}}]
    header = {"title": None, "labelColor": p["text"], "labelFontSize": 13, "labelPadding": 6, "labels": first}
    return {"title": {"text": text, "anchor": "start", "fontSize": 13, "fontWeight": 600, "color": p["text"],
                      "offset": 6},
            "transform": [{"filter": f"datum.group === '{group}'"}],
            "facet": {"column": {"field": "model", "type": "nominal", "sort": models, "header": header}},
            "spacing": 18, "spec": {"width": 200, "height": 130, "layer": layers},
            "resolve": {"scale": {"y": "shared"}, "axis": {"y": "shared"}}}


def mitigations_spec(p, data):
    rows = [(g, f, t) for g, f, t in ROWS if any(r["group"] == g and r.get(f) is not None for r in data)]
    return {
        "$schema": charts.SCHEMA, "config": charts.config(p), "data": {"values": data},
        "title": title("Engine optimisations, switched off one at a time"),
        "transform": [charts.engine_label()],
        "vconcat": [mitigations_row(p, data, g, f, t, i == 0, i == len(rows) - 1) for i, (g, f, t) in enumerate(rows)],
        "spacing": 26, "resolve": {"scale": {"color": "shared"}},
    }


SPECS = {"anatomy-batch": anatomy_batch_spec, "anatomy-models": anatomy_models_spec, "engines": engines_spec,
         "mitigations": mitigations_spec}


def mark(about):
    """The line a figure carries when its data are not the study's, drawn into the image so a figure lifted out of the
    post still says what it is; None for study data."""
    if about["dataset_status"] == "preview":
        return "Layout preview: a toy model's numbers, not measurements"
    if about["dataset_status"] == "validation":
        cards = ", ".join(c.replace("NVIDIA ", "") for c in about["cards"])
        return f"Validation run ({cards}; {', '.join(about['models'])}): not the study"
    return None


def for_print(spec, data, theme):
    """A legend of engines lists only the engines the figure's rows hold: the page's fixed palette names every engine
    the harness knows, llama.cpp included, which is not in the study."""
    present = sorted({r["engine"] for r in data if isinstance(r, dict) and r.get("engine")},
                     key=list(ENGINES).index)

    def walk(node):
        if isinstance(node, dict):
            enc = node.get("encoding")
            if isinstance(enc, dict):
                c = enc.get("color")
                if isinstance(c, dict) and c.get("field") == "engine" and isinstance(c.get("legend"), dict) and present:
                    c["legend"]["values"] = present
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(spec)
    return spec


def spec_for(name, data, theme, status=None):
    """The figure's spec for one theme: its data inline, a transparent background (the page or the post supplies it),
    the engine legend trimmed to the engines present, and the status line as a subtitle where the data are not the
    study's."""
    p = charts.PALETTES[theme]
    s = SPECS[name](p, data)
    for_print(s, data, theme)
    s["config"]["background"] = "transparent"
    if status:
        sub = s["title"].get("subtitle")
        s["title"] |= {"subtitle": [status] + ([sub] if isinstance(sub, str) else list(sub or [])),
                       "subtitleColor": p["secondary"], "subtitleFontSize": 12, "subtitleFont": charts.FONT}
    return s


def render(t, out, themes=("light", "dark"), png=tuple(SPECS), only=None, font=None, font_dir=None):
    """Write <name>-<theme>.svg for every figure, and <name>.png of the light one for the names in `png`."""
    import vl_convert as vlc

    vlc.register_font_directory(str(font_dir or FONTS))
    if font:
        charts.FONT = font
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status = mark(derive.about(t["rows"]))
    written = []
    for name, data in figures(t).items():
        if only and name not in only:
            continue
        for theme in themes:
            spec = spec_for(name, data, theme, status)
            path = out / f"{name}-{theme}.svg"
            path.write_text(vlc.vegalite_to_svg(spec))
            written.append(path)
            if name in png and theme == "light":
                spec["config"]["background"] = charts.PALETTES["light"]["surface"]
                (out / f"{name}.png").write_bytes(vlc.vegalite_to_png(spec, scale=2))
                written.append(out / f"{name}.png")
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path)
    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--font", help="a CSS font-family for the figures' text (default: the page's Instrument Sans)")
    p.add_argument("--font-dir", type=Path, help="a directory of .ttf/.otf files the renderer may use for --font")
    a = p.parse_args(argv)
    if a.cells:
        rows, _, noise = derive.load(a.cells)
    else:
        rows, noise = synthetic.cells(), None
    for path in render(derive.tables(rows, noise), a.out, font=a.font, font_dir=a.font_dir):
        print(path)


if __name__ == "__main__":
    main()
