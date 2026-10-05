"""The page's charts as Vega-Lite 6 specs, built here for both themes and filled with rows by the page script (or by
anatomy.report.static for the post's SVGs), so the page and the post draw from one definition.

Colour follows the entity on every chart (the palette checked for colour-blind separation and contrast, light and
dark, adjacent pairs):
- engines, categorical slots 1-4 in fixed order: vLLM blue, SGLang orange, TRT-LLM aqua, llama.cpp yellow. Four
  series make direct labels mandatory (yellow beside orange), and light-mode aqua and yellow sit under 3:1 on the
  surface, so every engine chart carries direct labels or a legend plus the table view;
- the anatomy's terms, slots no engine uses: floor violet, copies amber, kernel excess magenta, launch green, and
  host stall the neutral grey (the GPU idle), so blue never means both "vLLM" and a term; amber sits between violet
  and magenta, where it passes the colour-blind and normal-vision checks in both themes (beside green or grey it
  failed one);
- magnitude (the map): one sequential blue ramp, light to dark, its own steps on the dark surface.
"""
import math

from servebench.physics import PEAKS
from servebench.results import row_cell_id

SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"
FONT = "'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif"

PALETTES = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4",
              "amber": "#d08c00", "green": "#008300", "violet": "#4a3aa7", "grey": "#898781", "text": "#0b0b0b",
              "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
              "ramp": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
              "levels": ["#86b6ef", "#2a78d6", "#104281"]},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500", "magenta": "#d55181",
             "amber": "#c48400", "green": "#008300", "violet": "#9085e9", "grey": "#898781", "text": "#ffffff",
             "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19",
             "ramp": ["#104281", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6"],
             "levels": ["#184f95", "#3987e5", "#9ec5f4"]},
}
ENGINES = {"vllm": "blue", "sglang": "orange", "trtllm": "aqua", "llamacpp": "yellow"}
ENGINE_NAMES = {"vllm": "vLLM", "sglang": "SGLang", "trtllm": "TensorRT-LLM", "llamacpp": "llama.cpp"}
TERMS = {"floor": "violet", "copy": "amber", "kernel_exc": "magenta", "launch": "green", "host_stall": "grey"}
TERM_NAMES = {"floor": "Bandwidth floor", "copy": "Copies", "kernel_exc": "Kernel excess", "launch": "Launch cost",
              "host_stall": "Host stall"}
LEVELS = {"L0": "Runner", "L1": "Engine API", "L2": "Server"}
# every arm's name in one place, in the order the mitigations chart lists them: (label, the phrase "with ...", what it
# switches off, for "switching off X"; None for an arm that changes something rather than switching it off)
ARMS = {"graphs-off": ("CUDA graphs off", "with CUDA graphs off", "CUDA graphs"),
        "async-off": ("Async scheduling off", "with async scheduling off", "async scheduling"),
        "overlap-off": ("Overlap scheduler off", "with the overlap scheduler off", "the overlap scheduler"),
        "mrv2-off": ("Model Runner V2 off", "with Model Runner V2 off", "Model Runner V2"),
        "gpu-sampling": ("Sampling moved to the GPU", "with sampling on the GPU", None)}


def arm_name(arm):
    return ARMS.get(arm, (arm,))[0]
DATA = {"name": "rows"}


def config(p):
    # aria off (guides here, marks and titles below): every mark, axis and legend item would otherwise be read aloud
    # one by one; the chart is one image (its container: role img, labelled), beside a sentence that says what it
    # shows (the summaries)
    return {"background": "transparent", "font": FONT, "view": {"stroke": None}, "aria": False,
            "mark": {"aria": False},
            "axis": {"labelColor": p["secondary"], "titleColor": p["secondary"], "gridColor": p["grid"],
                     "domainColor": p["axis"], "tickColor": p["axis"], "labelFontSize": 12, "titleFontSize": 12,
                     "titleFontWeight": "normal", "labelFont": FONT, "titleFont": FONT},
            "legend": {"labelColor": p["secondary"], "titleColor": p["secondary"], "orient": "top",
                       "titleFontWeight": "normal", "labelFontSize": 12, "titleFontSize": 12, "labelFont": FONT},
            "text": {"font": FONT, "fontSize": 12},
            "title": {"color": p["text"], "fontSize": 13, "fontWeight": 600, "anchor": "start", "font": FONT,
                      "aria": False}}


def spec(p, body, height=240, rows_by=None):
    """A chart as wide as its container (it fits its width, and redraws on resize) and `height` tall; a chart of rows
    gives its height per row and the fields that make a row (rows_by), and whoever fills it sets its height from its
    rows (sized): Vega-Lite drops a per-row height beside a container's width, with a warning."""
    out = {"$schema": SCHEMA, "width": "container", "height": height, "autosize": {"type": "fit-x",
           "contains": "padding"}, "config": config(p), "data": DATA, **body}
    if rows_by:
        out |= {"height": height * 6, "usermeta": {"row_step": height, "rows_by": list(rows_by)}}
    return out


def sized(spec, rows):
    """A chart of rows, its height set from the rows it is filled with (the page script does the same)."""
    meta = spec.get("usermeta") or {}
    if meta.get("row_step"):
        spec["height"] = meta["row_step"] * max(1, len({tuple(r.get(f) for f in meta["rows_by"]) for r in rows}))
    return spec


def colour(p, field, entities, names, title=None, legend=True):
    """Colour by field, its values the entities in their fixed order, each in its own palette slot, legend names
    from names."""
    expr = " : ".join(f"datum.value === '{k}' ? '{v}'" for k, v in names.items()) + " : datum.value"
    return {"field": field, "type": "nominal", "title": title,
            "scale": {"domain": list(entities), "range": [p[s] for s in entities.values()]},
            "legend": {"labelExpr": expr} if legend else None}


def q(field, title, **extra):
    return {"field": field, "type": "quantitative", "title": title, **extra}


PARAM_TICKS = [5.96e8, 1.721e9, 4.022e9, 8.191e9, 1.4768e10, 3.2762e10]
PARAM_AXIS = {"values": PARAM_TICKS, "labelExpr": "datum.value < 1e9 ? '0.6B' : datum.value < 3e9 ? '1.7B' : "
              "datum.value < 6e9 ? '4B' : datum.value < 1.2e10 ? '8B' : datum.value < 2.4e10 ? '14B' : '32B'"}


def end_labels(p, y_field):
    """Direct labels at each engine's smallest model, where the engines differ most; the page script sets label_y
    (nudged apart so no two collide) and is_label on one row per engine."""
    return {"transform": [{"filter": "datum.is_label"}],
            "mark": {"type": "text", "align": "right", "dx": -10, "color": p["secondary"]},
            "encoding": {"x": {"field": "params", "type": "quantitative"},
                         "y": {"field": "label_y", "type": "quantitative"}, "text": {"field": "engine_name"}}}


def engine_label():
    expr = " : ".join(f"datum.engine === '{k}' ? '{v}'" for k, v in ENGINE_NAMES.items()) + " : datum.engine"
    return {"calculate": expr, "as": "engine_name"}


def model_label():
    return {"calculate": "replace(datum.model, /^.*\\//, '')", "as": "model_name"}


def trace(p):
    """The hero: GPU activity in a few decode steps of one cell, one row per engine on a shared microsecond axis.
    Blocks are kernels (each carrying its own launch cost, which CUPTI counts inside the kernel); the gap before the
    next step's tick is the host stall."""
    y = {"field": "engine_name", "type": "nominal", "title": None, "sort": list(ENGINE_NAMES.values()),
         "axis": {"domain": False, "ticks": False, "labelPadding": 10, "labelColor": p["text"], "minExtent": 96}}
    ticks = {"transform": [{"filter": "datum.kind === 'step'"}],
             "mark": {"type": "tick", "orient": "vertical", "color": p["muted"], "thickness": 1, "bandSize": 30},
             "encoding": {"y": y, "x": {"field": "start", "type": "quantitative"}}}
    blocks = {"transform": [{"filter": "datum.kind === 'kernel'"}],
              "mark": {"type": "bar", "height": {"band": 0.5}, "cornerRadius": 1},
              "encoding": {"y": y, "x": q("start", "Microseconds", axis={"grid": False, "tickCount": 6}),
                           "x2": {"field": "end"}, "color": colour(p, "engine", ENGINES, ENGINE_NAMES, legend=False),
                           "tooltip": [{"field": "engine_name", "title": "Engine"},
                                       {"field": "step", "title": "Step"},
                                       {"field": "start", "title": "Start (us)", "format": ",.0f"},
                                       {"field": "end", "title": "End (us)", "format": ",.0f"}]}}
    return spec(p, {"transform": [engine_label()], "layer": [ticks, blocks]}, height=190)


def occupancy(p, batch):
    """Occupied fraction against model size at one batch width, one line per engine, the 90% line marked."""
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"y": {"datum": 0.9}}}
    line = {"mark": {"type": "line", "strokeWidth": 2, "point": {"size": 64, "filled": True, "stroke": p["surface"],
                                                                 "strokeWidth": 2}},
            "encoding": {"x": q("params", "Model", scale={"type": "log", "domain": [2.5e8, 4e10]}, axis=PARAM_AXIS),
                         "y": q("occupied", "GPU busy", scale={"domain": [0, 1]}, axis={"format": "%"}),
                         "color": colour(p, "engine", ENGINES, ENGINE_NAMES, title=None),
                         "tooltip": [{"field": "engine_name", "title": "Engine"},
                                     {"field": "model_name", "title": "Model"},
                                     {"field": "occupied", "title": "Busy", "format": ".0%"},
                                     {"field": "step_ms", "title": "Step (ms)", "format": ".2f"}]}}
    return spec(p, {"title": f"Batch {batch}", "transform": [engine_label(), model_label(),
                                                             {"filter": f"datum.batch === {batch}"}],
                    "layer": [rule, line, end_labels(p, "occupied")]}, height=260)


def frontier(p):
    """Each engine's 90%-occupied frontier: the smallest batch at which a model keeps the GPU 90% occupied. Below a
    line, that engine's steps wait on the host."""
    line = {"mark": {"type": "line", "strokeWidth": 2, "point": {"size": 64, "filled": True, "stroke": p["surface"],
                                                                 "strokeWidth": 2}},
            "encoding": {"x": q("params", "Model", scale={"type": "log", "domain": [2.5e8, 4e10]}, axis=PARAM_AXIS),
                         "y": q("crossover_batch", "Batch where the GPU is 90% busy",
                                scale={"type": "log", "base": 2, "domain": [1, 256]},
                                axis={"values": [1, 2, 4, 8, 16, 32, 64, 128, 256]}),
                         "color": colour(p, "engine", ENGINES, ENGINE_NAMES, title=None),
                         "tooltip": [{"field": "engine_name", "title": "Engine"},
                                     {"field": "model_name", "title": "Model"},
                                     {"field": "crossover_text", "title": "Crossover batch"}]}}
    # a crossover above an unmeasured batch is only an upper bound: drawn hollow, so it never reads as a measurement
    bound = {"mark": {"type": "point", "size": 64, "filled": False, "fill": p["surface"], "fillOpacity": 1,
                      "strokeWidth": 2, "opacity": 1},
             "transform": [{"filter": "datum.bound === 'upper'"}],
             "encoding": {"x": line["encoding"]["x"], "y": line["encoding"]["y"],
                          "color": colour(p, "engine", ENGINES, ENGINE_NAMES, title=None),
                          "tooltip": line["encoding"]["tooltip"]}}
    return spec(p, {"transform": [engine_label(), model_label()],
                    "layer": [line, bound, end_labels(p, "crossover_batch")]}, height=320)


METRICS = {"occupied": ("GPU busy", ".0%", [0, 1]), "sol": ("Speed of light", ".0%", [0, 1]),
           "mbu": ("Bandwidth used (of achievable)", ".0%", [0, 1]), "step_ms": ("Step (ms)", ".2f", None)}


def heatmap(p, metric):
    """One engine's map: model size down, batch across, the metric as a sequential blue; values written in cells. A
    cell run and not accepted (gap, page.map_gaps) is an outlined square marked x, its reason on hover, so the map
    shows the whole coverage."""
    title, fmt, domain = METRICS[metric]
    scale = {"range": p["ramp"], **({"domain": domain} if domain else {"type": "log"})}
    measured = {"filter": f"isValid(datum['{metric}'])"}  # unmeasured: no filled cell
    gap = {"transform": [{"filter": "datum.gap === true"}],
           "mark": {"type": "rect", "fill": p["surface"], "stroke": p["muted"], "strokeWidth": 1,
                    "strokeDash": [3, 2], "cornerRadius": 2},
           "encoding": {"tooltip": [{"field": "model_name", "title": "Model"}, {"field": "batch", "title": "Batch"},
                                    {"field": "rejection", "title": "Not accepted"}]}}
    gap_mark = {"transform": [{"filter": "datum.gap === true"}], "mark": {"type": "text", "fontSize": 12,
                                                                        "color": p["muted"]},
                "encoding": {"text": {"value": "×"}}}
    rect = {"transform": [measured], "mark": {"type": "rect", "stroke": p["surface"], "strokeWidth": 2,
                                              "cornerRadius": 2},
            "encoding": {"color": {**q(metric, title), "scale": scale, "legend": None},
                         "tooltip": [{"field": "model_name", "title": "Model"}, {"field": "batch", "title": "Batch"},
                                     {"field": metric, "title": title, "format": fmt}]}}
    # ink that reads on the cell: light mode darkens with the value, dark mode lightens, so the switch inverts
    mid = (domain[1] * 0.55) if domain else 3.0
    on_high, on_low = ("#ffffff", "#0b0b0b") if p["surface"] == PALETTES["light"]["surface"] else ("#0b0b0b", "#ffffff")
    text = {"transform": [measured], "mark": {"type": "text", "fontSize": 11},
            "encoding": {"text": {"field": metric, "type": "quantitative", "format": fmt},
                         "color": {"condition": {"test": f"datum['{metric}'] > {mid}", "value": on_high},
                                   "value": on_low}}}
    return spec(p, {"transform": [model_label(), {"filter": f"isValid(datum['{metric}']) || datum.gap === true"}],
                    "encoding": {"x": {"field": "batch", "type": "ordinal", "title": "Batch", "sort": "ascending",
                                       "axis": {"labelAngle": 0, "domain": False, "ticks": False}},
                                 "y": {"field": "model_name", "type": "ordinal", "title": None,
                                       "sort": {"field": "params", "order": "descending"},
                                       "axis": {"domain": False, "ticks": False}}},
                    "layer": [gap, gap_mark, rect, text]}, height=36, rows_by=("model",))


def anatomy(p):
    """Where the step goes for one engine at one batch width, one bar per model: the floor, copies, kernel excess,
    launch cost and host stall as shares of the step (a term a little below zero, inside the tolerance, draws as zero;
    its value is in the tooltip)."""
    order = {k: i for i, k in enumerate(TERMS)}
    expr = " : ".join(f"datum.term === '{k}' ? {v}" for k, v in order.items()) + " : 9"
    name = " : ".join(f"datum.term === '{k}' ? '{v}'" for k, v in TERM_NAMES.items()) + " : datum.term"
    return spec(p, {
        "transform": [model_label(), {"fold": [f"{t}_ms" for t in TERMS], "as": ["term_key", "ms"]},
                      {"calculate": "replace(datum.term_key, '_ms', '')", "as": "term"},
                      {"calculate": expr, "as": "term_order"}, {"calculate": name, "as": "term_name"},
                      {"calculate": "max(datum.ms, 0)", "as": "ms_shown"},
                      {"calculate": "datum.ms / datum.step_ms", "as": "share"}],
        "mark": {"type": "bar", "height": {"band": 0.75}, "stroke": p["surface"], "strokeWidth": 2},
        "encoding": {
            "y": {"field": "model_name", "type": "nominal", "title": None, "sort": {"field": "params"},
                  "axis": {"domain": False, "ticks": False, "labelColor": p["text"], "labelPadding": 8}},
            "x": {**q("ms_shown", "Share of the step"), "stack": "normalize", "axis": {"format": "%", "tickCount": 5}},
            "color": colour(p, "term", TERMS, TERM_NAMES, title=None, legend=False),
            "order": {"field": "term_order"},
            "tooltip": [{"field": "model_name", "title": "Model"}, {"field": "term_name", "title": "Part"},
                        {"field": "share", "title": "Share", "format": ".0%"},
                        {"field": "ms", "title": "ms", "format": ".3f"},
                        {"field": "step_ms", "title": "Step (ms)", "format": ".3f"}]}}, height=26,
                rows_by=("model",))


RATIO_AXIS = {"labelExpr": "format(datum.value, '.2~f') + '×'"}


def points(p, x, y, extra, tooltip, colour_enc, size=70):
    """Filled, in a ring of the surface, where the difference is beyond noise (or is the reference itself); hollow, a
    ring of its colour, where it is within noise. One layer that swaps fill and ring by condition: two filtered layers
    left one without data whenever every point fell on one side, and Vega warned of an infinite extent there."""
    within = "datum.beyond === false"
    return [{"mark": {"type": "point", "filled": True, "size": size, "opacity": 1, "strokeWidth": 2},
             "encoding": {"x": x, "y": y, "tooltip": tooltip, **extra,
                          "fill": {"condition": {"test": within, "value": p["surface"]}, **colour_enc},
                          "stroke": {"condition": {"test": f"!({within})", "value": p["surface"]}, **colour_enc}}}]


def engines(p):
    """Each engine's step over the fastest engine's at the same model, batch and level, against model size: 1x is the
    fastest; a hollow point is within three times the run-to-run noise of it."""
    x = q("params", "Model", scale={"type": "log", "domain": [2.5e8, 4e10]}, axis=PARAM_AXIS)
    y = q("ratio", "Step, over the fastest engine's", scale={"zero": False, "nice": True}, axis=RATIO_AXIS)
    col = colour(p, "engine", ENGINES, ENGINE_NAMES, title=None)
    tip = [{"field": "engine_name", "title": "Engine"}, {"field": "model_name", "title": "Model"},
           {"field": "ratio", "title": "Over the fastest", "format": ".3f"},
           {"field": "step_ms", "title": "Step (ms)", "format": ".3f"},
           {"field": "fastest_name", "title": "Fastest"}]
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"y": {"datum": 1}}}
    line = {"mark": {"type": "line", "strokeWidth": 2}, "encoding": {"x": x, "y": y, "color": col}}
    fastest = " : ".join(f"datum.fastest === '{k}' ? '{v}'" for k, v in ENGINE_NAMES.items()) + " : datum.fastest"
    return spec(p, {"transform": [engine_label(), model_label(), {"calculate": fastest, "as": "fastest_name"}],
                    "layer": [rule, line, *points(p, x, y, {}, tip, col), end_labels(p, "ratio")]}, height=280)


def mitigations(p):
    """Every cell with one mitigation changed: its step over the step with every mitigation on, one row per engine and
    mitigation (the mitigations grouped, the engines in their fixed order within each), one point per model and batch;
    hollow within three times the run-to-run noise. A pair from two sessions (exploratory: drift between them is part
    of its difference) gets a row of its own, after the same-session row, labelled so."""
    arms = " : ".join(f"datum.arm === '{a}' ? {i}" for i, a in enumerate(ARMS)) + f" : {len(ARMS)}"
    engines = " : ".join(f"datum.engine === '{e}' ? {i}" for i, e in enumerate(ENGINES)) + " : 9"
    y = {"field": "row", "type": "nominal", "title": None, "sort": {"field": "row_order"},
         "axis": {"domain": False, "ticks": False, "labelColor": p["text"], "labelPadding": 10, "labelLimit": 260}}
    x = q("ratio", "Step with it changed, over the step with every mitigation on",
          scale={"type": "log", "base": 2, "nice": False, "domain": {"unionWith": [0.9, 1.25]}},
          axis={**RATIO_AXIS, "tickCount": 8})
    col = colour(p, "engine", ENGINES, ENGINE_NAMES, title=None, legend=False)
    tip = [{"field": "engine_name", "title": "Engine"}, {"field": "arm_label", "title": "Changed"},
           {"field": "model_name", "title": "Model"}, {"field": "batch", "title": "Batch"},
           {"field": "ratio", "title": "Step, changed / all on", "format": ".2f"},
           {"field": "occupied_on", "title": "GPU busy, all on", "format": ".0%"},
           {"field": "occupied_off", "title": "GPU busy, changed", "format": ".0%"},
           {"field": "sessions_text", "title": "Sessions"}]
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"x": {"datum": 1}}}
    across = "datum.exploratory === true"
    return spec(p, {"transform": [engine_label(), model_label(),
                                  {"calculate": f"datum.arm_label + ', ' + datum.engine_name + ({across} ? "
                                                "', across sessions' : '')", "as": "row"},
                                  {"calculate": f"20 * ({arms}) + 2 * ({engines}) + ({across} ? 1 : 0)",
                                   "as": "row_order"},
                                  {"calculate": "join(datum.sessions || [], ' and ')", "as": "sessions_text"}],
                    "layer": [rule, *points(p, x, y, {}, tip, col, size=56)]}, height=26,
                rows_by=("arm", "engine"))


def stack(p):
    """Each cell measured at another level of the stack: its step over the same cell's at the engine's headline level
    (the engine API; llama.cpp's server, which has no engine API), one row per engine, one point per model and batch,
    coloured and shaped by level (the runner and the server fall either side of 1, so they share a row); hollow within
    three times the run-to-run noise."""
    y = {"field": "engine_name", "type": "nominal", "title": None, "sort": list(ENGINE_NAMES.values()),
         "axis": {"domain": False, "ticks": False, "labelColor": p["text"], "labelPadding": 10}}
    x = q("ratio", "Step, over the engine API's", axis={**RATIO_AXIS, "tickCount": 6},
          scale={"zero": False, "nice": True, "domain": {"unionWith": [0.98, 1.02]}})
    level = " : ".join(f"datum.level === '{k}' ? '{v}'" for k, v in LEVELS.items()) + " : datum.level"
    col = {"field": "level", "type": "nominal", "title": None,
           "scale": {"domain": ["L0", "L2"], "range": [p["levels"][0], p["levels"][2]]},
           "legend": {"labelExpr": level.replace("datum.level", "datum.value")}}
    tip = [{"field": "engine_name", "title": "Engine"}, {"field": "level_name", "title": "Level"},
           {"field": "model_name", "title": "Model"}, {"field": "batch", "title": "Batch"},
           {"field": "ratio", "title": "Over the headline level", "format": ".3f"},
           {"field": "step_ms", "title": "Step (ms)", "format": ".3f"},
           {"field": "headline_ms", "title": "Headline level (ms)", "format": ".3f"}]
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"x": {"datum": 1}}}
    return spec(p, {"transform": [engine_label(), model_label(), {"calculate": level, "as": "level_name"}],
                    "layer": [rule, *points(p, x, y, {"shape": {"field": "level", "legend": col["legend"]}}, tip,
                                            col)]},
                height=40, rows_by=("engine",))


# the calculator's cards: memory bandwidth, B/s, from servebench.physics.PEAKS (one table of datasheet figures)
CARDS = {label: PEAKS[name].bandwidth for label, name in (("H100 PCIe", "NVIDIA H100 PCIe"),
                                                          ("H100 SXM", "NVIDIA H100 80GB HBM3"),
                                                          ("H200", "NVIDIA H200"), ("B200", "NVIDIA B200"))}


def calculator(p):
    """The scenario: the GPU busy share against model size on another card (one engine, one batch), beside what was
    measured on the H100 SXM: the measured line in grey, the scenario in the engine's colour (the page script sets that
    slot) inside the band its host fit's uncertainty allows, a hollow point where its batch lies outside the batches
    the host line was fitted on."""
    x = q("params", "Model", scale={"type": "log", "domain": [2.5e8, 4e10]}, axis=PARAM_AXIS)
    y = q("occupied", "GPU busy", scale={"domain": [0, 1]}, axis={"format": "%"})
    colour = {"field": "series", "type": "nominal", "title": None,
              "scale": {"domain": ["Measured on H100 SXM", "Predicted"], "range": [p["grey"], p["blue"]]}}
    rule = {"mark": {"type": "rule", "color": p["muted"], "strokeWidth": 1}, "encoding": {"y": {"datum": 0.9}}}
    band = {"transform": [{"filter": "isValid(datum.occupied_lo)"}],
            "mark": {"type": "area", "opacity": 0.18},
            "encoding": {"x": x, "y": {"field": "occupied_lo", "type": "quantitative"}, "y2": {"field": "occupied_hi"},
                         "color": colour}}  # the same legend as the line's, or Vega drops the merged one
    line = {"mark": {"type": "line", "strokeWidth": 2}, "encoding": {"x": x, "y": y, "color": colour}}
    dots = {"mark": {"type": "point", "filled": True, "size": 64, "strokeWidth": 2, "opacity": 1},
            "encoding": {"x": x, "y": y,
                         "fill": {"condition": {"test": "datum.extrapolated === true", "value": p["surface"]},
                                  **colour},
                         "stroke": {"condition": {"test": "datum.extrapolated !== true", "value": p["surface"]},
                                    **colour},
                         "tooltip": [{"field": "series", "title": "Series"}, {"field": "model_name", "title": "Model"},
                                     {"field": "occupied", "title": "Busy", "format": ".0%"},
                                     {"field": "occupied_lo", "title": "Busy, low end", "format": ".0%"},
                                     {"field": "occupied_hi", "title": "Busy, high end", "format": ".0%"},
                                     {"field": "step_ms", "title": "Step (ms)", "format": ".2f"}]}}
    return spec(p, {"transform": [model_label()], "layer": [rule, band, line, dots]}, height=260)


def all_specs():
    out = {}
    for theme, p in PALETTES.items():
        out[theme] = {"trace": trace(p), "occ_b1": occupancy(p, 1), "occ_b64": occupancy(p, 64),
                      "frontier": frontier(p), "anatomy": anatomy(p), "engines": engines(p),
                      "mitigations": mitigations(p), "stack": stack(p), "calculator": calculator(p),
                      **{f"map_{m}": heatmap(p, m) for m in METRICS}}
    return out


# the charts' rows: layout (direct labels nudged apart) beside the specs that draw them, one definition for the page
# and the post's figures (anatomy.report.page, anatomy.report.static)


def views(t, keep=dict):
    """The charts' rows, one definition for the page and the post's figures so the two cannot drift apart: occupancy at
    batch 1 and 64 and the frontier, with direct labels; the engine gaps, labelled per chart (one level and batch),
    nudged apart in proportion to that chart's spread; the anatomy (headline cells with a decomposition, the same
    test the patterns use) and the cells left out of it because their terms do not add up. keep: what each headline
    row carries (the page keeps a few columns)."""
    head = [keep(r) for r in t["head"]]
    by = {}
    for g in t["gaps"]:
        by.setdefault((g["level"], g["batch"]), []).append(g)
    gaps = []
    for gs in by.values():
        top = max(g["ratio"] for g in gs)
        gaps += with_labels(gs, "ratio", 0.07 * max(top - 1, 0.1), 1, top)
    # a busy chart holds the cells with a busy share (a timing-only cell has none): a batch with none draws empty
    busy = [r for r in head if r.get("occupied") is not None]
    return {"occupancy": {b: with_labels([r for r in busy if r["batch"] == b], "occupied", 0.06, 0, 1)
                          for b in (1, 64)},
            "frontier": with_labels(t["frontier"], "crossover_batch", 0.55, 1, 256, log=True), "gaps": gaps,
            "anatomy": [r for r in head if row_cell_id(r) in t["parts"]],
            "left_out": [{k: r[k] for k in ("engine", "model", "batch")} for r in head if r.get("consistent") is False]}


def with_labels(rows, y_field, gap, lo, hi, log=False):
    """Direct labels for line charts: one per engine at its smallest model (where engines differ most), nudged apart by
    `gap` (log2 units on a log axis) and kept inside [lo, hi]. Returns copies of rows, the labelled ones with is_label
    and label_y."""
    first = {}
    for r in rows:
        if r.get(y_field) is not None and (r["engine"] not in first or r["params"] < first[r["engine"]]["params"]):
            first[r["engine"]] = r
    t = (lambda v: math.log2(v)) if log else (lambda v: v)
    inv = (lambda v: 2 ** v) if log else (lambda v: v)
    ls = sorted(([r, t(r[y_field])] for r in first.values()), key=lambda x: x[1])
    for i in range(1, len(ls)):
        ls[i][1] = max(ls[i][1], ls[i - 1][1] + gap)
    if ls and ls[-1][1] > t(hi):
        ls[-1][1] = t(hi)
        for i in range(len(ls) - 2, -1, -1):
            ls[i][1] = min(ls[i][1], ls[i + 1][1] - gap)
    at = {id(r): max(inv(y), lo) for r, y in ls}
    return [{**r, "is_label": True, "label_y": at[id(r)]} if id(r) in at else dict(r) for r in rows]
