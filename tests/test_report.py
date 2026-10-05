import json

import pytest

from anatomy.report import charts, derive, synthetic
from anatomy.report.page import build


def test_crossover_interpolates_in_log_batch():
    assert derive.crossover_bound([(1, 0.95), (2, 0.99)]) == (1.0, "exact")
    assert derive.crossover_bound([(1, 0.5), (4, 0.7)]) == (None, None)  # never reached: no extrapolation
    assert derive.crossover_bound([(8, 0.8), (32, 1.0)])[0] == pytest.approx(16.0)  # halfway in log2 from 8 to 32
    assert derive.crossover_bound([(4, 0.85), (1, 0.3), (16, None)]) == (None, None)
    # a rejected batch below the first one over the line: only a bound, never interpolated across the gap
    assert derive.crossover_bound([(1, None), (8, 0.99), (64, 0.99)]) == (8.0, "upper")
    assert derive.crossover_bound([(1, 0.5), (8, None), (64, 0.95)]) == (64.0, "upper")
    assert derive.crossover_bound([(1, 0.95)]) == (1.0, "exact")


def test_synthetic_rows_are_shaped_like_tidy_rows():
    rows = synthetic.cells()
    head = derive.headline_rows(rows)
    assert {r["engine"] for r in head} == set(charts.ENGINES)
    assert all(r["synthetic"] and r["consistent"] for r in head)
    assert all(abs(r["floor_ms"] + r["kernel_exc_ms"] + r["launch_ms"] + r["host_stall_ms"] - r["step_ms"]) < 1e-9
               for r in head)
    assert {r["level"] for r in derive.stack(rows, derive.Noise())} == {"L0", "L2"}


def test_stack_sets_each_level_against_the_engines_headline_level():
    def r(engine, level, step, batch=16):
        return {"engine": engine, "arm": "defaults", "level": level, "model": "Qwen/Qwen3-0.6B", "params": 6e8,
                "batch": batch, "step_ms": step, "session": "S3"}
    rows = [r("sglang", "L0", 1.2), r("sglang", "L1", 1.0), r("sglang", "L2", 1.004), r("vllm", "L1", 1.0),
            r("vllm", "L1", 2.0, batch=64), r("llamacpp", "L0", 1.5), r("llamacpp", "L2", 2.0)]
    got = {(x["engine"], x["level"], x["headline_level"]): x for x in derive.stack(rows, derive.Noise())}
    assert set(got) == {("sglang", "L0", "L1"), ("sglang", "L2", "L1"), ("llamacpp", "L0", "L2")}  # vLLM: one level
    assert got["sglang", "L0", "L1"]["beyond"] and not got["sglang", "L2", "L1"]["beyond"]
    assert got["llamacpp", "L0", "L2"]["ratio"] == pytest.approx(0.75)  # no engine API: against its server


def test_mitigations_pair_each_arm_with_its_defaults():
    m = derive.mitigations(synthetic.cells(), derive.Noise())
    assert {x["arm"] for x in m if x["engine"] == "vllm"} == {"graphs-off", "async-off", "mrv2-off"}
    assert all(x["occupied_on"] is not None and x["ratio"] > 0 and x["beyond"] is not None for x in m)


def test_frontier_rises_with_smaller_models_for_a_host_bound_engine():
    f = [x for x in derive.tables(synthetic.cells())["frontier"] if x["engine"] == "llamacpp"]
    assert f and all(x["crossover_batch"] >= 1 for x in f)


def test_every_spec_uses_the_entity_colours():
    specs = charts.all_specs()
    for theme in ("light", "dark"):
        dumped = json.dumps(specs[theme]["occ_b1"])
        assert charts.PALETTES[theme]["blue"] in dumped and charts.PALETTES[theme]["yellow"] in dumped
        assert charts.PALETTES[theme]["violet"] in json.dumps(specs[theme]["anatomy"])
        assert specs[theme]["anatomy"]["data"] == {"name": "rows"}


def test_page_builds_from_synthetic_rows(tmp_path):
    build(None, tmp_path)
    html = (tmp_path / "index.html").read_text()
    assert "synthetic" in html.lower() and "vega-embed" in html
    assert not (tmp_path / "CNAME").exists() and "huggingface" not in html  # no address, no dataset: no links
    data = json.loads((tmp_path / "data" / "page.json").read_text())
    assert data["headline"] and data["frontier"] and data["specs"]["light"]["trace"] and "rows" not in data
    assert data["principles"] and "patterns" not in data  # the page leads with principles; every pattern is a file
    assert json.loads((tmp_path / "data" / "patterns.json").read_text())["patterns"]
    assert (tmp_path / "data" / "cells.csv").exists()
    assert all(f'id="{p["section"]}"' in html for p in data["principles"])  # each links to its chart


def test_page_predictions_come_from_the_tested_model():
    """The calculator's rows are anatomy.fit.predict's, computed in Python; the page script only filters them."""
    from anatomy import fit
    from anatomy.report.derive import headline_fits
    from anatomy.report.page import predictions

    rows = synthetic.cells()
    fits = headline_fits(rows)
    preds = predictions(rows, fits)
    llama = [r for r in derive.headline_rows(rows) if r["engine"] == "llamacpp"]
    direct = fit.predict(llama, fits["llamacpp"], charts.CARDS["B200"] / charts.CARDS["H100 SXM"])
    page = [p for p in preds if p["engine"] == "llamacpp" and p["card"] == "B200"]
    key = ("model", "batch", "occupied")
    assert page and [tuple(p[k] for k in key) for p in page] == [tuple(d[k] for k in key) for d in direct]
    assert {p["engine"] for p in preds} == {"llamacpp"}  # the toy's others wait on the host too rarely to fit


def test_changelog_rows():
    from anatomy.report.page import changelog
    rows = changelog()
    assert rows and rows[0]["Version set"] == "2026-09" and rows[0]["vLLM"] == "0.30.0"


def test_page_from_a_cell_table_with_gaps_is_strict_json(tmp_path):
    """Real cell tables have missing values (a cell with no trace): loaded from Parquet they must become None, not NaN,
    and the page's data must stay strict JSON (the browser's JSON.parse rejects NaN and Infinity)."""
    import json
    import re

    import pandas as pd

    from anatomy.report.page import build
    from anatomy.tidy import load_cells

    rows = [{k: v for k, v in r.items() if k != "sources"} for r in synthetic.cells()]
    for r in rows[::3]:
        r["occupied"] = None        # a float column with gaps: Parquet stores NaN
        r["kernel_exc_ms"] = None
    for r in rows:                  # a list column: Parquet gives it back as an array, which JSON refuses
        r["launch_floor_share_range"] = [0.2, 0.25]
    pd.DataFrame(rows).to_parquet(tmp_path / "cells.parquet")
    loaded = load_cells(tmp_path / "cells.parquet")
    assert all(r["occupied"] is None or r["occupied"] == r["occupied"] for r in loaded)  # None, never NaN
    assert all(r["launch_floor_share_range"] == [0.2, 0.25] for r in loaded)
    build(tmp_path / "cells.parquet", tmp_path / "site")
    reject = lambda c: (_ for _ in ()).throw(ValueError(c))  # noqa: E731
    json.loads((tmp_path / "site" / "data" / "page.json").read_text(), parse_constant=reject)
    html = (tmp_path / "site" / "index.html").read_text()
    embedded = re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1)
    json.loads(embedded, parse_constant=reject)


def page_data(rows):
    from anatomy.report import page, patterns
    t = derive.tables(rows)
    return page.page_data(t, patterns.report(t), derive.about(rows))


def test_page_headline_rows_are_the_accepted_ones():
    rows = synthetic.cells()
    bad = next(r for r in rows if r["arm"] == "defaults" and r["engine"] == "sglang")
    bad |= {"valid": False, "problems": ["1 stalled gaps (preemption or queueing)"]}
    data = page_data(rows)
    key = lambda r: (r["engine"], r["model"], r["batch"], r["level"])  # noqa: E731
    assert key(bad) not in {key(r) for r in data["headline"]}


def test_mitigations_pair_with_the_accepted_defaults_at_the_same_prompt():
    def r(arm, step, prompt=896, **kw):
        return {"engine": "vllm", "arm": arm, "level": "L1", "model": "Qwen/Qwen3-0.6B", "batch": 1,
                "prompt_len": prompt, "step_ms": step, "occupied": 0.5, **kw}
    rows = [r("defaults", 10.0), r("defaults", 50.0, prompt=8064), r("graphs-off", 20.0)]
    m = derive.mitigations(rows, derive.Noise())
    assert len(m) == 1 and m[0]["ratio"] == pytest.approx(2.0)        # against the 896-token defaults, not 8064
    rows[0] |= {"valid": False, "problems": ["stalled"]}
    assert derive.mitigations(rows, derive.Noise()) == []             # a rejected baseline pairs with nothing


def test_every_export_says_what_its_data_is_and_a_study_page_needs_accepted_cells(tmp_path, monkeypatch):
    """The status travels with the data: preview (the toy model), validation (a validation grid's cell, or any card
    but the study's), study; in page.json and patterns.json beside their numbers, so a detached file says it. A page
    built from study data refuses principles that cite no accepted cell."""
    from anatomy.report import page
    rows = [{k: v for k, v in r.items() if k != "synthetic"} for r in synthetic.cells()]
    assert page_data(synthetic.cells())["about"]["dataset_status"] == "preview"
    for r in rows:
        r["gpu"] = "NVIDIA H100 80GB HBM3"   # the study's card
    assert page_data(rows)["about"]["dataset_status"] == "study"
    rows[0]["validation"] = True
    assert page_data(rows)["about"]["dataset_status"] == "validation"
    rows[0]["validation"] = False
    rows[1]["gpu"] = "NVIDIA H100 NVL"
    about = page_data(rows)["about"]
    assert about["dataset_status"] == "validation" and about["cards"] == ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL"]
    rows[1]["gpu"] = "NVIDIA H100 80GB HBM3"
    monkeypatch.setattr(page.derive, "load", lambda cells: (rows, ([], None), derive.Noise()))
    monkeypatch.setattr(page.patterns, "check", lambda report, rows: ["crossover: cites no cell"])
    with pytest.raises(SystemExit, match="cites no cell"):
        page.build(tmp_path / "cells.parquet", tmp_path / "site")


def test_patterns_name_only_differences_beyond_noise(monkeypatch):
    from anatomy.report import patterns as P
    monkeypatch.setitem(derive.TOLERANCES, "step_noise", 0.006)  # validation's 0.6% (the study's is S1's, 1.0%)
    cell = {"engine": "vllm", "level": "L1", "session": "S2"}
    assert derive.compare({**cell, "step_ms": 1.03}, {**cell, "step_ms": 1.0}, derive.Noise())["beyond"]  # 3 x 0.85%
    assert not derive.compare({**cell, "step_ms": 1.02}, {**cell, "step_ms": 1.0}, derive.Noise())["beyond"]
    rows = [{"engine": e, "arm": "defaults", "level": lvl, "model": "Qwen/Qwen3-0.6B", "params": 6e8, "batch": 1,
             "prompt_len": 896, "step_ms": step, "valid": True, "session": "S2"}
            for e, lvl, step in (("sglang", "L1", 1.50), ("vllm", "L1", 1.51), ("trtllm", "L1", 1.80),
                                 ("llamacpp", "L2", 3.10), ("vllm", "L2", 1.52))]
    t = derive.tables(rows)
    gaps = t["gaps"]
    # like with like: engine APIs against engine APIs, servers against servers (never llama.cpp's server against L1)
    assert [p["text"] for p in P.engine_patterns(t)] == [
        "0.6B, batch 1, the engine API (L1): SGLang has the shortest step, 1.50 ms; vLLM within noise of it; "
        "TensorRT-LLM's 20% longer (± 0.8%, one SD).",
        "0.6B, batch 1, the server (L2): vLLM has the shortest step, 1.52 ms; llama.cpp's 2.0x as long."]
    assert sum(g["sigma"] is not None for g in gaps) == 3                # three comparisons, not five rows
    found, stats = P.patterns(t)
    assert stats["comparisons"] == 4                                     # and vLLM's server against its engine API
    # chance by family: a group of three against its fastest, two pairs, the noise assumed (no noise block)
    assert stats["expected_by_chance"] == pytest.approx(derive.chance(3, None) + 2 * derive.chance(2, None))


def noise_rows(engine="vllm", sds=((1, 0.001), (8, 0.003), (64, 0.006)), arm="defaults", model="Qwen/Qwen3-0.6B"):
    return [{"session": "S1", "engine": engine, "arm": arm, "level": "L1", "model": model, "batch": b,
             "processes": 6, "sd": sd} for b, sd in sds]


def test_noise_follows_the_batch_and_never_drops_below_its_floor():
    noise = derive.Noise(noise_rows())
    at = lambda b, **kw: noise.sigma({"engine": "vllm", "level": "L1", "arm": "defaults",  # noqa: E731
                                      "model": "Qwen/Qwen3-0.6B", "params": 6e8, "batch": b, **kw})
    assert at(1)[0] == pytest.approx(derive.FLOOR) and at(1)[1] == 6   # 0.1% fitted, lifted to the floor; an end: 6 df
    assert at(64)[0] > at(8)[0] > derive.FLOOR                          # the spread grows with the batch
    assert at(256) == at(64)                                            # held flat beyond the batches measured
    assert at(64, arm="mrv2-off") == at(64)                             # no noise cells of its own: the largest
    assert derive.Noise().sigma({"engine": "vllm", "level": "L1", "batch": 1, "model": "Qwen/Qwen3-0.6B"}) == \
        (derive.TOLERANCES["step_noise"], None)                         # no noise block: the tolerance, assumed


def test_noise_grows_with_drift_between_sessions_once_per_engine():
    sentinels = [{"session": s, "position": p, "engine": "vllm", "level": "L1", "arm": "defaults",
                  "model": "Qwen/Qwen3-0.6B", "batch": 1, "step_ms": x, "valid": v}
                 for s, p, x, v in (("S2", "start", 1.00, "True"), ("S2", "end", 1.02, "True"),
                                    ("S3", "start", 1.05, "True"), ("S3", "end", 9.0, "False"))]
    noise = derive.Noise(noise_rows(sds=((1, 0.004), (64, 0.004))), sentinels)
    assert noise.drift("vllm", "S3", "S2") == pytest.approx(1.05 / 1.01 - 1)  # valid readings' means: 9.0 is not one
    a = {"engine": "vllm", "level": "L1", "session": "S3", "step_ms": 1.03, "batch": 1, "model": "Qwen/Qwen3-0.6B"}
    b = {**a, "session": "S2", "step_ms": 1.0}
    s1 = noise.sigma(a)[0]
    assert noise.combined(a, {**b, "session": "S3"})[0] == pytest.approx(s1 * 2 ** 0.5)
    assert noise.combined(a, b)[0] == pytest.approx((2 * s1 ** 2 + (1.05 / 1.01 - 1) ** 2) ** 0.5)  # drift once
    assert not derive.compare(a, b, noise)["beyond"]                      # 3% across sessions that drifted 4%
    assert derive.compare(a, {**b, "session": "S3"}, noise)["beyond"]     # 3% within one session
    other = {**b, "engine": "sglang"}                                     # no SGLang sentinel: charged the tolerance
    assert noise.combined(a, other)[0] > noise.combined(a, b)[0]


def test_chance_grows_when_the_fastest_is_picked_and_the_noise_is_estimated():
    assert derive.chance(2, None) == pytest.approx(0.0027, abs=0.001)    # a pair, known noise: the normal tail
    assert derive.chance(4, None) > 3 * derive.chance(2, None) * 0.9     # three comparisons against a picked minimum
    assert derive.chance(2, 5) > 3 * derive.chance(2, None)              # five degrees of freedom: a fatter tail


def test_crossover_needs_the_gpu_to_stay_busy():
    # llama-server's occupancy falls with the batch: above the line at batch 1 is not "from batch 1"
    assert derive.crossover_bound([(1, 0.92, 0.005), (8, 0.85, 0.005), (64, 0.70, 0.005)]) == (None, None)
    assert derive.crossover_bound([(1, 0.95, 0.01), (8, 0.89, 0.01), (64, 0.95, 0.01)]) == (1.0, "exact")  # in noise
    assert derive.crossover_bound([(1, 0.95, 0.001), (8, 0.80, 0.001), (64, 0.95, 0.001)])[0] > 8


KINDS = ["crossover", "anatomy", "engines", "mitigations", "attribution", "stack", "host"]


def test_patterns_from_the_synthetic_grid_cover_every_kind():
    from anatomy.report.patterns import patterns
    found, stats = patterns(derive.tables(synthetic.cells()))
    kinds = [p["kind"] for p in found]
    assert stats["comparisons"] > 0 and 0 < stats["expected_by_chance"] < stats["comparisons"] * 0.01
    assert set(KINDS) <= set(kinds)
    assert kinds == sorted(kinds, key=KINDS.index)


def test_principles_are_few_ordered_like_the_page_and_say_what_the_tables_say(monkeypatch):
    from anatomy.report.patterns import principles
    monkeypatch.setitem(derive.TOLERANCES, "step_noise", 0.006)  # the figures below assume validation's 0.6%
    ps = principles(derive.tables(synthetic.cells()))
    order = ["crossover", "anatomy", "engines", "mitigations", "stack", "calculator"]
    assert [p["section"] for p in ps] == [s for s in order if s in {p["section"] for p in ps}] and len(ps) <= 6
    assert all(p["claim"].endswith(".") and p["evidence"].endswith(".") for p in ps)

    def r(engine, level, step, occupied=0.95, model="Qwen/Qwen3-0.6B", params=6e8):
        return {"engine": engine, "arm": "defaults", "level": level, "model": model, "params": params, "batch": 1,
                "prompt_len": 896, "step_ms": step, "occupied": occupied, "valid": True, "session": "S2"}
    rows = [r("sglang", "L1", 1.50), r("sglang", "L2", 1.505), r("vllm", "L1", 1.60), r("vllm", "L2", 1.61),
            r("llamacpp", "L2", 3.0, occupied=0.8), r("llamacpp", "L2", 20.0, 0.99, "Qwen/Qwen3-8B", 8e9),
            {**r("vllm", "L2", 1.9), "batch": 64,  # a server cell rejected: its sequences fell behind the batch
             "problems": ["44 sequences fell behind the batch (preemption or queueing)"]}]
    got = {p["section"]: p for p in principles(derive.tables(rows))}
    assert {k: v for k, v in got["crossover"].items() if k != "cells"} == {
        "section": "crossover", "claim": "Only models up to 0.6B leave the GPU waiting on the host.",
        "evidence": "Under 90% busy beyond noise in 1 of 4 cells: llama.cpp in 1 of its 2 cells (0.6B, batch 1). vLLM "
                    "and SGLang never."}
    assert len(got["crossover"]["cells"]) == 4  # every cell the claim read, by its id
    assert got["engines"]["claim"] == "The slowest engine's step is up to 6.7% longer than the fastest."
    # within noise is stated as the most it could hide, per engine: the difference plus three times its noise; the
    # server is the whole stack's difference, not HTTP's alone
    assert got["stack"]["claim"] == "The server stack costs nothing beyond noise over the engine API."
    assert got["stack"]["evidence"].startswith("Within noise in both cells measured at both") or \
        got["stack"]["evidence"].startswith("Within noise in all")
    assert "the most it could hide is 3.2% for vLLM, 2.9% for SGLang" in got["stack"]["evidence"] or \
        "for vLLM" in got["stack"]["evidence"]
    assert got["stack"]["evidence"].endswith("Not compared: the one server cell rejected, vLLM at 64 streams on 0.6B "
                                             "(sequences fell behind the batch: a frontend that could not keep its "
                                             "streams even on this host).")
    assert all(p["cells"] for p in got.values())
    assert "anatomy" not in got and "mitigations" not in got  # nothing to say without traces or arms: no sentence


def test_the_anatomy_draws_only_cells_with_a_decomposition_the_same_set_as_the_post():
    from anatomy.report import static
    rows = synthetic.cells()
    gone = next(r for r in rows if r["arm"] == "defaults" and r["engine"] == "vllm" and r["batch"] == 1)
    gone |= {"floor_share": None, "kernel_ms": None}  # no node trace: no decomposition, though floor_ms is physics
    data = page_data(rows)
    key = lambda r: (r["engine"], r["model"], r["batch"])  # noqa: E731
    assert key(gone) not in {key(r) for r in data["anatomy"]}
    figs = static.figures(derive.tables(rows))
    page_keys = {(r["engine"], static.short(r["model"]), r["batch"]) for r in data["anatomy"]}
    smallest = static.models_of(static.anatomy_rows(derive.tables(rows)))[0]
    expected = {k for k in page_keys if k[1] == smallest and k[2] in static.REGULAR}
    assert {key(r) for r in figs["anatomy-batch"]} == expected
    assert {key(r) for r in figs["anatomy-models"]} <= page_keys
    assert {r["batch"] for r in figs["engines"]} <= {1, 16} and figs["mitigations"]


def test_a_noisier_host_scales_that_engines_noise_in_that_session():
    sentinels = [{"session": "S2", "position": p, "engine": "vllm", "level": "L1", "arm": "defaults",
                  "model": "Qwen/Qwen3-0.6B", "batch": 1, "step_ms": x, "valid": "True"}
                 for p, x in (("start", 1.00), ("end", 1.06))]
    noise = derive.Noise(noise_rows(sds=((1, 0.004), (64, 0.004))), sentinels)
    cell = {"engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B", "params": 6e8, "batch": 1,
            "step_ms": 1.03, "session": "S2"}
    before = noise.sigma(cell)[0]
    (check,) = noise.check([cell])
    assert not check["holds"] and check["readings"] == 3            # 3% spread on a host S1 put at 0.4%
    assert noise.sigma(cell)[0] == pytest.approx(check["sd"]) and check["sd"] > before
    assert noise.sigma({**cell, "session": "S3"})[0] == before       # another session's host is not touched


def test_tracing_overhead_is_a_bias_on_a_trace_fraction_never_a_resolved_stall():
    noise = derive.Noise()
    cell = {"engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B", "params": 6e8, "batch": 1,
            "occupied": 0.87, "occupied_source": "trace", "profiler_overhead": 0.0}
    assert derive.waiting(cell, noise)
    assert not derive.waiting({**cell, "profiler_overhead": 0.05}, noise)  # 5% of tracing could be the whole gap


def test_measured_occupancy_noise_replaces_the_step_noise_proxy():
    rows = noise_rows(sds=((1, 0.002), (64, 0.002)))
    rows += [{**r, "occupied_sd": occ} for r, occ in zip(noise_rows(sds=((1, 0.002), (64, 0.002))), (0.004, 0.012))]
    cell = {"engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B", "params": 6e8,
            "occupied": 0.95, "occupied_source": "trace"}
    measured, proxy = derive.Noise(rows), derive.Noise(noise_rows(sds=((1, 0.002), (64, 0.002))))
    assert derive.occupancy_noise({**cell, "batch": 64}, measured) > derive.occupancy_noise({**cell, "batch": 1},
                                                                                           measured)
    assert derive.occupancy_noise({**cell, "batch": 64}, measured) > derive.occupancy_noise({**cell, "batch": 64},
                                                                                           proxy)


def test_occupancy_noise_widens_with_a_noisier_session_and_an_arm_never_borrows_a_smaller_spread():
    """The external review's probe: scaling a session's noise tenfold moved the step's sigma tenfold and left the
    occupancy's where it was. And a host-bound arm the noise block traced only for defaults (about 98% busy) took the
    defaults' small spread; it now takes the larger of any arm's and its own step-derived one."""
    rows = [{**r, "occupied_sd": 0.01} for r in noise_rows(sds=((1, 0.01), (64, 0.01)))]
    noise = derive.Noise(rows)
    cell = {"engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B", "params": 6e8,
            "batch": 1, "session": "S2", "occupied": 0.8, "occupied_source": "trace"}
    before = derive.occupancy_noise(cell, noise)
    noise.scale[("vllm", "S2")] = 10
    assert derive.occupancy_noise(cell, noise) == pytest.approx(10 * before)
    assert derive.occupancy_noise({**cell, "session": "S3"}, noise) == pytest.approx(before)
    noisy = derive.Noise([*rows, *noise_rows(sds=((1, 0.2), (64, 0.2)), arm="graphs-off")])
    arm = {**cell, "arm": "graphs-off", "occupied": 0.15}  # timed, never traced, by the noise block
    from_steps = noisy.sigma(arm)[0] * 0.15                                      # about 0.03: its own steps' noise
    assert from_steps > 0.02 and derive.occupancy_noise(arm, noisy) == pytest.approx(from_steps)
    quiet = derive.Noise([*rows, *noise_rows(sds=((1, 0.001), (64, 0.001)), arm="graphs-off")])
    assert derive.occupancy_noise(arm, quiet) == pytest.approx(derive.occupancy_noise(cell | {"session": "S3"},
                                                                                       noise))  # never below 0.011
    traced = derive.Noise([*rows, *({**r, "occupied_sd": 0.03}
                                    for r in noise_rows(sds=((1, 0.05), (64, 0.05)), arm="graphs-off"))])
    own = derive.occupancy_noise(arm, traced)                                    # its own arm traced: that spread
    assert own == pytest.approx(traced.line(traced.occupancy, arm)[0]) and own == pytest.approx(0.03, rel=0.15)


def test_a_comparison_across_sessions_is_exploratory_and_never_a_principle():
    """The external review's probe: a defaults cell from S2 and an arm cell from S3 made a mitigation comparison. A
    drift term is not a controlled change: the pair is listed as exploratory and leaves the mitigations every principle
    and chart reads; within one session it stays. Every pair names its cells by one id, release included."""
    from anatomy.report import patterns, synthetic
    rows = synthetic.cells()
    base = next(r for r in rows if r["engine"] == "vllm" and r["arm"] == "defaults")
    arm = next(r for r in rows if r["engine"] == "vllm" and r["arm"] != "defaults"
               and r["model"] == base["model"] and r["batch"] == base["batch"])
    apart = derive.tables([{**base, "session": "S2"}, {**arm, "session": "S3"}])
    assert apart["mitigations"] == [] and len(apart["exploratory"]) == 1
    (x,) = apart["exploratory"]
    assert x["sessions"] == ["S2", "S3"] and x["cells"][0] == f"vllm-{base['version']}/defaults/L1/Qwen3-0.6B/b1"
    assert patterns.report(apart)["exploratory"] == apart["exploratory"]
    together = derive.tables([{**base, "session": "S2"}, {**arm, "session": "S2"}])
    assert len(together["mitigations"]) == 1 and together["exploratory"] == []


def test_the_greedy_token_check_is_summarised_for_the_page(tmp_path):
    (tmp_path / "agreement.csv").write_text(
        "kind,session,model,a,b,prompts,identical,median_first_divergence,flagged,odd\n"
        "identity,S2,Qwen/Qwen3-0.6B,vllm-0.30.0/defaults/L1,vllm-0.30.0/async-off/L1,8,8,32,False,\n"
        "identity,S2,Qwen/Qwen3-0.6B,sglang-0.5.20/defaults/L1,sglang-0.5.20/overlap-off/L1,8,7,32,True,\n"
        "pair,S2,Qwen/Qwen3-0.6B,sglang-0.5.20/defaults/L1,vllm-0.30.0/defaults/L1,8,2,6.5,False,\n"
        "pair,S2,Qwen/Qwen3-0.6B,llamacpp-v0.5.0/defaults/L2,vllm-0.30.0/defaults/L1,8,0,0,True,llamacpp\n")
    a = derive.agreement(tmp_path / "agreement.csv")
    assert a == {"identity": [1, 2], "pairs": 2, "median_first_divergence": [0.0, 6.5], "flagged": ["llamacpp"],
                 "probe_errors": 0}
    assert derive.agreement(tmp_path / "missing.csv") is None


def test_a_serial_arm_is_judged_on_the_traces_busy_time_not_occupied_times_step():
    """occupied is the trace's own fraction where tracing stretched the step: occupied x step then carries the
    tracing overhead (2.4% here), enough to fail gpu_work_held against an exact serial arm whose busy time held."""
    def r(arm, step, occupied, busy):
        return {"engine": "sglang", "arm": arm, "level": "L1", "model": "Qwen/Qwen3-0.6B", "batch": 1,
                "prompt_len": 896, "step_ms": step, "occupied": occupied, "busy_ms": busy}
    rows = [r("defaults", 1.000, 0.952, 0.975), r("overlap-off", 1.300, 0.975 / 1.3, 0.976)]
    (m,) = derive.mitigations(rows, derive.Noise())
    assert (m["busy_on"], m["busy_off"]) == (0.975, 0.976) and m["stall_on"] == pytest.approx(0.025)
    assert m["gpu_work_held"] is True


def test_the_map_shows_cells_run_and_not_accepted_with_their_reason():
    """The map is the grid's whole coverage: a failed or rejected cell is drawn outlined with its reason, and the
    engine's summary counts it, never a silent hole."""
    from anatomy.report import page, patterns
    rows = synthetic.cells()
    i = next(i for i, r in enumerate(rows) if r["engine"] == "vllm" and r["arm"] == "defaults" and r["level"] == "L1")
    rows[i] = {**rows[i], "valid": False, "problems": ["stalled"]}
    (gap,) = [g for g in page.map_gaps(rows) if g["engine"] == "vllm"]
    assert gap["gap"] is True and gap["batch"] == rows[i]["batch"] and "stalled" in gap["rejection"]
    said = patterns.summaries(derive.tables(rows))["map"]["occupied"]["vllm"]
    assert said.endswith("1 more was run and not accepted (outlined).")


def test_the_posts_figures_keep_to_the_regular_batches():
    """The grid's "max" cells (14B at 211 sequences, the KV limit) stay on the page and off the post's figures."""
    from anatomy.report import static
    rows = synthetic.cells()
    odd = {**next(r for r in rows if r["arm"] == "defaults" and r["engine"] == "vllm" and r["batch"] == 128),
           "batch": 211}
    figs = static.figures(derive.tables(rows + [odd]))
    drawn = {r["batch"] for rs in figs.values() for r in rs if r.get("batch")}
    assert 211 not in drawn and {1, 64} <= drawn
    assert static.regular([{"batch": 211}, {"batch": 64}, {"model": "x"}]) == [{"batch": 64}, {"model": "x"}]


def test_a_printed_figure_carries_its_own_legend_of_what_it_holds():
    """Each figure's spec stands alone: the anatomy figures carry the terms' legend, and a legend of engines lists
    only the engines in the figure's rows (the page's palette names every engine the harness knows)."""
    from anatomy.report import static
    rows = [r for r in synthetic.cells() if r["engine"] != "llamacpp"]
    figs = static.figures(derive.tables(rows))
    legends = {}
    def walk(n):
        if isinstance(n, dict):
            c = (n.get("encoding") or {}).get("color")
            if isinstance(c, dict) and isinstance(c.get("legend"), dict):
                legends.setdefault(c["field"], []).append(c["legend"])
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(static.spec_for("anatomy-batch", figs["anatomy-batch"], "light"))
    assert legends["term"] and all("labelExpr" in lg for lg in legends["term"])
    legends.clear()
    walk(static.spec_for("engines", figs["engines"], "light"))
    assert legends["engine"] and all(lg.get("values") == ["vllm", "sglang", "trtllm"] for lg in legends["engine"])
    status = static.spec_for("engines", figs["engines"], "dark", status="Layout preview")
    assert status["title"]["subtitle"][0] == "Layout preview" and status["config"]["background"] == "transparent"


