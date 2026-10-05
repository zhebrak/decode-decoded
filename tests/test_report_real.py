"""The report on real cells: tidy's output for a validation session (an H100 NVL, Qwen3-0.6B, every engine;
tests/fixtures/cells-nvl, the cell table only, no traces; `make fixture-cells` rebuilds it from the raw session). A
column tidy now writes and the fixture lacks fails test_the_fixture_is_tidys_current_table, so the fixture is rebuilt
and a column renamed in tidy and still read by the report fails here, not after the study's sessions. The synthetic
rows the other report tests use lack most of tidy's columns."""
import json
import re
from pathlib import Path

import pytest

from anatomy.report import derive, page, patterns, static, synthetic
from anatomy.tidy import load_cells

CELLS = Path(__file__).parent / "fixtures" / "cells-nvl" / "cells.parquet"


@pytest.fixture(scope="module")
def rows():
    return load_cells(CELLS)


def strict(text):
    return json.loads(text, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_the_synthetic_rows_carry_only_columns_tidy_writes(rows):
    real = set().union(*(r.keys() for r in rows))
    assert set().union(*(r.keys() for r in synthetic.cells())) - real == {"synthetic"}


def test_the_tables_and_the_principles_build_from_real_cells(rows):
    t = derive.tables(rows)
    assert t["head"] and t["gaps"] and t["stack"] and t["parts"] and t["crossovers"]
    rep = patterns.report(t)
    assert rep["principles"] and all(p["claim"].endswith(".") and p["evidence"].endswith(".")
                                     for p in rep["principles"])
    assert rep["patterns"] and rep["stats"]["comparisons"] > 0
    figures = static.figures(t)
    assert figures["engines"] and figures["mitigations"] and figures["anatomy-batch"]


def test_the_page_builds_from_real_cells_as_a_validation_preview(tmp_path, rows):
    index = page.build(CELLS, tmp_path)
    html = index.read_text()
    assert "Validation preview" in html and "What the measurements show" in html
    data = strict((tmp_path / "data" / "page.json").read_text())
    assert data["about"]["dataset_status"] == "validation" and data["principles"] and data["headline"]
    assert data["about"]["cards"] == ["NVIDIA H100 NVL"] and data["about"]["measured"]
    # the explorer: every defaults cell at its headline level, accepted or not, its checks in words
    ex = data["explorer"]
    assert ex and {c["engine"] for c in ex} == {"vllm", "sglang", "trtllm", "llamacpp"}
    assert all(c["accepted"] or c["rejection"] for c in ex)
    untested = "ran, but the contention stretched the step too little to test it"
    assert {v for c in ex for _, v in c["checks"]} <= {"passed", "failed", "not run", untested}
    assert strict((tmp_path / "data" / "patterns.json").read_text())["about"]["dataset_status"] == "validation"
    # the opening picture: three whole steps of every engine's node trace, cut where tidy read the cell (its own
    # anchor: llama.cpp's per-step copy, where its graph launch misses the step its KV view grows)
    trace = data["trace"]  # flat rows, one engine each (derive.strips)
    assert {r["engine"] for r in trace} == {"vllm", "sglang", "trtllm", "llamacpp"}
    steps = {e: sum(r["kind"] == "step" and r["engine"] == e for r in trace) for e in ("vllm", "llamacpp")}
    assert steps == {"vllm": 4, "llamacpp": 4} and data["trace_stretch"] is not None
    embedded = re.search(r'<script type="application/json" id="page-data">(.*?)</script>', html, re.S).group(1)
    strict(embedded)
    assert (tmp_path / "data" / "cells.csv").read_bytes() == CELLS.with_suffix(".csv").read_bytes()  # tidy's own


def test_the_fixture_is_tidys_current_table(tmp_path):
    """Every column tidy writes today, for a timing unit, a graph trace and a node trace, is in the fixture: a rename
    or a new column in tidy fails here until `make fixture-cells` rebuilds it."""
    from anatomy.tidy import build
    from tests.nsysfake import fake_trace
    from tests.test_tidy import STEP, jsonl, realistic_unit

    realistic_unit(tmp_path, "S2", batches=(64,))
    profile = [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}]
    graph = realistic_unit(tmp_path, "S2", capture="graph")
    jsonl(graph / "driver.jsonl", profile)
    fake_trace(graph / "graph.1.sqlite", graphs=[(i * STEP, i * STEP + 9_000_000, 1) for i in range(20)])
    node = realistic_unit(tmp_path, "S2", capture="node")
    jsonl(node / "driver.jsonl", profile)
    fake_trace(node / "node.1.sqlite", kernels=[(i * STEP + j * 1_000_000, i * STEP + j * 1_000_000 + 900_000,
                                                 f"gemm_{j}", {"graphNodeId": j, "graphId": 1, "correlationId": i})
                                                for i in range(20) for j in range(9)])
    written = set().union(*(r.keys() for r in build([tmp_path / "S2"], bw=3.1e12)))
    assert written - set().union(*(r.keys() for r in load_cells(CELLS))) == set()
