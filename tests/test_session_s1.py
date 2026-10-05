"""The S1-shaped session (tests/s1fake.py) through tidy and patterns, against golden outputs: noise.csv,
sentinels.csv, reruns.csv, tolerances-suggested.yaml and patterns.json's statistics must not change under a
refactor, and change only in a commit that means to change them (UPDATE_GOLDEN=1 pytest tests/test_session_s1.py,
then read the diff). The NVL fixture has none of these rows; this session is what pins them."""
import json
import os
from pathlib import Path

import pytest

from anatomy import tidy
from anatomy.report import patterns
from tests.s1fake import SESSION, make_session

GOLDEN = Path(__file__).parent / "golden" / "s1-fake"
PINNED = ("noise.csv", "sentinels.csv", "reruns.csv", "tolerances-suggested.yaml", "agreement.csv")


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("s1")
    session = make_session(root)
    out = root / "out"
    tidy.main([str(session), "--calibrate", str(session), "--out", str(out)])
    patterns.main([str(out / "cells.parquet"), "--json", str(out / "patterns.json")])
    stats = json.loads((out / "patterns.json").read_text())["stats"]
    (out / "stats.json").write_text(json.dumps(stats, indent=1, sort_keys=True) + "\n")
    return out


def check(name, got):
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(got)
    assert path.exists(), f"no golden {path}: UPDATE_GOLDEN=1 pytest {__file__}"
    assert got == path.read_text(), f"{name} changed; if meant, UPDATE_GOLDEN=1 pytest {__file__} and read the diff"


@pytest.mark.parametrize("name", [*PINNED, "stats.json"])
def test_the_session_reads_as_pinned(outputs, name):
    check(name, (outputs / name).read_text())


def test_what_the_session_holds(outputs):
    """The shape the goldens pin, in words: so a diff in them is read against this."""
    import pandas as pd

    noise = pd.read_csv(outputs / "noise.csv")
    rows = {(r.engine, r.level, r.batch): r for r in noise.itertuples()}
    assert rows["vllm", "L1", 1].processes == 6 and rows["vllm", "L1", 8].processes == 5    # one batch-8 error
    assert rows["vllm", "L1", 8].invalid == 1 and rows["vllm", "L1", 8].reasons.startswith("noise-end-p3:")
    assert rows["llamacpp", "L2", 1].processes == 6 and rows["llamacpp", "L2", 8].processes == 5  # fragmented slots
    assert "slots that are not one run" in rows["llamacpp", "L2", 8].reasons
    assert rows["vllm", "L1", 1].occupied_processes == 6
    sent = pd.read_csv(outputs / "sentinels.csv")
    assert sorted(set(sent.position)) == ["end", "start", "unlocked"] and set(sent.redo.dropna()) == {"redo1"}
    assert (sent[sent.position == "end"].drift_ok.astype(bool)).all()      # 0.8-0.9% drift, within 3%
    reruns = pd.read_csv(outputs / "reruns.csv")
    assert list(reruns.unit) == ["vllm-0.30.0/defaults/L1/Qwen3-0.6B/timing"]  # the bring-up's 8% repeats
    assert f"{SESSION}" in set(reruns.session)
    agree = pd.read_csv(outputs / "agreement.csv")
    assert set(agree.kind) == {"pair"} and not agree.flagged.any()
    cells = tidy.load_cells(outputs / "cells.parquet")
    assert {r["engine"] for r in cells} == {"vllm", "sglang", "trtllm"}      # llama.cpp ran only noise, sentinels and
    assert all(r["session"] == SESSION for r in cells)                     # the probe: never cells
    assert {(r["engine"], r["level"]) for r in cells} >= {("vllm", "L1"), ("vllm", "L2")}  # the server's cell too
