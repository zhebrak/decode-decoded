"""anatomy.gate on the S1-shaped session (tests/s1fake.py): each item fails on the fault the session carries and
passes on the clean session, and the gate's exit code follows."""
import pytest

from anatomy import gate
from tests.s1fake import host, make_session

GRID = """extends: configs/grid/v1.yaml
validation: true
sentinels:
  - {engine: vllm, arm: defaults, level: L1, model: Qwen/Qwen3-0.6B, batches: [1]}
  - {engine: llamacpp, arm: defaults, level: L2, model: Qwen/Qwen3-0.6B, batches: [1]}
noise:
  session: S1
  processes: 3
  with_sentinels: true
  cells:
    small: {engines: [vllm, llamacpp], arms: [defaults], levels: [headline], models: [Qwen/Qwen3-0.6B],
            batches: [1, 8]}
blocks:
  bringup:
    session: S1
    engines: [vllm, sglang, trtllm]
    arms: [defaults]
    levels: [headline]
    cells: [[Qwen/Qwen3-0.6B, 1]]
    captures: {timing: all}
  hostload:
    session: S1
    engines: [vllm]
    arms: [defaults]
    levels: [headline]
    cells: [[Qwen/Qwen3-0.6B, 1]]
    captures: {graph: all, hostload: all}
"""


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    p = tmp_path_factory.mktemp("grid") / "s1.yaml"
    p.write_text(GRID)
    return p


def items(g):
    return {name: (status, detail) for name, status, detail in g.items}


def test_g1_names_every_fault_of_the_session(tmp_path, grid):
    session = make_session(tmp_path)
    host(session, locked=False)
    g = gate.run("g1", session, grid, commit="fake0000")
    got = items(g)
    assert not g.ok()
    assert got["ruler"][0] == gate.PASS
    assert "model error bare 48 ns, floored at worst 61 ns over 2 windows" in got["ruler"][1]
    assert got["clock lock"][0] == gate.FAIL and "refused" in got["clock lock"][1]
    assert got["noise block"][0] == gate.FAIL and "5 of 6 processes, 1 excluded" in got["noise block"][1]
    assert "slots that are not one run" in got["noise block"][1]
    assert got["engines within bands"][0] == gate.PASS
    assert got["cells accepted"][0] == gate.FAIL and "trtllm" in got["cells accepted"][1]  # its unit failed
    assert got["llama-server slots"][0] == gate.FAIL
    assert got["busy-time test"][0] == gate.PASS and got["hostload loops"][0] == gate.PASS
    assert got["A/A rates (info)"][0] == gate.PASS and "processes" in got["A/A rates (info)"][1]


def test_g1_passes_on_a_clean_session(tmp_path, grid):
    session = make_session(tmp_path, clean=True)
    host(session)
    g = gate.run("g1", session, grid, commit="fake0000")
    got = items(g)
    assert g.ok(), [(n, d) for n, s, d in g.items if s == gate.FAIL]
    assert got["noise block"] == (gate.PASS, "4 cells from 6 processes each")
    assert got["cells accepted"][0] == gate.PASS and got["clock lock"][1] == "1830 MHz throughout"


def test_g1_fails_when_the_pdl_chain_was_skipped(tmp_path, grid):
    session = make_session(tmp_path, clean=True, pdl=False)
    host(session)
    got = items(gate.run("g1", session, grid, commit="fake0000"))
    assert got["ruler"][0] == gate.FAIL and "skipped: Triton without launch_pdl" in got["ruler"][1]


def test_smoke_reads_the_units_the_host_and_the_probe(tmp_path, grid):
    session = make_session(tmp_path)
    host(session, help_files=False)
    got = items(gate.run("smoke", session, grid, block_session="S1", commit="fake0000"))
    assert got["units ok"][0] == gate.FAIL and "no end id" in got["units ok"][1]
    assert got["deployed commit"][0] == gate.PASS
    assert got["hostload loops"][0] == gate.PASS and got["busy-time test"][0] == gate.PASS
    assert got["trace exports"][0] == gate.PASS and got["step anchors"][0] == gate.PASS
    assert got["DRAM counters"][0] == gate.UNTESTED
    assert got["clock lock"][0] == gate.PASS
    assert got["greedy probe"][0] == gate.PASS and "none flagged" in got["greedy probe"][1]
    assert got["--help captured"] == (gate.FAIL, "0 of 4 images' --help in host/help")
    assert got["scrub"][0] == gate.PASS
    clean = make_session(tmp_path / "clean", clean=True)
    host(clean)
    got = items(gate.run("smoke", clean, grid, block_session="S1", commit="other"))
    assert got["units ok"][0] == gate.PASS and got["deployed commit"][0] == gate.FAIL
    assert got["--help captured"][0] == gate.PASS


def test_g2_reads_the_sentinels_and_the_coverage(tmp_path, grid):
    session = make_session(tmp_path)
    host(session)
    got = items(gate.run("g2", session, grid, block_session="S1"))
    assert got["sentinel drift"] == (gate.PASS, "3 closing readings within drift")  # two engines, one redo
    assert got["coverage"][0] == gate.PASS and "with a record or an explained failure" in got["coverage"][1]
    assert got["redo round"][0] == gate.UNTESTED


def test_the_cli_exits_with_the_gate(tmp_path, grid, capsys):
    session = make_session(tmp_path, clean=True)
    host(session)
    assert gate.main(["g1", "--session", str(session), "--grid", str(grid), "--commit", "fake0000"]) == 0
    out = capsys.readouterr().out
    assert "gate g1: passed" in out and out.count("pass") >= 6
    assert gate.main(["g2", "--session", str(session), "--grid", str(grid), "--block-session", "S1"]) == 0
    import shutil
    shutil.rmtree(session / "sglang-0.5.20")  # a unit the session never ran
    assert gate.main(["g2", "--session", str(session), "--grid", str(grid), "--block-session", "S1"]) == 1
    assert "coverage" in capsys.readouterr().out


def test_g2_reads_this_sessions_sentinels_only_and_the_redo_round_from_the_list(tmp_path, grid):
    """With --calibrate, tidy's sentinels.csv holds S1's readings too: G2 judges the session's own. The redo item
    takes a flagged unit as rerun only when its provenance was written after the list (anatomy.grid.redone's rule)."""
    import json
    import os
    import shutil
    import time

    session = make_session(tmp_path)
    host(session)
    other = tmp_path / "S1-other"
    shutil.copytree(session, other)
    got = items(gate.run("g2", session, grid, block_session="S1", calibrate=other))
    assert got["sentinel drift"] == (gate.PASS, "3 closing readings within drift")   # not 6
    qc = session / "qc"
    qc.mkdir()
    unit = "vllm-0.30.0/defaults/L1/Qwen3-0.6B/timing"
    (qc / "reruns.csv").write_text(f"session,unit,batches,ranges\nS1-fake,{unit},1,B=1: repeats 16%\n")
    got = items(gate.run("g2", session, grid, block_session="S1"))
    assert got["redo round"][0] == gate.FAIL and "never rerun" in got["redo round"][1]
    meta = session / unit / "meta.json"
    m = json.loads(meta.read_text())
    m["written"] = time.time() + 60   # the redo wrote the unit anew, after the list
    meta.write_text(json.dumps(m))
    os.utime(qc / "reruns.csv", (time.time() - 60, time.time() - 60))
    got = items(gate.run("g2", session, grid, block_session="S1"))
    assert got["redo round"] == (gate.PASS, "1 flagged units rerun")


def test_a_grid_of_one_session_is_judged_as_that_session(tmp_path, grid):
    """The smoke grid's blocks and noise block are session SMOKE: without --block-session the gate reads them as
    S1's would be read, never "untested" because the grid has no S1."""
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(grid.read_text().replace("session: S1", "session: SMOKE"))
    session = make_session(tmp_path, clean=True)
    host(session)
    got = items(gate.run("g1", session, smoke, commit="fake0000"))
    assert got["noise block"][0] == gate.PASS and got["cells accepted"][0] == gate.PASS
    v1 = items(gate.run("g1", session, "configs/grid/v1.yaml", block_session="S3", commit="fake0000"))
    assert v1["noise block"][0] == gate.FAIL and "no noise block for session S3" in v1["noise block"][1]
