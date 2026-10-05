"""Regressions from the pre-SXM review: evidence must survive retries and cannot silently pass missing or mixed."""
import json
import shutil
import sqlite3
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from anatomy import gate, grid, tidy
from anatomy.calibration import calibrations, ruler_calibration
from anatomy.decompose import decompose, exposed_launch_ms, suggested_overlap_model
from anatomy.noise import sentinel_drift
from anatomy.report.derive import Noise
from ruler import run as ruler_run
from servebench.nsys.scrub import leaks, scrub
from servebench.results import read_jsonl, read_meta, read_status, update_meta
from servebench.stream_client import Wave
from tests.nsysfake import fake_trace
from tests.s1fake import CLOCKS, MODEL, SESSION, STEP, graph_unit, host, jsonl, llama_log, make_session, ruler, unit
from tests.test_grid import mounts
from tests.test_served import CMDS, Shell
from tests.test_served import unit as served_unit
from tests.test_tidy import ruler_files


def test_scrub_replaces_an_unwritable_export_without_changing_its_trace(tmp_path):
    path = fake_trace(tmp_path / "export.sqlite", kernels=[(0, 10, "kernel")])
    with sqlite3.connect(path) as db:
        db.execute("insert into StringIds values (99, 'SSH_CONNECTION=canary')")
    path.chmod(0o444)
    before = path.stat().st_ino
    assert scrub(path) == 1 and not leaks(path)
    assert path.stat().st_ino != before
    with sqlite3.connect(path) as db:
        assert db.execute("select value from StringIds where id=99").fetchone() == ("",)
        assert db.execute("select end from CUPTI_ACTIVITY_KIND_KERNEL").fetchone() == (10,)
    assert scrub(path) == 0
    with pytest.raises(sqlite3.OperationalError):
        scrub(tmp_path / "missing.sqlite")
    assert not (tmp_path / "missing.sqlite").exists()


def test_failed_ruler_rerun_cannot_lend_the_previous_calibration(tmp_path, monkeypatch):
    session = tmp_path / SESSION
    ruler(session)
    assert ruler_calibration([session])
    monkeypatch.setenv("SESSION", SESSION)
    monkeypatch.setenv("DA_NSYS", "/opt/nsys")
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr(ruler_run.env, "gpu_query", lambda *a, **kw: {"name": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setattr(ruler_run.env, "host", lambda: {})
    monkeypatch.setattr(ruler_run.gpu, "lock_clocks", lambda mhz: True)
    unlocked = []
    monkeypatch.setattr(ruler_run.gpu, "unlock_clocks", lambda: unlocked.append(True))

    def failed(*args):
        raise SystemExit("measurement failed")
    monkeypatch.setattr(ruler_run, "measure", failed)
    with pytest.raises(SystemExit, match="measurement failed"):
        ruler_run.main([])
    assert not ruler_calibration([session]) and unlocked
    assert (session / "ruler.attempt-1" / "ruler_report.json").exists()
    assert not (session / "ruler" / "ruler_report.json").exists()


def small_grid(tmp_path, captures):
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump({"extends": "configs/grid/v1.yaml", "sentinels": [], "noise": None,
                                   "blocks": {"main": {"session": "S1", "engines": ["vllm"], "arms": ["defaults"],
                                                       "levels": ["L1"], "models": [MODEL], "batches": [1],
                                                       "captures": captures}}}))
    return grid.load(path)


@pytest.mark.parametrize("fault", ["missing unit", "missing batch", "failed unit"])
def test_required_capture_evidence_cannot_pass_on_timing_alone(tmp_path, fault):
    unit(tmp_path, "vllm", "timing", (1,), "main")
    g = gate.Gate("g1")
    plan = small_grid(tmp_path, {"timing": "all", "node": "all"})
    if fault != "missing unit":
        d = graph_unit(tmp_path, "vllm", "main", STEP, int(STEP * 0.9), capture="node")
        if fault == "missing batch":
            (d / "node.1.sqlite").unlink()
        else:
            (d / "status.json").write_text(json.dumps({"ok": False, "error": "profile failed"}))
    gate.required_evidence(g, plan, "S1", tmp_path / SESSION)
    assert not g.ok() and "node" in g.items[0][2]


@pytest.mark.parametrize("contents", [None, "", CLOCKS, CLOCKS + "t,[N/A],0,0,0,0x0\n"])
def test_clock_gate_requires_usable_samples_for_every_unit(tmp_path, contents):
    d = unit(tmp_path, "vllm", "timing", (1,), "main")
    host(tmp_path / SESSION)
    if contents is None:
        (d / "clocks.csv").unlink()
    else:
        (d / "clocks.csv").write_text(contents)
    g = gate.Gate("smoke")
    gate.clock(g, tmp_path / SESSION, {"clock_mhz": 1830})
    assert not g.ok() and "no usable clock samples" in g.items[0][2]


@pytest.mark.parametrize("capture", ["graph", "node", "hostload"])
@pytest.mark.parametrize("fault", ["thermal", "shared", "clock", "revision"])
def test_trace_conditions_reject_the_anatomy(tmp_path, capture, fault):
    unit(tmp_path, "vllm", "timing", (1,), "main")
    d = graph_unit(tmp_path, "vllm", "main", STEP, int(STEP * 0.9), capture=capture)
    if fault == "thermal":
        log = d / "clocks.csv"
        log.write_text(log.read_text().replace("0x0000000000000000", "0x0000000000000020"))
    elif fault == "shared":
        (d / "gpu-processes.csv").write_text("timestamp,pid\nt,10\nt,20\n")
    elif fault == "clock":
        update_meta(d, sm_clock_lock_mhz=None)
    else:
        update_meta(d, model_revision="wrong-snapshot")
    (row,) = tidy.build([tmp_path / SESSION], bw=3e12)
    assert not row["accepted"] and f"{capture} trace:" in row["rejection"]
    assert row["capture_conditions"][capture]["problem"]


def test_metrics_use_their_own_session_and_calibration(tmp_path):
    unit(tmp_path / "timing", "vllm", "timing", (1,), "main")
    graph_unit(tmp_path / "timing", "vllm", "main", STEP, int(STEP * 0.9))
    timing = tmp_path / "timing" / SESSION
    ruler(timing)
    d = graph_unit(tmp_path / "metrics", "vllm", "metrics", STEP, int(STEP * 0.9), capture="metrics")
    trace = d / "metrics.1.sqlite"
    trace.unlink()
    fake_trace(trace, graphs=[(i * STEP, i * STEP + 9_000_000, 1) for i in range(20)],
               metrics={"DRAM Read Bandwidth [Throughput %]": [(t, 50) for t in range(0, 20 * STEP, 100_000)]})
    metrics = tmp_path / "metrics" / "S3-test"
    (tmp_path / "metrics" / SESSION).rename(metrics)
    d = next(p.parent for p in metrics.rglob("meta.json"))
    report = ruler(metrics) / "ruler_report.json"
    cal = json.loads(report.read_text())
    cal["calibration"]["dram_counter_scale"] = 2.0
    report.write_text(json.dumps(cal))
    (row,) = tidy.build([timing, metrics])
    assert row["accepted"] and row["metrics_valid"] and not row["mixed_sessions"]
    assert row["metrics_ruler_from"] == "S3-test"
    assert row["dram_bytes_per_step"] == pytest.approx(0.5 * 3.35e12 * 2 * 0.010)
    log = d / "clocks.csv"
    log.write_text(log.read_text().replace("0x0000000000000000", "0x0000000000000020"))
    (row,) = tidy.build([timing, metrics])
    assert row["accepted"] and not row["metrics_valid"] and row["metrics_problem"]
    assert row.get("dram_over_model") is None
    g = gate.Gate("smoke")
    gate.dram(g, metrics, [row])
    assert not g.ok()


def test_explicit_unlocked_ruler_is_never_lent_to_a_locked_run(tmp_path):
    for name, lock in (("S1", None), ("S2", 1980)):
        ruler_files(tmp_path / name, report=name == "S1")
        (tmp_path / name / "ruler" / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": lock}))
    own, _ = calibrations([tmp_path / "S2"], calibrate=[tmp_path / "S1"])
    assert own["S2"][1]["calibration_lent"] is False


def test_a_hostload_export_without_a_window_cannot_count_as_evidence(tmp_path):
    unit(tmp_path, "vllm", "timing", (1,), "main")
    d = graph_unit(tmp_path, "vllm", "main", STEP, int(STEP * 0.9), capture="hostload")
    jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": 1, "window_from_end": 200, "profile_steps": 16}])
    (row,) = tidy.build([tmp_path / SESSION], bw=3e12)
    assert not row["accepted"] and "hostload trace: too few" in row["rejection"]


@pytest.mark.parametrize("position", ["start", "end"])
def test_invalid_sentinel_cannot_pass_the_drift_gate(tmp_path, position):
    for pos in ("start", "end"):
        d = unit(tmp_path, "vllm", "timing", (1,), f"sentinel-{pos}")
        if pos == position:
            recs = read_jsonl(d / "driver.jsonl")
            for r in recs:
                if r["kind"] == "long":
                    r["tokens"] = [1]
            jsonl(d / "driver.jsonl", recs)
    rows = sentinel_drift([tmp_path / SESSION])
    closing = next(r for r in rows if r["position"] == "end")
    assert closing["drift_ok"] is False
    import pandas as pd
    pd.DataFrame(rows).to_csv(tmp_path / "sentinels.csv", index=False)
    g = gate.Gate("g2")
    gate.sentinels(g, tmp_path, tmp_path / SESSION)
    assert not g.ok()


def test_missing_planned_sentinel_fails_g2(tmp_path):
    plan = small_grid(tmp_path, {"timing": "all"})
    plan["sentinels"] = [{"engine": "vllm", "arm": "defaults", "level": "L1", "model": MODEL, "batches": [1]}]
    g = gate.Gate("g2")
    gate.sentinels(g, tmp_path, tmp_path / SESSION, plan)
    assert not g.ok() and "missing or invalid" in g.items[0][2]


@pytest.mark.parametrize("kind", ["smoke", "g1"])
def test_real_gate_dispatch_requires_the_planned_node_capture(tmp_path, kind):
    from tests.test_gate import GRID
    session = make_session(tmp_path, clean=True)
    host(session)
    path = tmp_path / "grid.yaml"
    path.write_text(GRID.replace("captures: {graph: all, hostload: all}",
                                 "captures: {graph: all, node: all, hostload: all}"))
    g = gate.run(kind, session, path, block_session="S1", commit="fake0000")
    assert not g.ok()
    assert any(name == "required evidence" and status == gate.FAIL and "node" in detail
               for name, status, detail in g.items)


@pytest.mark.parametrize("change", ["batches", "cpus", "clock", "revision", "pin"])
def test_resume_rejects_changed_work_without_overwriting_evidence(tmp_path, monkeypatch, change):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr(grid.env, "model_revision", lambda model: "revision-a")
    monkeypatch.setattr(grid.env, "pinned", lambda model: "revision-a")
    u = next(iter(grid.expand(small_grid(tmp_path, {"timing": "all"}))))
    calls = []
    kw = {"run": lambda *a, **k: calls.append(True) or 0, "idle": lambda: None,
          "provenance": {"sm_clock_lock_mhz": 1980}, "cpuset": "0-3"}
    assert grid.run_unit(u, SESSION, mounts, **kw)[0]
    out = tmp_path / SESSION / u.key
    before = (read_meta(out), read_status(out))
    assert grid.run_unit(u, SESSION, mounts, **kw) == (True, 0.0) and len(calls) == 1
    if change == "batches":
        u = replace(u, batches=(1, 8))
    elif change == "cpus":
        kw["cpuset"] = "4-7"
    elif change == "clock":
        kw["provenance"] = {"sm_clock_lock_mhz": None}
    else:
        monkeypatch.setattr(grid.env, "model_revision" if change == "revision" else "pinned", lambda m: "revision-b")
    named = {"batches": "unit.batches", "cpus": "commands.run", "clock": "execution.sm_clock_lock_mhz",
             "revision": "model_revision", "pin": "model_pin"}[change]
    with pytest.raises(grid.ResumeMismatch, match=f"differs from this one in {named}"):  # the field, for the operator
        grid.run_unit(u, SESSION, mounts, **kw)
    assert (read_meta(out), read_status(out)) == before and len(calls) == 1


def test_a_unit_finished_before_its_run_was_recorded_is_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr(grid.env, "model_revision", lambda model: "revision-a")
    monkeypatch.setattr(grid.env, "pinned", lambda model: "revision-a")
    u = next(iter(grid.expand(small_grid(tmp_path, {"timing": "all"}))))
    kw = {"run": lambda *a, **k: 0, "idle": lambda: None}
    grid.run_unit(u, SESSION, mounts, **kw)
    out = tmp_path / SESSION / u.key
    saved = json.loads((out / "meta.json").read_text())
    for k in ("run_spec", "run_fingerprint"):
        saved.pop(k)
    (out / "meta.json").write_text(json.dumps(saved))
    with pytest.raises(grid.ResumeMismatch, match="before its run was recorded"):
        grid.run_unit(u, SESSION, mounts, **kw)
    assert read_status(out)["ok"] and json.loads((out / "meta.json").read_text()) == saved  # untouched


def test_resume_mismatch_exits_before_qc_and_does_not_abandon_the_unit(tmp_path, monkeypatch):
    small_grid(tmp_path, {"timing": "all"})
    monkeypatch.setattr(grid.env, "host", lambda: {})
    monkeypatch.setattr(grid.env, "pin_problems", lambda models: [])

    seen = []

    def mismatch(*a, **kw):
        seen.append(True)
        raise grid.ResumeMismatch("settings changed")

    def abandoned(*a, **kw):
        pytest.fail("a completed attempt was abandoned")
    monkeypatch.setattr(grid, "run_unit", mismatch)
    monkeypatch.setattr(grid, "abandon", abandoned)
    with pytest.raises(SystemExit) as e:
        grid.main([str(tmp_path / "grid.yaml"), "--session", "S1", "--session-id", SESSION, "--run", "--no-gpu"])
    assert e.value.code == 2 and seen


def test_sentinel_noise_never_compares_different_levels():
    row = {"engine": "vllm", "arm": "defaults", "level": "L2", "model": MODEL, "batch": 8, "session": "S2"}
    sentinels = [{**row, "position": pos, "step_ms": 11.0, "valid": True} for pos in ("start", "end")]
    noise = Noise([{**row, "processes": 6, "sd": 0.006}], sentinels)
    assert noise.check([{**row, "level": "L1", "step_ms": 10.0}]) == []
    assert noise.scale == {}
    (check,) = noise.check([{**row, "step_ms": 11.0}])
    assert check["holds"] and check["level"] == "L2" and check["sd"] == 0
    noisy = [{**row, "step_ms": 12.0}]
    first = noise.check(noisy)
    assert not first[0]["holds"] and noise.check(noisy) == first
    assert Noise(sentinels=[{k: v for k, v in r.items() if k != "level"} for r in sentinels]).readings == {}


def test_empty_rebuild_replaces_sidecars_and_failed_build_preserves_generation(tmp_path, monkeypatch):
    session = make_session(tmp_path)
    out = tmp_path / "out"
    args = [str(session), "--calibrate", str(session), "--out", str(out)]
    tidy.main(args)
    (out / "README.md").write_text("keep me\n")
    before = {p.name: p.read_bytes() for p in out.iterdir()}

    def fail(*a, **kw):
        raise RuntimeError("generation failed")
    with monkeypatch.context() as m:
        m.setattr("anatomy.noise.sentinel_drift", fail)
        with pytest.raises(RuntimeError, match="generation failed"):
            tidy.main(args)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    for d in session.iterdir():
        if d.name != "ruler":
            shutil.rmtree(d)
    tidy.main(args)
    for name in ("sentinels.csv", "noise.csv", "agreement.csv"):  # an empty table is no file, and the old one is gone
        assert not (out / name).exists() and gate.read_csv(out / name) == []
    assert not (out / "tolerances-suggested.yaml").exists()
    assert (out / "README.md").read_text() == "keep me\n"


def pdl_windows(*pairs):
    return [{"pdl_model_error_ns": raw, "pdl_model_corrected_error_ns": corrected} for raw, corrected in pairs]


@pytest.mark.parametrize("windows, tick, spread, expected", [
    (pdl_windows((60, 20), (70, 10)), 8, 10, "corrected"),  # every window, by more than the resolution
    (pdl_windows((-20, 60), (10, -70)), 8, 10, "raw"),      # absolute errors: raw closer in every window
    (pdl_windows((60, 20), (10, 70)), 8, 10, None),         # the windows disagree
    (pdl_windows((25, 20), (26, 20)), 8, 10, None),         # a gap within the resolution
    (pdl_windows((60, 20)), None, 10, None),                # the %globaltimer step was not read
    (pdl_windows((60, 20)), 8, None, None),                 # one untraced pass: the truth's spread unknown
    ([], 8, 10, None)])
def test_the_overlap_model_is_chosen_only_when_every_window_clears_the_resolution(windows, tick, spread, expected):
    cal = {"pdl_windows": windows, "pdl_timer_step_ns": tick, "pdl_truth_spread_ns": spread, "pdl_truth_passes": 3}
    model, evidence = suggested_overlap_model(cal)
    assert model == expected and evidence["verdict"].startswith(expected or "undecided")
    if windows:
        assert evidence["min_gap_ns"] >= 0 and evidence["windows"] == len(windows)


def capped_log(d, held, capped=0.9, n=100):
    """clocks.csv of a cell at the power cap: `capped` of the busy samples under sw_power_cap at 1455 MHz (SGLang's
    14B at the KV bound on the pod read 0.899), the rest at `held`, what the lock holds where the cap lets go."""
    rows = [f"2026/10/02 10:{i // 60:02d}:{i % 60:02d}.000, {1455 if i < int(capped * n) else held}, 2619, 60, "
            f"{690.0 if i < int(capped * n) else 400.0}, {'0x0000000000000004' if i < int(capped * n) else '0x0'}"
            for i in range(n)]
    (d / "clocks.csv").write_text(CLOCKS + "\n".join(rows) + "\n")


def test_the_lock_is_judged_where_nothing_slowed_the_clock(tmp_path):
    session = tmp_path / SESSION
    d = unit(tmp_path, "vllm", "timing", (1,), "main", lock=1980)
    host(session)

    def judged():
        (row,) = tidy.build([session], bw=3e12)
        g = gate.Gate("smoke")
        gate.clock(g, session, {"clock_mhz": 1980})
        return row, g
    capped_log(d, 1980)  # the median of every sample sits at the cap; the lock held where the cap let go
    row, g = judged()
    assert row["accepted"] and row["sm_mhz"] == 1455 and row["sm_mhz_unthrottled"] == 1980
    assert row["slowdowns"]["sw_power_cap"] == 0.9 and g.ok() and g.items[0][2] == "1980 MHz throughout"
    capped_log(d, 1980, capped=1.0)  # capped in every busy sample: the lock cannot be judged, the cell stands
    row, g = judged()
    assert row["accepted"] and row["sm_mhz_unthrottled"] is None and g.ok() and "not judged" in g.items[0][2]
    capped_log(d, 1700)  # the lock did not hold where it could have
    row, g = judged()
    assert not row["accepted"] and "1700 MHz where nothing slowed it" in row["rejection"]
    assert not g.ok() and "1700 MHz where nothing slowed it" in g.items[0][2]


def test_a_card_that_holds_less_than_the_lock_asked_is_read_against_what_its_ruler_held(tmp_path):
    """The H100 PCIe VM: the driver took a lock at 1755 MHz and the card held 1620, idle and busy alike, no slowdown
    bit set; its ruler ran at the same 1620, so the cells are read against that, and the lock asked for is a note."""
    from ruler import validate
    session = tmp_path / SESSION
    d = unit(tmp_path, "vllm", "timing", (1,), "main", lock=1755)
    host(session)
    r = ruler(session)
    (r / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": 1755}))
    capped_log(r, 1620, capped=0.0)  # the ruler's own clock log
    report = json.loads((r / "ruler_report.json").read_text())
    report["card"] = validate.card(r)
    (r / "ruler_report.json").write_text(json.dumps(report))
    assert report["card"]["sm_mhz_held"] == 1620 and ruler_calibration([session])["ruler_clock_held_mhz"] == 1620
    assert tidy.clock_reference([session]) == {SESSION: 1620}
    capped_log(d, 1620, capped=0.0)
    (row,) = tidy.build([session], bw=3e12)
    assert row["accepted"] and row["sm_mhz_unthrottled"] == 1620 and row["sm_clock_lock_mhz"] == 1755
    assert row["ruler_clock_held_mhz"] == 1620
    g = gate.Gate("smoke")  # a validation grid notes the card's clock; the study's grid fails on it: its page would
    gate.clock(g, session, {"clock_mhz": 1755, "validation": True})  # name a clock the cells did not run at
    assert g.ok() and "asked 1755 MHz and the card held 1620" in g.items[0][2]
    g = gate.Gate("smoke")
    gate.clock(g, session, {"clock_mhz": 1755})
    assert not g.ok() and "holds 1620 MHz of the 1755 asked" in g.items[0][2]
    report["ok"] = False  # a ruler whose report failed still read the card's clock: tidy and the gate agree on it
    (r / "ruler_report.json").write_text(json.dumps(report))
    (row,) = tidy.build([session], bw=3e12)
    assert row["accepted"] and tidy.clock_reference([session]) == {SESSION: 1620}
    (r / "clocks.csv").unlink()  # and without the log, the report's card still carries what it held
    assert tidy.clock_reference([session]) == {SESSION: 1620}
    capped_log(r, 1620, capped=0.0)
    capped_log(d, 1755, capped=0.0)  # a unit at the clock asked for, where the ruler held 1620: not the same clock
    (row,) = tidy.build([session], bw=3e12)
    assert not row["accepted"] and "expected 1620 MHz (what the ruler held)" in row["rejection"]
    g = gate.Gate("smoke")
    gate.clock(g, session, {"clock_mhz": 1755, "validation": True})
    assert not g.ok() and "expected 1620 MHz (what the ruler held)" in g.items[0][2]
    s = unit(tmp_path, "vllm", "timing", (1,), "sentinel-start", lock=1755)  # the sentinel and noise readers share
    capped_log(s, 1620, capped=0.0)                                            # the reference
    (opening,) = sentinel_drift([session])
    assert opening["valid"] and opening["problem"] is None


def test_a_fragmented_traced_wave_reaches_the_slots_gate(tmp_path):
    session = tmp_path / SESSION
    unit(tmp_path, "llamacpp", "timing", (8,), "main")
    d = graph_unit(tmp_path, "llamacpp", "main", STEP, int(STEP * 0.9), batches=(8,))
    (d / "container.log").write_text(llama_log([8], waves=1, fragment=(8,)))  # the traced wave alone ran fragmented
    (row,) = tidy.build([session], bw=3e12)
    assert not row["accepted"] and row["slots_problem"].startswith("graph trace: ")
    g = gate.Gate("g1")
    gate.slots(g, session, [row], [])
    assert not g.ok() and "graph trace" in g.items[0][2]
    g = gate.Gate("g1")  # a noise reading whose occupancy was excluded for its slots
    gate.slots(g, session, [{**row, "slots_problem": None}],
               [{"engine": "llamacpp", "level": "L2", "batch": 8, "reasons": "",
                 "occupied_reasons": "llama-server ran batch 8 in slots that are not one run"}])
    assert not g.ok() and "noise L2 b8" in g.items[0][2]


def test_a_pdl_launch_is_charged_at_least_its_release():
    """The H100 PCIe's chain: a dependent launch leaves 535 ns exposed however early it started, where the bare model
    charged an overlap past the node cost nothing."""
    kw = dict(kernels=10, node_floor_us=1.0, overlaps_ns=[2000] * 5 + [300] * 5, steps=1)
    assert exposed_launch_ms(**kw) == pytest.approx(5 * 700 / 1e6)  # bare: past the floor, nothing
    assert exposed_launch_ms(**kw, release_ns=535) == pytest.approx((5 * 535 + 5 * 700) / 1e6)
    dec = decompose(step_ms=10, occupied=0.9, kernel_ms=8, floor_ms=1, kernels_per_step=10, node_floor_us=1.0,
                    inflation_ns=0, overlaps_ns=[2000] * 10, steps=1, release_ns=535)
    assert dec["launch_floor_ms"] == pytest.approx(10 * 535 / 1e6) and dec["launch_release_ns"] == 535
    bare = decompose(step_ms=10, occupied=0.9, kernel_ms=8, floor_ms=1, kernels_per_step=10, node_floor_us=1.0,
                     inflation_ns=0, overlaps_ns=[2000] * 10, steps=1)
    assert bare["launch_floor_ms"] == 0 and bare["launch_release_ns"] is None
    assert exposed_launch_ms(1, 1.129, [2000], 1, 0, -40) == 0  # a release below zero is no release, never a credit


def test_the_overlap_choice_reads_the_floored_errors_and_skips_windows_charged_alike():
    def floored(raw, corr, bare=(0, 0)):
        return {"pdl_model_error_ns": bare[0], "pdl_model_corrected_error_ns": bare[1],
                "pdl_model_floored_error_ns": raw, "pdl_model_floored_corrected_error_ns": corr}
    cal = {"pdl_timer_step_ns": 32, "pdl_truth_spread_ns": 2, "pdl_truth_passes": 3}
    # on the plateau both models charge the release: the PCIe's windows at 1, 2 and 4 us, and its 0.5 us one too
    model, evidence = suggested_overlap_model({**cal, "pdl_windows": [floored(-139, -139, (-441, -324)),
                                                                     floored(0, 0, (-535, -535))]})
    assert model is None and evidence["informative"] == 0 and "alike" in evidence["verdict"]
    # one window below the plateau where the models part: it alone decides, the bare errors never read
    model, evidence = suggested_overlap_model({**cal, "pdl_windows": [floored(-200, -100, (-441, -324)),
                                                                     floored(0, 0)]})
    assert model == "corrected" and evidence["informative"] == 1 and evidence["windows"] == 2


def test_raw_overlap_keeps_the_kernel_duration_correction():
    kw = {"step_ms": 10, "occupied": 0.9, "kernel_ms": 8, "floor_ms": 1, "kernels_per_step": 10,
          "node_floor_us": 1, "inflation_ns": 100, "overlaps_ns": [500] * 10, "steps": 1}
    raw, corrected = (decompose(**kw, overlap_model=m) for m in ("raw", "corrected"))
    assert raw["kernel_ms_corrected"] == corrected["kernel_ms_corrected"] == pytest.approx(7.999)
    assert raw["launch_floor_ms"] == pytest.approx(0.005)
    assert corrected["launch_floor_ms"] == pytest.approx(0.006)


def test_wave_deadline_cancels_a_dripping_partial_sse_line(tmp_path):
    connected, closed = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.end_headers()
            if body["max_tokens"] == 96:  # the served unit's warm-up succeeds; its profiled wave hangs
                self.wfile.write(b"data: [DONE]\n\n")
                return
            connected.set()
            try:
                while True:
                    self.wfile.write(b" ")  # activity forever, but never a complete line
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                closed.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wave = Wave(f"http://127.0.0.1:{server.server_port}/v1/completions", [[1]], 10).start(timeout=0.2)
        assert connected.wait(1)
        with pytest.raises(TimeoutError, match="deadline"):
            wave.join()
        assert all(not t.is_alive() for t in wave.threads) and closed.wait(1)
        from anatomy.served import run_served
        cmds = {**CMDS, "ready": f"http://127.0.0.1:{server.server_port}/health"}
        started = time.monotonic()
        error = run_served(served_unit(capture="graph"), cmds, tmp_path, sh=Shell(),
                           ready=lambda *a, **k: True, timeout=0.2)
        assert "deadline passed" in error and time.monotonic() - started < 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(1)


def node_trace(d, name, step_ns=STEP, kernels_per_step=10, overlap_ns=2000, busy_ns=500_000):
    """A node capture of 20 steps: one graph launch a step and kernels_per_step kernels of 50 us, each but the first
    starting overlap_ns before its predecessor ends, as llama.cpp's programmatic dependent launches do."""
    graphs, kernels = [], []
    for j in range(20):
        t = t0 = j * step_ns
        graphs.append((t0, t0 + busy_ns, 1))
        for k in range(kernels_per_step):
            start = t - overlap_ns if k else t
            kernels.append((start, start + 50_000, "kernel", {"graphNodeId": k, "graphId": 1}))
            t = start + 50_000
    fake_trace(d / name, kernels=kernels, graphs=graphs)


def overlapping_session(root, with_ruler=True, pdl=True):
    """A session of one vLLM cell with a timing, a graph and a node capture whose kernels overlap, and a ruler or none:
    where the PDL release reaches the cells through tidy."""
    unit(root, "vllm", "timing", (1,), "main")
    graph_unit(root, "vllm", "main", STEP, int(STEP * 0.9))
    d = unit(root, "vllm", "node", (1,), "main")
    jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": 1, "window_from_end": 18, "profile_steps": 16}])
    node_trace(d, "node.1.sqlite")
    if with_ruler:
        ruler(root / SESSION, pdl=pdl)
    return root / SESSION


def test_the_release_prices_the_cells_and_is_lent_or_borrowed_with_its_source_named(tmp_path):
    own = overlapping_session(tmp_path / "own")
    (row,) = tidy.build([own], bw=3e12)
    assert row["kernels_per_step"] == 10 and row["launch_release_ns"] == 310.0 and row["launch_release_from"] == SESSION
    # 9 overlapped kernels at the release (their 1,895 ns of corrected overlap passes the 980 ns node), one at the node
    floor_ms = (9 * 310 + 980) / 1e6
    assert row["launch_floor_ms"] == pytest.approx(floor_ms)
    lo, hi = row["launch_floor_share_range"]  # the range spans the floor's own range, the release's absence and the
    assert lo == pytest.approx(790 / 1e6 / row["step_ms"])  # widest node's release: its low end one bare kernel at the
    assert hi == pytest.approx((9 * 400 + 980) / 1e6 / row["step_ms"])  # floor's low end, its high end the 400 ns
    assert row["launch_release_wide_ns"] == 400.0  # the 264-block node's, carried for the record
    # a session that measured nothing of its own borrows S1's calibration whole, the release with it
    other = overlapping_session(tmp_path / "other", with_ruler=False)
    other = other.rename(other.with_name("S2-fake"))
    (lent,) = tidy.build([other], bw=3e12, calibrate=[own])
    assert lent["launch_release_ns"] == 310.0 and lent["launch_release_from"] == SESSION
    assert lent["launch_release_wide_ns"] == 400.0
    # an own ruler that passed without the chain's release takes S1's, named: never the bare model unannounced
    third = overlapping_session(tmp_path / "third", pdl=False)
    third = third.rename(third.with_name("S3-fake"))
    (borrowed,) = tidy.build([third], bw=3e12, calibrate=[own])
    assert borrowed["ruler_from"] == "S3-fake" and borrowed["launch_release_ns"] == 310.0
    assert borrowed["launch_release_from"] == SESSION and borrowed["launch_release_wide_ns"] == 400.0
    assert tidy.build([third], bw=3e12)[0]["launch_release_ns"] is None  # alone, there is none to take


def test_a_stalled_prefill_marks_the_wave_and_excludes_the_reading(tmp_path):
    """The PCIe VM: one fresh llama-server evaluated 896-token prompts in 1.7-2.0 s against 100 ms in four waves; the
    median cadence never saw it, the span differential did (10.3 ms a step against 4.7)."""
    from anatomy.llama_log import llama_prefill
    session = tmp_path / SESSION
    d = unit(tmp_path, "llamacpp", "timing", (8,), "main")
    (d / "container.log").write_text(llama_log([8], stall=((8, 1),)))  # the second measured wave's first request
    assert llama_prefill(d, read_jsonl(d / "driver.jsonl")) == {
        8: "llama-server's prefill stalled: 2000 ms for a request of a measured wave against the server's median 100"}
    (row,) = tidy.build([session], bw=3e12)
    assert row["accepted"] and row["prefill_problem"].startswith("llama-server's prefill stalled")  # the cell stands:
    # its range rule sends the unit to the redo round; a noise or sentinel reading is excluded with the reason
    s = unit(tmp_path, "llamacpp", "timing", (1,), "sentinel-start")
    (s / "container.log").write_text(llama_log([1], stall=((1, 0),)))
    (opening,) = sentinel_drift([session])
    assert not opening["valid"] and "prefill stalled" in opening["problem"]
    g = gate.Gate("g1")
    gate.slots(g, session, [{**row, "slots_problem": None}],
               [{"engine": "llamacpp", "level": "L2", "batch": 1, "reasons": opening["problem"],
                 "occupied_reasons": ""}])
    assert not g.ok() and "prefill stalled" in g.items[0][2]
