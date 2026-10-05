import json

import pytest

from anatomy import calibration
from anatomy.metrics import rejection
from anatomy.tidy import build, load_cells, write
from servebench.results import Unit, unit_dir, write_meta, write_status
from tests.nsysfake import fake_trace

STEP = 10_000_000  # 10 ms decode steps


def make_unit(root, capture, **kw):
    u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture=capture,
             batches=(64,), **kw)
    d = unit_dir("S2-test", u, root=root)
    write_meta(d, u)
    write_status(d, True, 1.0)
    return d


def jsonl(path, recs):
    path.write_text("".join(json.dumps(wave(r)) + "\n" for r in recs))


def wave(r):
    """A wave record as the drivers write it: every sequence's tokens and the wave's gen_len, filled in where a test
    does not set them."""
    if r.get("kind") not in ("long", "short") or "tokens" in r:
        return r
    gen = r.get("gen_len", 288 if r["kind"] == "long" else 96)
    return {**r, "gen_len": gen, "tokens": [gen] * r["batch"]}


def test_one_cell_end_to_end(tmp_path):
    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    cadence = {"step_ns": STEP, "interval_ns": STEP}
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i, "cadence": cadence} for i in range(3)]
          + [{"kind": "differential", "batch": 64, "step_ns": [STEP]}])
    prof = [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}]
    g = make_unit(tmp_path, "graph")
    jsonl(g / "driver.jsonl", prof)
    # 20 steps: a 8 ms graph replay then 1 ms of eager sampling, then 1 ms waiting on the host
    fake_trace(g / "graph.1.sqlite", graphs=[(i * STEP, i * STEP + 8_000_000, 1) for i in range(20)],
               kernels=[(i * STEP + 8_000_000, i * STEP + 9_000_000, "argmax") for i in range(20)])
    n = make_unit(tmp_path, "node")
    jsonl(n / "driver.jsonl", prof)
    kernels = []
    for i in range(20):  # inside the graph: 4 kernels of 1.9 ms with 0.1 ms gaps (7.6 ms of kernels in 8 ms)
        kernels += [(i * STEP + j * 2_000_000, i * STEP + j * 2_000_000 + 1_900_000, "gemm",
                     {"graphNodeId": j, "graphId": 1, "correlationId": i}) for j in range(4)]
        kernels.append((i * STEP + 8_000_000, i * STEP + 9_000_000, "argmax"))
    fake_trace(n / "node.1.sqlite", kernels=kernels)
    (session / "ruler").mkdir()
    (session / "ruler" / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": 3.1e12}))

    rows = build([session])
    assert len(rows) == 1
    r = rows[0]
    assert r["step_ms"] == pytest.approx(10) and r["estimators_agree"] and r["session"] == "S2-test"
    assert r["occupied"] == pytest.approx(0.9) and r["profiled_step_ms"] == pytest.approx(10)
    assert r["profiler_overhead"] == pytest.approx(0) and r["overhead_ok"]
    assert r["occupied_source"] == "trace"  # the steps agree: one process's own fraction
    assert "host_stall_resolved" not in r and "occupied_noise" not in r  # the noise is derive's, per cell
    assert r["kernel_ms"] == pytest.approx(8.6)  # 7.6 ms of GEMMs + 1 ms sampling per step
    assert r["host_stall_ms"] == pytest.approx(1.0) and r["bubbles_ms"] == pytest.approx(0.4)
    assert r["launch_ms"] == pytest.approx(0.4) and r["launch_model"] is False  # no ruler calibration: gaps only
    assert r["floor_ms"] == pytest.approx(r["bytes_per_step"] / 3.1e12 * 1e3) and r["consistent"]
    assert set(r["sources"]) == {"timing", "graph", "node"}
    cal = {"graph_node_floor_us": 1.0, "graph_node_floor_range_us": [0.8, 1.0]}
    r = build([session], calibration=cal)[0]  # 5 kernels a step, none under its predecessor: 4-5 us of a 10 ms step
    assert r["overlapped_per_step"] == 0 and r["launch_floor_share_range"] == pytest.approx([4e-4, 5e-4])

    rows_fast = build([session], bw=1e15)  # a floor far below the kernels: every term non-negative, nothing flagged
    assert rows_fast[0]["consistent"] is True and rows_fast[0]["kernel_exc_ms"] > 0

    df = write(rows, tmp_path / "results")
    assert (tmp_path / "results" / "cells.csv").exists() and df.loc[0, "batch"] == 64


def timing_unit(root, session, step_ns, block="main", batches=(64,)):
    u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture="timing",
             batches=batches, block=block)
    d = unit_dir(session, u, root=root)
    write_meta(d, u)
    write_status(d, True, 1.0)
    cadence = {"step_ns": step_ns, "interval_ns": step_ns}
    jsonl(d / "driver.jsonl", [{"kind": "long", "batch": b, "repeat": i, "cadence": cadence} for b in batches
                               for i in range(3)])
    return d


def test_sentinels_never_become_cells_and_a_later_measurement_supersedes(tmp_path):
    """S3's unlocked sentinel never replaces S2's numbers; a cell S3 measured again (a rerun) does, S2's copy kept as a
    duplicate."""
    first = timing_unit(tmp_path, "S2", 10_000_000)
    timing_unit(tmp_path, "S3", 12_000_000, block="sentinel-unlocked")
    timing_unit(tmp_path, "S3", 11_000_000, block="served")
    rows = build([tmp_path / "S2", tmp_path / "S3"], bw=3.1e12)
    assert len(rows) == 1 and rows[0]["step_ms"] == pytest.approx(11) and rows[0]["session"] == "S3"
    assert rows[0]["duplicate_sources"] == [f"S2/{first.relative_to(tmp_path / 'S2')}"]  # no machine's paths
    from anatomy.noise import sentinel_drift
    drift = sentinel_drift([tmp_path / "S2", tmp_path / "S3"])
    assert [(d["position"], d["step_ms"]) for d in drift] == [("unlocked", pytest.approx(12))]


def test_an_anchor_that_fires_three_times_a_step_is_caught(tmp_path):
    """llama.cpp's get_rows may run three times a step: anchors then count 3x the steps the driver saw."""
    from anatomy.metrics import accepted
    timing_unit(tmp_path, "S2", 10_000_000)
    u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture="graph",
             batches=(64,))
    d = unit_dir("S2", u, root=tmp_path)
    write_meta(d, u)
    write_status(d, True, 1.0)
    jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 176, "profile_steps": 64}])
    third = STEP // 3
    fake_trace(d / "graph.1.sqlite", graphs=[(i * third, i * third + third // 2, 1) for i in range(3 * 64 + 1)])
    r = build([tmp_path / "S2"], bw=3.1e12)[0]
    assert r["anchor_ok"] is False and not accepted(r)


def test_dram_counters_check_the_byte_model(tmp_path):
    """A metrics capture whose counters saw exactly the model's bytes per step reads dram_over_model 1."""
    from anatomy.metrics import physics
    from servebench.physics import H100_SXM

    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    cadence = {"step_ns": STEP, "interval_ns": STEP}
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i, "cadence": cadence} for i in range(3)])
    m = make_unit(tmp_path, "metrics")
    jsonl(m / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    want = physics("Qwen/Qwen3-8B", 64, 896, 288, 10.0)["bytes_per_step"]
    pct = want / (STEP / 1e9) / H100_SXM.bandwidth * 100   # the read throughput that moves exactly those bytes
    fake_trace(m / "metrics.1.sqlite", graphs=[(i * STEP, i * STEP + 9_000_000, 1) for i in range(20)],
               metrics={"DRAM Read Bandwidth [Throughput %]": [(t, pct) for t in range(0, 20 * STEP, 100_000)]})
    r = build([session], bw=3.1e12, calibration={"dram_counter_scale": 1.0})[0]
    assert r["dram_over_model"] == pytest.approx(1.0, rel=1e-3)
    r = build([session], bw=3.1e12, calibration={})[0]   # the ruler never read the counters: no ratio, never 1
    assert r.get("dram_over_model") is None and r["dram_pct"] is not None


@pytest.mark.parametrize("busy_ns, profiled_ns, occupied, ok", [
    (9_000_000, 11_000_000, 0.9, True),    # tracing cost the host 1 ms a step: the GPU's 9 ms of work is unchanged
    (10_500_000, 11_000_000, 10.5 / 11, False),  # more GPU work per step than the whole unprofiled step: two
                                                 # processes differing, the trace's own fraction, and rejected
    (10_200_000, 10_250_000, 10.2 / 10.25, True),  # stretched but GPU-bound in its own trace (SGLang at batch 64 on
                                                   # a power-capped card): busy over the untraced step would read 1.02
    (9_000_000, 9_800_000, 9 / 9.8, True),  # the profiled process 2% faster: two processes differing, not tracing,
                                            # so the trace's own fraction, never a host stall made of the difference
    (9_000_000, 9_500_000, 9 / 9.5, False),  # 5% faster: not the same conditions, rejected
])
def test_occupancy_is_the_trace_fraction_unless_tracing_stretched_the_step(tmp_path, busy_ns, profiled_ns, occupied,
                                                                             ok):
    from anatomy.metrics import accepted

    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    cadence = {"step_ns": STEP, "interval_ns": STEP}
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i, "cadence": cadence} for i in range(3)])
    g = make_unit(tmp_path, "graph")
    jsonl(g / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    fake_trace(g / "graph.1.sqlite", graphs=[(i * profiled_ns, i * profiled_ns + busy_ns, 1) for i in range(20)])
    r = build([session], bw=3.1e12, calibration={})[0]
    assert r["occupied"] == pytest.approx(occupied)
    assert r["occupied_trace"] == pytest.approx(busy_ns / profiled_ns)
    assert r["trace_consistent"] is ok and accepted(r) is ok


def test_trace_strip_from_a_node_trace(tmp_path):
    from anatomy.strip import strip
    kernels = []
    for i in range(14):  # 14 steps of 10 ms: 4 graph kernels sharing a correlation id, then an eager argmax
        kernels += [(i * STEP + j * 2_000_000, i * STEP + j * 2_000_000 + 1_900_000, "gemm",
                     {"graphNodeId": j, "graphId": 1, "correlationId": i}) for j in range(4)]
        kernels.append((i * STEP + 8_000_000, i * STEP + 9_000_000, "argmax"))
    path = fake_trace(tmp_path / "node.1.sqlite", kernels=kernels)
    cut = strip(path)
    ticks = [r["start"] for r in cut if r["kind"] == "step"]
    assert ticks == [0.0, 10_000.0, 20_000.0, 30_000.0]       # three whole steps, in us from the first
    assert sum(r["kind"] == "kernel" for r in cut) == 15       # 5 kernels a step
    assert {r["class"] for r in cut if r["kind"] == "kernel"} == {"gemm", "sampling"}
    # the cell's own window: counted back from the capture's last anchor, as tidy places it (window_from_end less the
    # offset), with the arm's anchor; the argmax marks every step too, so the same cut comes out either way
    from anatomy.strip import cut_of
    assert cut_of({"anchor": "argmax", "window_from_end": 6, "trace_offset_steps": 1}) == ("argmax", -5)
    assert cut_of({}) == ("graph", 8)                          # an older table: the graph anchor, SKIP steps in
    back = strip(path, "argmax", skip=-5)
    assert [r["start"] for r in back if r["kind"] == "step"] == [0.0, 10_000.0, 20_000.0, 30_000.0]
    assert [r["step"] for r in back if r["kind"] == "step"] == [0, 1, 2, 3]
    assert strip(path, "argmax", skip=-2) == []                # too few whole steps left after it


def test_anchors_are_checked_by_their_period_against_the_unprofiled_step(tmp_path):
    """No token count enters: an L2 trace whose anchors fire once a step passes, twice a step fails."""
    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                "cadence": {"step_ns": STEP, "interval_ns": STEP}} for i in range(3)])
    g = make_unit(tmp_path, "graph")
    jsonl(g / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 34, "profile_steps": 32}])
    fake_trace(g / "graph.1.sqlite", graphs=[(i * STEP, i * STEP + 8_000_000, 1) for i in range(40)])
    assert build([session], bw=3.1e12, calibration={})[0]["anchor_ok"]
    (g / "graph.1.sqlite").unlink()
    fake_trace(g / "graph.1.sqlite", graphs=[(i * STEP // 2, i * STEP // 2 + 4_000_000, 1) for i in range(80)])
    r = build([session], bw=3.1e12, calibration={})[0]
    assert r["anchor_ok"] is False and "anchors do not mark the steps" in rejection(r)

def test_l2_traces_are_found_by_exact_batch(tmp_path):
    from anatomy.tidy import windows
    for b in (1, 16):
        (tmp_path / f"graph-b{b}.sqlite").write_text("")
    got = windows(tmp_path, "graph", "L2", (1, 16, 64))
    assert got == {1: tmp_path / "graph-b1.sqlite", 16: tmp_path / "graph-b16.sqlite"}  # not b16 for b1; no b64


def test_an_ok_rerun_beats_a_failed_first_attempt(tmp_path):
    def timing(session, step_ns, ok):
        u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B",
                 capture="timing", batches=(64,))
        d = unit_dir(session, u, root=tmp_path)
        write_meta(d, u)
        write_status(d, ok, 1.0)
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                    "cadence": {"step_ns": step_ns, "interval_ns": step_ns}} for i in range(3)])
    timing("S2-a", 14_000_000, False)   # timed out after a bad run
    timing("S3-b", 10_000_000, True)    # the rerun
    r = build([tmp_path / "S2-a", tmp_path / "S3-b"], bw=3.1e12, calibration={})[0]
    assert r["step_ms"] == pytest.approx(10.0) and r["duplicate_sources"][0].startswith("S2-a/")


def test_cadence_is_estimated_again_from_the_raw_arrivals(tmp_path):
    """A record written by an older estimator (a 3-step jitter counted as a stall) is read by the current one."""
    import gzip
    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    seq = [(i * STEP + (5 * STEP if i > 150 else 0), i) for i in range(1, 289)]   # one 6-step hiccup at batch 1
    for r in range(3):
        with gzip.open(t / f"chunks-b64-r{r}.json.gz", "wt") as f:
            json.dump([seq], f)
    stale = {"step_ns": STEP, "interval_ns": STEP, "stalls": 1}
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": r, "cadence": stale} for r in range(3)])
    row = build([session], bw=3.1e12, calibration={})[0]
    assert row["stalls"] == 0 and row["valid"]


@pytest.mark.parametrize("first, ok", [(0, True), (100, True), (150, False)])
def test_the_window_is_counted_back_from_the_waves_last_step(tmp_path, first, ok):
    """A capture spans the wave; the window is the 64 steps 176 back from its last. A capture that opened late still
    has them as long as it holds 176 anchors; one that does not is rejected, never read at another context."""
    from anatomy.tidy import window_summary
    graphs = [(i * STEP, i * STEP + 9_000_000, 1) for i in range(first, 287)]  # decode steps of a 288-token wave
    path = fake_trace(tmp_path / "graph.1.sqlite", graphs=graphs)
    s = window_summary(path, {"window_from_end": 176, "profile_steps": 64}, "graph")
    if ok:
        assert s["steps"] == 64 and s["window_ns"] == [111 * STEP, 175 * STEP]  # the same steps, however late it opened
    else:
        assert s["steps"] is None


@pytest.mark.parametrize("graph_period, hostload_period, invariant", [
    (10_000_000, 12_000_000, True), (10_000_000, 10_100_000, None),
    (11_000_000, 11_500_000, None),  # tracing stretched 10%, the hostload trace 4.5% more: no test
    (11_000_000, 13_000_000, True)])
def test_a_hostload_check_that_did_not_stretch_the_host_tests_nothing(tmp_path, graph_period, hostload_period,
                                                                      invariant):
    session = tmp_path / "S2-test"
    t = make_unit(tmp_path, "timing")
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                "cadence": {"step_ns": STEP, "interval_ns": STEP}} for i in range(3)])
    for capture, period in (("graph", graph_period), ("hostload", hostload_period)):
        d = make_unit(tmp_path, capture)
        jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": 64}])
        fake_trace(d / f"{capture}.1.sqlite", graphs=[(i * period, i * period + 9_000_000, 1) for i in range(20)])
    r = build([session], bw=3.1e12, calibration={})[0]
    assert r["busy_invariant"] is invariant   # a 1% stretch is not a test of the assumption


def test_the_noise_table_is_the_spread_between_processes_and_never_cells(tmp_path):
    from anatomy.noise import noise_table
    for n, step in ((1, 10_000_000), (2, 10_100_000), (3, 9_900_000)):
        d = make_unit(tmp_path, "timing", block=f"noise-start-p{n}")
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                    "cadence": {"step_ns": step, "interval_ns": step}} for i in range(3)])
    for n, step in ((1, 10_200_000), (2, 10_300_000), (3, 10_100_000)):   # the session's end: 2% slower
        d = make_unit(tmp_path, "timing", block=f"noise-end-p{n}")
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                    "cadence": {"step_ns": step, "interval_ns": step}} for i in range(3)])
    rows = noise_table([tmp_path / "S2-test"])
    assert len(rows) == 1 and rows[0]["processes"] == 6    # both ends pooled: drift over the session is noise too
    assert rows[0]["start_to_end"] == pytest.approx(0.02) and rows[0]["sd"] == pytest.approx(0.0140, abs=2e-4)
    assert build([tmp_path / "S2-test"], bw=3.1e12, calibration={}) == []


def test_calibration_sessions_give_the_ruler_and_never_cells(tmp_path):
    """S1 runs the ruler and the noise block; S2 holds the main grid. --calibrate S1 reads S1's ruler, and S1's own
    bring-up cell never displaces S2's; without any ruler tidy refuses, rather than drop every floor and MBU."""
    from anatomy.tidy import main
    for session, step in (("S1-test", 9_000_000), ("S2-test", STEP)):
        u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture="timing",
                 batches=(64,))
        d = unit_dir(session, u, root=tmp_path)
        write_meta(d, u)
        write_status(d, True, 1.0)
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                    "cadence": {"step_ns": step, "interval_ns": step}} for i in range(3)])
    with pytest.raises(SystemExit, match="--calibrate"):
        main([str(tmp_path / "S2-test"), "--out", str(tmp_path / "out")])
    (tmp_path / "S1-test" / "ruler").mkdir()
    (tmp_path / "S1-test" / "ruler" / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": 3.1e12}))
    main([str(tmp_path / "S2-test"), "--calibrate", str(tmp_path / "S1-test"), "--out", str(tmp_path / "out")])
    rows = load_cells(tmp_path / "out" / "cells.parquet")
    assert len(rows) == 1 and rows[0]["session"] == "S2-test" and rows[0]["step_ms"] == pytest.approx(10)
    assert rows[0]["bw_achievable"] == pytest.approx(3.1e12)


def test_an_engine_s_own_gpu_processes_are_not_sharing(tmp_path):
    """TensorRT-LLM's LLM-API process and its MPI worker both hold a GPU context: two processes are its own; a third
    is another job."""
    from anatomy.tidy import own_processes, unit_conditions
    d = make_unit(tmp_path, "timing")
    (d / "gpu-processes.csv").write_text("timestamp,pid,process_name,used_memory\n"
                                         "t1, 100, x, 588\nt1, 101, x, 9000\nt2, 100, x, 588\nt2, 101, x, 9000\n")
    assert own_processes({"engine": "trtllm", "version": "1.2.1", "arm": "defaults"}) == 2
    assert own_processes({"engine": "vllm", "version": "0.30.0", "arm": "defaults"}) == 1
    assert unit_conditions(d, own=2)["gpu_shared"] is False and unit_conditions(d, own=1)["gpu_shared"] is True
    with open(d / "gpu-processes.csv", "a") as f:
        f.write("t3, 100, x, 588\nt3, 101, x, 9000\nt3, 555, other, 2000\n")
    assert unit_conditions(d, own=2)["gpu_shared"] is True


def sweep_unit(root, session, batches, step_ns=STEP, ok=True, attempt=False, written=None, version="0.30.0"):
    u = Unit(engine="vllm", version=version, arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture="timing",
             batches=tuple(batches))
    d = unit_dir(session, u, root=root)
    write_meta(d, u)
    if written is not None:
        meta = json.loads((d / "meta.json").read_text())
        (d / "meta.json").write_text(json.dumps({**meta, "written": written}))
    write_status(d, ok, 1.0)
    jsonl(d / "driver.jsonl", [{"kind": "long", "batch": b, "repeat": i,
                                "cadence": {"step_ns": step_ns, "interval_ns": step_ns}}
                               for b in batches for i in range(3)])
    if attempt:
        (d.parent / f"{d.name}.attempt-1").mkdir()
    return d


@pytest.mark.parametrize("order", [("S2", "S2b"), ("S2b", "S2")])
def test_a_rerun_supersedes_in_any_session_order_and_drops_no_batch(tmp_path, order):
    """A rerun session (S2b, on a new host) re-measures batch 64 only, flagged in S2: in either order the rerun is read
    for 64 (the later unit), S2's copy named as a duplicate, and batch 1, which only S2 measured, stays; the rerun
    lands, so reruns.csv would not list the unit again."""
    sweep_unit(tmp_path, "S2", (1, 64), step_ns=10_000_000, written=1000.0)
    sweep_unit(tmp_path, "S2b", (64,), step_ns=9_000_000, attempt=True, written=2000.0)
    rows = {r["batch"]: r for r in build([tmp_path / s for s in order], bw=3.1e12, calibration={})}
    assert set(rows) == {1, 64}
    assert rows[64]["session"] == "S2b" and rows[64]["step_ms"] == pytest.approx(9.0)
    assert rows[64]["duplicate_sources"] == ["S2/vllm-0.30.0/defaults/L1/Qwen3-8B/timing"]
    assert rows[64]["sources"]["timing"].startswith("S2b/")
    assert rows[1]["session"] == "S2" and "duplicate_sources" not in rows[1]
    assert rows[64]["attempts"] == 2 and rows[1]["attempts"] == 1        # one attempt set aside before the kept one
    # the rerun read 9 ms where the run it superseded read 10: the cell keeps that ratio, and its noise is at least it
    from anatomy.report.derive import Noise
    assert rows[64]["rerun_ratio"] == pytest.approx(0.9) and "rerun_ratio" not in rows[1]
    assert Noise().sigma(rows[64])[0] == pytest.approx(0.1)
    assert rows[1]["sources"] == {"timing": "S2/vllm-0.30.0/defaults/L1/Qwen3-8B/timing"}  # no machine's paths


def test_a_refresh_at_a_new_release_keeps_both_and_the_views_read_the_newer(tmp_path, monkeypatch):
    """A refresh measures a new release: another cell, never a clash, so both releases stay in the table, and only the
    views' filter (metrics.current_versions, by the version-set files) reads the newer."""
    from anatomy import metrics
    monkeypatch.setattr(metrics, "release_ranks", lambda: {("vllm", "0.30.0"): 0, ("vllm", "0.31.0"): 1})
    sweep_unit(tmp_path, "S2", (1, 64), step_ns=10_000_000)
    sweep_unit(tmp_path, "R1", (1, 64), step_ns=9_000_000, version="0.31.0")
    rows = build([tmp_path / "R1", tmp_path / "S2"], bw=3.1e12, calibration={})
    assert sorted((r["version"], r["batch"]) for r in rows) == [("0.30.0", 1), ("0.30.0", 64), ("0.31.0", 1),
                                                                ("0.31.0", 64)]
    assert not any(r.get("duplicate_sources") for r in rows)
    assert metrics.current_versions(rows) == {"vllm": "0.31.0"}
    assert {r["version"] for r in metrics.defaults(rows, only_accepted=False)} == {"0.31.0"}


def test_each_session_is_read_at_its_own_card_s_bandwidth(tmp_path):
    from anatomy.tidy import resolve
    sweep_unit(tmp_path, "S1", (64,))
    sweep_unit(tmp_path, "S2", (64,))
    sweep_unit(tmp_path, "S3", (1,))
    for session, bw in (("S1", 3.1e12), ("S2", 3.0e12)):
        (tmp_path / session / "ruler").mkdir()
        (tmp_path / session / "ruler" / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": bw}))
    rows = {r["session"]: r for r in build([tmp_path / "S2", tmp_path / "S3"], calibrate=[tmp_path / "S1"])}
    assert rows["S2"]["bw_achievable"] == pytest.approx(3.0e12) and rows["S2"]["bandwidth_from"] == "S2"
    assert rows["S3"]["bw_achievable"] == pytest.approx(3.1e12) and rows["S3"]["bandwidth_from"] == "S1"
    assert resolve(rows["S2"]["sources"]["timing"], [tmp_path / "S2"]).is_dir()


def test_mbu_above_one_unsets_mbu_and_keeps_the_cell(tmp_path):
    from anatomy.metrics import accepted
    sweep_unit(tmp_path, "S2", (64,), step_ns=1_000_000)  # 1 ms: an 8B step faster than any card's bandwidth
    r = build([tmp_path / "S2"], bw=3.1e12, calibration={})[0]
    assert r["mbu"] is None and r["sol"] is None and "above 1" in r["mbu_flag"] and accepted(r)


def test_the_trace_window_moves_to_the_timing_window_s_context():
    """The timing window is the middle of the span every sequence decodes in. Counted back from the last anchor, a
    window sits half the sequences' start spread earlier where the last anchor is the first sequence's end (graph
    launches, full-batch copies), half later where it is the last sequence's end (a kernel every step runs)."""
    from anatomy.tidy import trace_offset
    assert trace_offset("graph", 5.8) == 3 and trace_offset("memcpy-dtoh", 5.8) == 3
    assert trace_offset("argmax|ArgMax", 5.8) == -3 and trace_offset("graph", None) == 0


def test_finish_spread_from_the_raw_waves(tmp_path):
    import gzip

    from anatomy.tidy import finish_spread
    seqs = [[(t * STEP + k * STEP, t) for t in range(1, 289)] for k in range(4)]   # starts 0..3 steps apart
    with gzip.open(tmp_path / "chunks-b4-r0.json.gz", "wt") as f:
        json.dump(seqs, f)
    recs = [{"kind": "long", "batch": 4, "repeat": 0, "cadence": {"step_ns": STEP}}]
    assert finish_spread(tmp_path, recs) == {4: pytest.approx(3.0)}


def test_tolerances_suggested_from_the_noise_block(tmp_path):
    from anatomy.noise import suggested_tolerances
    for n, (step, diff) in enumerate(((10.0e6, 10.1e6), (10.1e6, 10.1e6), (9.9e6, 10.2e6)), 1):
        d = make_unit(tmp_path, "timing", block=f"noise-start-p{n}")
        jsonl(d / "driver.jsonl", [*({"kind": "long", "batch": 64, "repeat": i,
                                      "cadence": {"step_ns": step, "interval_ns": step}} for i in range(3)),
                                   {"kind": "differential", "batch": 64, "step_ns": [diff] * 3}])
    got = suggested_tolerances([tmp_path / "S2-test"])
    assert got["step_noise"] == pytest.approx(0.01)            # SD 1.0% between the three processes, to 0.1% up
    assert got["agree"] == pytest.approx(0.035)                # |10.2 / 9.9 - 1| = 3.03%, to 0.5% up
    assert got["sentinel_drift"] == pytest.approx(3 * 0.01) and got["from"]["noise_cells"] == 1


def noise_unit(root, block, capture="timing", engine="vllm", level="L1", step=STEP, batches=(64,), valid=True):
    u = Unit(engine=engine, version="0.30.0" if engine == "vllm" else "0.5.20", arm="defaults", level=level,
             model="Qwen/Qwen3-0.6B", capture=capture, batches=batches, block=block)
    d = unit_dir("S1", u, root=root)
    write_meta(d, u)
    write_status(d, True, 1.0)
    if capture == "timing":
        cadence = {"step_ns": step, "interval_ns": step, **({} if valid else {"stalls": 1})}
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": b, "repeat": i, "cadence": cadence}
                                   for b in batches for i in range(3)])
    return d


def test_the_noise_block_reads_runners_and_occupancy_like_cells_and_skips_invalid_readings(tmp_path):
    from anatomy.noise import noise_table
    for n, (step, busy) in enumerate(((10_000_000, 9.0e6), (10_100_000, 9.2e6), (9_900_000, 8.8e6)), 1):
        noise_unit(tmp_path, f"noise-start-p{n}", step=step)
        g = noise_unit(tmp_path, f"noise-start-p{n}", capture="graph")
        jsonl(g / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
        fake_trace(g / "graph.1.sqlite", graphs=[(i * step, i * step + int(busy), 1) for i in range(20)])
        r = noise_unit(tmp_path, f"noise-start-p{n}", capture="runner", engine="sglang", level="L0")
        (r / "one_batch.jsonl").write_text(json.dumps({"batch_size": 64, "median_decode_latency": 0.01 * (1 + n / 100)})
                                           + "\n")
    noise_unit(tmp_path, "noise-end-p1", step=13_000_000, valid=False)   # a stalled process: no measure of noise
    rows = {(r["engine"], r["level"]): r for r in noise_table([tmp_path / "S1"])}
    vllm = rows["vllm", "L1"]
    assert vllm["processes"] == 3 and vllm["invalid"] == 1 and vllm["sd"] == pytest.approx(0.01, abs=1e-4)
    assert vllm["occupied_processes"] == 3 and vllm["occupied"] == pytest.approx(0.9, abs=2e-3)
    assert vllm["occupied_sd"] == pytest.approx(0.011, abs=5e-4)      # 0.900, 0.911, 0.889 of their own steps
    assert rows["sglang", "L0"]["processes"] == 3                     # the runner's own result file, as a cell's


def test_tolerances_hold_to_their_bands_and_a_noisy_engine_fails_g1_instead_of_loosening_them(tmp_path):
    from anatomy.noise import BANDS, suggested_tolerances
    for n, step in enumerate((10.0e6, 10.02e6, 9.98e6), 1):            # vLLM: 0.2% between processes
        noise_unit(tmp_path, f"noise-start-p{n}", step=step)
    for n, step in enumerate((10.0e6, 11.0e6, 9.0e6), 1):              # SGLang: 10% between processes
        noise_unit(tmp_path, f"noise-start-p{n}", engine="sglang", step=step)
    got = suggested_tolerances([tmp_path / "S1"], set_from="S1")
    assert got["set_from"] == "S1" and got["step_noise"] == pytest.approx(0.002)    # vLLM's, SGLang's never enters
    assert got["agree"] == BANDS["agree"][0] and got["sentinel_drift"] == BANDS["sentinel_drift"][0]  # clamped up
    assert [(f["engine"], f["statistic"]) for f in got["g1_failures"]] == [("sglang", "step_noise"),
                                                                          ("sglang", "sentinel_drift")]
    assert {p["engine"] for p in got["per_engine"]} == {"vllm", "sglang"}


def test_calibrate_alone_writes_noise_and_tolerances_and_reads_a_session_once(tmp_path, capsys):
    from anatomy.tidy import main
    for n, step in enumerate((10.0e6, 10.1e6, 9.9e6), 1):
        noise_unit(tmp_path, f"noise-start-p{n}", step=step)
    (tmp_path / "S1" / "ruler").mkdir()
    (tmp_path / "S1" / "ruler" / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": 3.1e12}))
    main(["--calibrate", str(tmp_path / "S1"), "--out", str(tmp_path / "out")])
    import yaml
    got = yaml.safe_load((tmp_path / "out" / "tolerances-suggested.yaml").read_text())
    assert got["set_from"] == "S1" and got["step_noise"] == pytest.approx(0.01)
    main([str(tmp_path / "S1"), "--calibrate", str(tmp_path / "S1"), "--out", str(tmp_path / "out2")])
    import csv
    with open(tmp_path / "out2" / "noise.csv") as f:
        assert [int(r["processes"]) for r in csv.DictReader(f)] == [3]   # read once: 3 processes, never 6


def test_timing_rerun_on_another_host_is_never_read_against_the_first_hosts_traces(tmp_path):
    from anatomy.metrics import accepted
    sweep_unit(tmp_path, "S2", (64,), written=1000.0)
    g = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-8B", capture="graph",
             batches=(64,))
    d = unit_dir("S2", g, root=tmp_path)
    write_meta(d, g)
    write_status(d, True, 1.0)
    jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    fake_trace(d / "graph.1.sqlite", graphs=[(i * STEP, i * STEP + 9_000_000, 1) for i in range(20)])
    assert accepted(build([tmp_path / "S2"], bw=3.1e12, calibration={})[0])
    sweep_unit(tmp_path, "S2b", (64,), written=2000.0)                   # the timing alone, on a new host
    r = build([tmp_path / "S2", tmp_path / "S2b"], bw=3.1e12, calibration={})[0]
    assert r["mixed_sessions"] and "different sessions" in rejection(r)


def test_g2_is_drift_within_a_session_and_across_sessions_a_host_difference(tmp_path):
    from anatomy.noise import sentinel_drift
    for session, start, end, t0 in (("S1", 10.0e6, 10.1e6, 0), ("S2", 10.5e6, 11.0e6, 100)):
        for position, step, t in (("start", start, t0), ("end", end, t0 + 50)):
            u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-0.6B",
                     capture="timing", batches=(1,), block=f"sentinel-{position}")
            d = unit_dir(session, u, root=tmp_path)
            write_meta(d, u)
            (d / "meta.json").write_text(json.dumps({**json.loads((d / "meta.json").read_text()), "written": t}))
            write_status(d, True, 1.0)
            jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 1, "repeat": i,
                                        "cadence": {"step_ns": step, "interval_ns": step}} for i in range(3)])
    rows = {(r["session"], r["position"]): r for r in sentinel_drift([tmp_path / "S1", tmp_path / "S2"])}
    assert rows["S1", "start"]["drift"] is None and rows["S1", "end"]["drift"] == pytest.approx(0.01)
    assert rows["S2", "end"]["drift"] == pytest.approx(0.5 / 10.5) and rows["S2", "end"]["drift_ok"] is False
    assert rows["S2", "start"]["host_drift"] == pytest.approx(0.05) and rows["S2", "start"]["drift_ok"] is None


def test_reruns_only_needs_no_ruler(tmp_path):
    import csv

    from anatomy.tidy import main
    d = sweep_unit(tmp_path, "S2", (64,))
    jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                "cadence": {"step_ns": s, "interval_ns": s}} for i, s in enumerate((9e6, 10e6, 11e6))])
    main([str(tmp_path / "S2"), "--reruns-only", "--out", str(tmp_path / "out")])
    with open(tmp_path / "out" / "reruns.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["session"], r["unit"], r["batches"]) for r in rows] == [("S2", "vllm-0.30.0/defaults/L1/Qwen3-8B/timing",
                                                                         "64")]


def test_a_ruler_file_without_a_measured_figure_is_no_bandwidth(tmp_path):
    (tmp_path / "S2" / "ruler").mkdir(parents=True)
    (tmp_path / "S2" / "ruler" / "bandwidth.json").write_text('{"read_bytes_per_s": null, "error": "failed"}')
    assert calibration.bandwidth([tmp_path / "S2"]) is None


def realistic_unit(root, session, *, block="main", model="Qwen/Qwen3-8B", capture="timing", batches=(64,), steps=None,
                   ok=True, written=0.0, status=True, meta=True):
    """A unit directory as anatomy.grid leaves it: meta.json (with when it was written), status.json, the driver's
    records; status=False is a unit killed before it wrote one, meta=False an abandoned one that only has a status."""
    u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model=model, capture=capture,
             batches=tuple(batches), block=block)
    d = unit_dir(session, u, root=root)
    d.mkdir(parents=True, exist_ok=True)
    if meta:
        write_meta(d, u)
        (d / "meta.json").write_text(json.dumps({**json.loads((d / "meta.json").read_text()), "written": written}))
    if status:
        write_status(d, ok, 1.0, None if ok else "exit 1")
    if capture == "timing":
        jsonl(d / "driver.jsonl", [{"kind": "long", "batch": b, "repeat": i,
                                    "cadence": {"step_ns": s, "interval_ns": s}}
                                   for b in batches for i, s in enumerate(steps or [STEP] * 3)])
    return d


def set_aside(d, ok, n=1):
    """An earlier attempt, as anatomy.grid.set_aside moves it: the whole directory, its meta and status included."""
    old = d.with_name(f"{d.name}.attempt-{n}")
    old.mkdir()
    (old / "meta.json").write_text((d / "meta.json").read_text())
    write_status(old, ok, 1.0, None if ok else "exit 1")
    jsonl(old / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i,
                                  "cadence": {"step_ns": 3 * STEP, "interval_ns": 3 * STEP}} for i in range(3)])


def test_a_realistic_set_of_sessions(tmp_path):
    """S1 (noise block at both ends, sentinels, ruler, host, the box's QC), S2 (a failed attempt set aside, a unit
    redone from reruns.csv with its original set aside, redo sentinels, an abandoned and a killed unit, a trace) and
    S2b (a timing rerun on another host): every rule tidy applies across sessions, on the shapes the grid leaves."""
    import csv

    from anatomy.metrics import accepted
    from anatomy.noise import noise_table, sentinel_drift
    from anatomy.tidy import main
    small = "Qwen/Qwen3-0.6B"
    for pos, t0 in (("start", 10.0), ("end", 500.0)):
        for n, step in enumerate((10.0e6, 10.05e6, 9.95e6), 1):        # wide repeats: never a rerun all the same
            realistic_unit(tmp_path, "S1", block=f"noise-{pos}-p{n}", model=small, steps=[step, 1.2 * step, step],
                           written=t0 + n)
    for pos, t, step in (("start", 1.0, 10.0e6), ("end", 900.0, 10.1e6), ("unlocked", 950.0, 9.0e6)):
        realistic_unit(tmp_path, "S1", block=f"sentinel-{pos}", model=small, batches=(1,), steps=[step] * 3,
                       written=t)
    (tmp_path / "S1" / "ruler").mkdir()
    (tmp_path / "S1" / "ruler" / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": 3.1e12}))
    (tmp_path / "S1" / "host").mkdir()
    (tmp_path / "S1" / "host" / "lscpu.txt").write_text("Model name: some CPU\n")
    (tmp_path / "S1" / "qc").mkdir()
    (tmp_path / "S1" / "qc" / "reruns.csv").write_text("session,unit,batches,ranges\n")

    failed_first = realistic_unit(tmp_path, "S2", batches=(1, 64), written=1000.0)
    set_aside(failed_first, ok=False)                                   # failed once, then ran ok
    redone = realistic_unit(tmp_path, "S2", model="Qwen/Qwen3-4B", written=3000.0)
    set_aside(redone, ok=True)                                          # ok, flagged, redone on the same box
    graph = realistic_unit(tmp_path, "S2", capture="graph", written=1100.0)
    jsonl(graph / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    fake_trace(graph / "graph.1.sqlite", graphs=[(i * STEP, i * STEP + 9_000_000, 1) for i in range(20)])
    for pos, t, step in (("start", 10.0, 10.0e6), ("end", 2500.0, 10.2e6), ("redo1-start", 2600.0, 10.0e6),
                         ("redo1-end", 3100.0, 10.4e6)):
        realistic_unit(tmp_path, "S2", block=f"sentinel-{pos}", model=small, batches=(1,), steps=[step] * 3,
                       written=t)
    realistic_unit(tmp_path, "S2", model="Qwen/Qwen3-14B", meta=False, ok=False)   # abandoned: a status, no meta
    realistic_unit(tmp_path, "S2", model="Qwen/Qwen3-32B", status=False, written=2000.0)  # killed mid-unit
    realistic_unit(tmp_path, "S2b", written=4000.0, steps=[0.9 * STEP] * 3)         # a timing rerun, another host

    roots = [tmp_path / "S2", tmp_path / "S2b"]
    rows = {(r["model"].split("-")[-1], r["batch"]): r for r in build(roots, bw=3.1e12, calibration={})}
    assert set(rows) == {("8B", 1), ("8B", 64), ("4B", 64)}             # nothing from the abandoned or killed units
    assert rows["8B", 1]["attempts"] == 2 and rows["4B", 64]["attempts"] == 2
    assert rows["4B", 64]["step_ms"] == pytest.approx(10.0)              # the redo, never the set-aside original
    b64 = rows["8B", 64]
    assert b64["session"] == "S2b" and b64["duplicate_sources"] == ["S2/vllm-0.30.0/defaults/L1/Qwen3-8B/timing"]
    assert b64["mixed_sessions"] and not accepted(b64)                   # S2b's timing against S2's trace: rejected
    assert accepted(rows["8B", 1])

    noise = noise_table([tmp_path / "S1"])
    assert [r["processes"] for r in noise] == [6]                        # three at each end; set-asides never count
    drift = {(r["session"], r["position"], r["redo"]): r for r in sentinel_drift([tmp_path / "S1", tmp_path / "S2"])}
    assert drift["S2", "end", "redo1"]["drift"] == pytest.approx(0.04)   # a redo's sentinels, against S2's own start
    assert drift["S2", "end", None]["drift"] == pytest.approx(0.02)
    assert drift["S1", "unlocked", None]["drift_ok"] is None             # its own series, never the gate

    main([str(tmp_path / "S2"), str(tmp_path / "S2b"), "--reruns-only", "--out", str(tmp_path / "qc")])
    with open(tmp_path / "qc" / "reruns.csv") as f:
        listed = [r["unit"] for r in csv.DictReader(f)]
    assert not [u for u in listed if u.startswith(("noise-", "sentinel-"))]


def test_a_forward_compatible_cuda_driver_is_recorded(tmp_path):
    from anatomy import tidy
    banner = ("== TensorRT-LLM ==\nCUDA Forward Compatibility mode ENABLED.\n"
              "  Using CUDA 13.1 driver version 590.44.01 with kernel driver version 580.105.08.\n")
    (tmp_path / "container.log").write_text(banner)
    assert tidy.cuda_compat(tmp_path) == "590.44.01 over 580.105.08"
    (tmp_path / "container.log").write_text("vLLM API server starting\n")
    assert tidy.cuda_compat(tmp_path) is None


def test_a_step_holding_two_steps_kernels_fails_the_anchor_check():
    """Two steps under one anchor are a 1/64 miscount the period check cannot see; the kernel ratio can."""
    from anatomy.metrics import rejection
    from anatomy.tidy import MISSED_ANCHOR, anchor_check
    assert anchor_check(1.0, 64, 1.0, 1.003) is True and anchor_check(1.0, 64, 1.0, None) is True
    assert anchor_check(1.0, 64, 1.0, MISSED_ANCHOR) is False and anchor_check(1.0, 64, 1.0, 2.0) is False
    assert "one missed" in rejection({"step_ms": 1.0, "node_anchor_ok": False})


def ruler_files(root, node_us=1.0, bw=3.1e12, ok=True, gpu="NVIDIA H100 80GB HBM3", nsys="2026.4.1", report=True):
    d = root / "ruler"
    d.mkdir(parents=True, exist_ok=True)
    (d / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": bw}))
    (d / "launch_latency.json").write_text(json.dumps({"graph_node_us": node_us}))
    (d / "host.json").write_text(json.dumps({"id": "x", "gpu": {"name": gpu}, "tools": {"nsys": nsys}}))
    if report:
        cal = {"graph_node_floor_us": node_us, "graph_node_spin_us": 0.8, "graph_node_floor_range_us": [0.8, node_us],
               "node_trace_inflation_ns": 105.0}
        (d / "ruler_report.json").write_text(json.dumps({"ok": ok, "calibration": cal}))


def test_a_failed_ruler_never_prices_a_cell(tmp_path):
    from anatomy.calibration import ruler_calibration
    ruler_files(tmp_path / "S1", ok=False)
    assert ruler_calibration([tmp_path / "S1"]) == {}
    ruler_files(tmp_path / "S2")
    assert ruler_calibration([tmp_path / "S2"])["node_trace_inflation_ns"] == 105.0


def test_a_calibration_is_lent_only_to_the_same_card_and_at_its_own_node_cost(tmp_path):
    """A session whose ruler did not pass keeps its own empty node always; it borrows S1's node-trace inflation on the
    same GPU and Nsight, and S1's spin-graph end and DRAM scale only within card_match of S1's node cost and
    bandwidth. Another Nsight borrows nothing, and says so (calibration_lent False)."""
    from anatomy.calibration import calibrations
    ruler_files(tmp_path / "S1")
    ruler_files(tmp_path / "S2", node_us=1.02, report=False)            # 2% off: the same card type
    ruler_files(tmp_path / "S3", node_us=1.12, report=False)            # 12% off: not
    ruler_files(tmp_path / "S4", report=False, nsys="2025.1.0")         # another Nsight: not
    ruler_files(tmp_path / "S5", ok=False)                              # its own failed; the card matches
    own, _ = calibrations([tmp_path / s for s in ("S2", "S3", "S4", "S5")], calibrate=[tmp_path / "S1"])
    s2_cal, s2_from = own["S2"][1], own["S2"][2]
    assert s2_from == "S1" and s2_cal["graph_node_floor_us"] == 1.02 and s2_cal["node_trace_inflation_ns"] == 105.0
    assert s2_cal["graph_node_floor_range_us"] == [0.8, 1.02] and s2_cal["calibration_lent"] is True
    # 12% off: the card's own node kept (the launch term stays modelled), the tracer's inflation lent, no spin end
    assert own["S3"][1] == {"graph_node_floor_us": 1.12, "graph_node_floor_range_us": [1.12, 1.12],
                            "node_trace_inflation_ns": 105.0, "calibration_lent": True} and own["S3"][2] == "S1"
    # another Nsight: its own node only, and nothing lent
    assert own["S4"][1] == {"graph_node_floor_us": 1.0, "graph_node_floor_range_us": [1.0, 1.0],
                            "calibration_lent": False} and own["S4"][2] is None
    assert own["S5"][2] == "S1" and own["S1"][2] == "S1"
    (tmp_path / "S6").mkdir()  # a validation run with no ruler of its own: borrows whole, and says it is unchecked
    own, _ = calibrations([tmp_path / "S6"], calibrate=[tmp_path / "S1"])
    assert own["S6"][1]["graph_node_floor_us"] == 1.0 and own["S6"][2] == "S1 (unchecked)"


def llama_log(waves, start_task=0, n_slots=None):
    """llama-server's log lines for waves of (slots, steps, graph-reuse count at the wave's end), after its load_model
    line when n_slots is given (a server's start)."""
    lines, task = [], start_task
    if n_slots:
        lines.append(f"0.00.900.000 I srv    load_model: initializing, n_slots = {n_slots}, n_ctx_slot = 1248, "
                     "kv_unified = 'false'")
    for slots, steps, reused in waves:
        for s in slots:
            lines.append(f"0.01.000.000 I slot launch_slot_: id {s:3d} | task {task} | processing task, is_child = 0")
            lines.append(f"0.02.000.000 I slot print_timing: id {s:3d} | task {task} |        eval time =    10.00 ms "
                         f"/   {steps} tokens (    0.10 ms per token)")
            lines.append(f"0.02.000.000 I slot print_timing: id {s:3d} | task {task} |    graphs reused =  {reused:8d}")
            task += 1
    return "\n".join(lines) + "\n"


def test_llama_servers_waves_must_run_in_one_run_of_slots_reusing_their_graph(tmp_path):
    """The pod's slow units: after batch 1 a 256-slot server ran batch 8 in slots {239-245, 254}, its graph reused a
    few times a wave. A server sized for its batch uses every slot in every wave."""
    from anatomy.llama_log import llama_slots
    recs = [{"kind": "long", "batch": 1}, {"kind": "long", "batch": 8}]
    whole = llama_log([([255], 96, 90), ([255], 288, 380), (range(248, 256), 96, 470), (range(248, 256), 288, 760)])
    (tmp_path / "container.log").write_text(whole)
    assert llama_slots(tmp_path, recs) == {}
    split = llama_log([([255], 96, 90), ([255], 288, 380), ([*range(239, 246), 254], 96, 385),
                       ([*range(239, 246), 254], 288, 390)])
    (tmp_path / "container.log").write_text(split)
    assert "not one run" in llama_slots(tmp_path, recs)[8]
    stalled = llama_log([([255], 96, 90), ([255], 288, 380), (range(248, 256), 96, 385), (range(248, 256), 288, 390)])
    (tmp_path / "container.log").write_text(stalled)
    assert llama_slots(tmp_path, recs)[8] == "llama-server reused its graph 5 times in a 288-step wave"
    fresh = llama_log([([0], 96, 90), ([0], 288, 380)]) + llama_log([(range(8), 96, 92), (range(8), 288, 380)], 2)
    (tmp_path / "container.log").write_text(fresh)  # a server per batch: the second's count starts again
    assert llama_slots(tmp_path, recs) == {}


def test_each_server_of_a_unit_is_judged_on_its_own_share_of_the_log(tmp_path):
    """A server per batch appends to one container.log, its task ids from 0 again and its graph count from scratch:
    keyed by task alone, three sound servers read as 'reused its graph 66 times'. Cut per server, each batch is judged
    on its own server; a count that fits no wave plan says so, and an old log (no warm-up wave) is read as it is."""
    from anatomy.llama_log import llama_slots
    recs = [{"kind": "long", "batch": b} for b in (1, 64, 256)]
    per_batch = "".join(llama_log([(range(b), 96, 90), (range(b), 288, 375)], n_slots=b) for b in (1, 64, 256))
    (tmp_path / "container.log").write_text(per_batch)
    assert llama_slots(tmp_path, recs) == {}
    wide = llama_log([(range(8), 96, 90), (range(8), 288, 375)], n_slots=256)
    (tmp_path / "container.log").write_text(wide)  # a per-batch server sized for 256
    assert llama_slots(tmp_path, [{"kind": "long", "batch": 8}]) == {8: "a server of 256 slots ran batch 8"}
    short = llama_log([(range(8), 96, 90), (range(5), 288, 375)], n_slots=8)
    (tmp_path / "container.log").write_text(short)  # requests missing: never a silent pass
    assert llama_slots(tmp_path, [{"kind": "long", "batch": 8}])[8].startswith("could not cut the log: 13 launches")
    old = llama_log([([0], 288, 285), (range(8), 288, 570)])  # measured waves only, one shared server
    (tmp_path / "container.log").write_text(old)
    assert llama_slots(tmp_path, [{"kind": "profile", "batch": 1}, {"kind": "profile", "batch": 8}]) == {}
    failed = llama_log([([0], 96, 90), ([0], 288, 375)])  # batch 8's wave failed on the shared server: nothing after
    (tmp_path / "container.log").write_text(failed)
    assert llama_slots(tmp_path, [{"kind": "long", "batch": 1}, {"kind": "error", "batch": 8},
                                  {"kind": "long", "batch": 64}]) == {}


def test_a_unit_that_loaded_another_snapshot_than_the_pin_is_rejected(tmp_path):
    """configs/models/revisions.yaml pins each model: a box whose cache held an older snapshot would load that one."""
    from anatomy.metrics import accepted
    from servebench.env import pinned
    d = sweep_unit(tmp_path, "S2", (64,))
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps({**meta, "model_revision": pinned("Qwen/Qwen3-8B")}))
    r = build([tmp_path / "S2"], bw=3.1e12, calibration={})[0]
    assert r["model_revision"] == pinned("Qwen/Qwen3-8B") and "revision_problem" not in r
    (d / "meta.json").write_text(json.dumps({**meta, "model_revision": "0" * 40}))
    r = build([tmp_path / "S2"], bw=3.1e12, calibration={})[0]
    assert not accepted(r) and "not its pin b968826d" in r["rejection"]


def test_a_graphs_off_cell_reads_its_kernels_from_its_graph_trace(tmp_path):
    """With CUDA graphs off the graph trace holds every kernel as a row already: the cell gets its kernel time and
    the five parts from it, with no node capture; a node capture, where there is one, wins."""
    def unit(capture):
        u = Unit(engine="vllm", version="0.30.0", arm="graphs-off", level="L1", model="Qwen/Qwen3-8B",
                 capture=capture, batches=(64,))
        d = unit_dir("S2-test", u, root=tmp_path)
        write_meta(d, u)
        write_status(d, True, 1.0)
        return d
    t = unit("timing")
    cadence = {"step_ns": STEP, "interval_ns": STEP}
    jsonl(t / "driver.jsonl", [{"kind": "long", "batch": 64, "repeat": i, "cadence": cadence} for i in range(3)])
    g = unit("graph")
    jsonl(g / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    # eager steps, no graph launched: 4 kernels of 1.9 ms each step (the last vLLM's argmax, its step anchor with
    # graphs off), then the host's 2.4 ms
    eager = [(i * STEP + j * 2_000_000, i * STEP + j * 2_000_000 + 1_900_000, "argmax" if j == 3 else "gemm")
             for i in range(20) for j in range(4)]
    fake_trace(g / "graph.1.sqlite", kernels=eager)
    r = build([tmp_path / "S2-test"], bw=3.1e12, calibration={})[0]
    assert r["kernel_ms"] == pytest.approx(7.6) and r["kernels_per_step"] == pytest.approx(4)
    assert r["kernel_exc_ms"] is not None and "node" not in r["sources"] and r["eager_launch"] is True
    # node tracing's inflation is a graph node's: none comes off kernels launched one by one, so no gap is made up
    cal = {"graph_node_floor_us": 1.0, "node_trace_inflation_ns": 100.0}
    lent = build([tmp_path / "S2-test"], bw=3.1e12, calibration=cal)[0]
    assert lent["kernel_exc_ms"] == pytest.approx(r["kernel_exc_ms"] - 4 * 1.0e-3, abs=1e-9)  # the 4 launches only
    n = unit("node")
    jsonl(n / "driver.jsonl", [{"kind": "profile", "batch": 64, "window_from_end": 18, "profile_steps": 16}])
    fake_trace(n / "node.1.sqlite", kernels=[(a, b - 100_000, name) for a, b, name in eager])
    assert build([tmp_path / "S2-test"], bw=3.1e12, calibration={})[0]["kernel_ms"] == pytest.approx(7.2)


def test_greedy_tokens_agree_across_arms_and_engines_and_an_engine_that_parts_at_once_is_flagged(tmp_path):
    """An arm that changes no arithmetic gives its defaults' very tokens; engines part where bf16 lets them, and one
    that parts from every other at once, on most prompts, while they agree, is flagged."""
    from anatomy.agreement import agreement

    def unit(engine, arm, level, outputs):
        u = Unit(engine=engine, version={"vllm": "0.30.0", "sglang": "0.5.20", "trtllm": "1.2.1",
                                         "llamacpp": "v0.5.0"}[engine], arm=arm, level=level,
                 model="Qwen/Qwen3-0.6B", capture="timing" if level == "L1" else "served", batches=(1,))
        d = unit_dir("S2", u, root=tmp_path)
        write_meta(d, u)
        write_status(d, True, 1.0)
        jsonl(d / "driver.jsonl", [{"kind": "greedy", "prompt": i, "ids": ids} for i, ids in enumerate(outputs)])
    same = [[1, 2, 3, 4, 5, 6]] * 3
    unit("vllm", "defaults", "L1", same)
    unit("vllm", "async-off", "L1", same)
    unit("sglang", "defaults", "L1", [[1, 2, 3, 4, 9, 9]] * 3)      # parts at 4: bf16's latitude
    unit("trtllm", "defaults", "L1", same)
    unit("llamacpp", "defaults", "L2", [[7, 7, 7, 7, 7, 7]] * 3)    # parts at 0 from everyone: a broken conversion
    rows = agreement([tmp_path / "S2"])
    ident = next(r for r in rows if r["kind"] == "identity")
    assert ident["b"] == "vllm-0.30.0/async-off/L1" and ident["identical"] == 3 and ident["flagged"] is False
    pairs = {(r["a"].split("-")[0], r["b"].split("-")[0]): r for r in rows if r["kind"] == "pair"}
    assert pairs[("sglang", "vllm")]["median_first_divergence"] == 4 and pairs[("trtllm", "vllm")]["identical"] == 3
    assert all(r["flagged"] is True and r["odd"] == "llamacpp" for k, r in pairs.items() if "llamacpp" in k)
    assert not any(r["flagged"] or r["odd"] for k, r in pairs.items() if "llamacpp" not in k)


def test_a_same_session_redo_keeps_the_ratio_to_the_run_it_set_aside(tmp_path):
    """session.sh's quality-control round reruns a flagged unit on the same box: the first run moves to
    <unit>.attempt-1 (anatomy.grid.set_aside), and the cell keeps the ratio between the two."""
    from anatomy.grid import set_aside
    d = sweep_unit(tmp_path, "S2", (64,), step_ns=10_000_000, written=1000.0)
    set_aside(d)
    sweep_unit(tmp_path, "S2", (64,), step_ns=9_000_000, written=2000.0)
    r = build([tmp_path / "S2"], bw=3.1e12, calibration={})[0]
    assert r["step_ms"] == pytest.approx(9.0) and r["attempts"] == 2 and r["rerun_ratio"] == pytest.approx(0.9)


def test_a_cell_measured_at_another_clock_than_its_ruler_is_rejected(tmp_path):
    """The launch term prices each kernel at the node cost the ruler measured at its SM clock; a unit at another lock
    (or an unlocked ruler beside locked cells) is not priced by it. Lending needs the same lock too."""
    from anatomy.calibration import calibrations
    from anatomy.metrics import accepted
    d = sweep_unit(tmp_path, "S2", (64,))
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps({**meta, "sm_clock_lock_mhz": 1980}))
    ruler_files(tmp_path / "S2")
    report = json.loads((tmp_path / "S2" / "ruler" / "ruler_report.json").read_text())
    for lock, ok in ((1980, True), (None, False)):
        (tmp_path / "S2" / "ruler" / "ruler_report.json").write_text(
            json.dumps({**report, "card": {"sm_clock_lock_mhz": lock}}))
        r = build([tmp_path / "S2"], bw=3.1e12)[0]
        assert accepted(r) is ok, r["rejection"]
    assert "calibrated at an unlocked clock, measured at 1980 MHz" in r["rejection"]
    ruler_files(tmp_path / "S1")
    (tmp_path / "S1" / "ruler" / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": 1980}))
    ruler_files(tmp_path / "S3", report=False)
    (tmp_path / "S3" / "ruler" / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": 1755}))
    own, _ = calibrations([tmp_path / "S3"], calibrate=[tmp_path / "S1"])
    assert own["S3"][1]["calibration_lent"] is False  # another lock: nothing of S1's lent


def test_noise_and_sentinel_readings_are_judged_per_batch_by_the_cells_rules(tmp_path):
    """A reading taken while another process used the GPU, or in llama-server slots that are not one run, is no
    measure of noise or drift: excluded with its reason, per batch, so one batch's fault leaves the unit's other
    batches standing (noise units are never rerun). A unit with no status at all measured nothing."""
    from anatomy.noise import noise_table, sentinel_drift
    from tests.s1fake import llama_log, timing_records
    for n, step in enumerate((10.0e6, 10.1e6, 9.9e6), 1):
        noise_unit(tmp_path, f"noise-start-p{n}", step=step, batches=(1, 8))
    shared = noise_unit(tmp_path, "noise-end-p1", step=10.05e6, batches=(1, 8))
    (shared / "gpu-processes.csv").write_text("timestamp,pid,process_name,used_memory\nt1, 1, x, 9\nt1, 2, other, 9\n")
    never = noise_unit(tmp_path, "noise-end-p2", step=10.0e6, batches=(1, 8))
    (never / "status.json").unlink()
    rows = {r["batch"]: r for r in noise_table([tmp_path / "S1"]) if r["engine"] == "vllm"}
    assert rows[1]["processes"] == 3 and rows[1]["invalid"] == 1 and "shared the GPU" in rows[1]["reasons"]
    assert rows[1]["reasons"].startswith("noise-end-p1:") and rows[8]["invalid"] == 1
    # llama-server: batch 8 in fragmented slots, batch 1 whole
    for n in (1, 2, 3):
        u = Unit(engine="llamacpp", version="v0.5.0", arm="defaults", level="L2", model="Qwen/Qwen3-0.6B",
                 capture="served", batches=(1, 8), block=f"noise-start-p{n}")
        d = unit_dir("S1", u, root=tmp_path)
        write_meta(d, u)
        write_status(d, True, 1.0)
        jsonl(d / "driver.jsonl", timing_records("llamacpp", (1, 8), 12_000_000 + n * 10_000))
        (d / "container.log").write_text(llama_log([1, 8], fragment=(8,) if n == 2 else ()))
    rows = {r["batch"]: r for r in noise_table([tmp_path / "S1"]) if r["engine"] == "llamacpp"}
    assert rows[1]["processes"] == 3 and rows[1]["invalid"] == 0
    assert rows[8]["processes"] == 2 and rows[8]["invalid"] == 1 and "slots that are not one run" in rows[8]["reasons"]
    # a sentinel read beside a foreign GPU process is not a reference for drift
    s = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-0.6B", capture="timing",
             batches=(1,), block="sentinel-start")
    d = unit_dir("S1", s, root=tmp_path)
    write_meta(d, s)
    write_status(d, True, 1.0)
    cadence = {"step_ns": STEP, "interval_ns": STEP}
    jsonl(d / "driver.jsonl", [{"kind": "long", "batch": 1, "repeat": i, "cadence": cadence} for i in range(3)])
    (d / "gpu-processes.csv").write_text("timestamp,pid,process_name,used_memory\nt1, 1, x, 9\nt1, 2, other, 9\n")
    (row,) = sentinel_drift([tmp_path / "S1"])
    assert row["valid"] is False and "shared the GPU" in row["problem"]


def test_llama_slots_allow_the_greedy_probes_requests_after_the_last_wave(tmp_path):
    """The agreement probe sends its prompts to the unit's last server after the measured waves: launches the waves
    do not account for, which must never fail the cut (the laptop rehearsal did, 56 launches for 24 or 48) and are
    never judged as a wave."""
    from anatomy.llama_log import llama_slots
    from tests.s1fake import llama_log, timing_records
    recs = timing_records("llamacpp", (1, 4), 10_000_000)      # 3 long + 3 short waves per batch, then 8 greedy prompts
    d = tmp_path
    (d / "container.log").write_text(llama_log([1, 4], probes=8))
    assert llama_slots(d, recs) == {}
    without = [r for r in recs if r.get("kind") != "greedy"]   # no probe records: the 8 launches are unexplained
    assert "could not cut the log" in llama_slots(d, without)[4]
    (d / "container.log").write_text(llama_log([1, 4], probes=8, fragment=(4,)))
    assert "not one run" in llama_slots(d, recs)[4]            # the probe never hides a fragmented wave


def test_the_suggested_tolerances_are_applied_in_place_with_every_comment_kept(tmp_path):
    import re
    import shutil
    from pathlib import Path

    import yaml

    from anatomy.noise import apply_suggested
    tol = tmp_path / "tolerances.yaml"
    shutil.copy(Path("configs/tolerances.yaml"), tol)
    before = tol.read_text()
    sug = tmp_path / "tolerances-suggested.yaml"
    before_values = yaml.safe_load(before)
    sug.write_text(yaml.safe_dump({"set_from": "S1-test", "step_noise": 0.0041, "agree": 0.02, "range_ok": 0.05,
                                   "sentinel_drift": 0.0123, "g1_failures": [], "per_engine": []}))
    changed = apply_suggested(sug, tol)
    after = yaml.safe_load(tol.read_text())
    assert after["set_from"] == "S1-test" and after["step_noise"] == 0.0041 and after["sentinel_drift"] == 0.0123
    assert after["overlap_model"] == before_values["overlap_model"] and "overlap_model" not in changed  # no choice:
    # the checked-in value stands; the keys that changed are the ones whose checked-in value differed
    assert set(changed) == {k for k in ("set_from", "step_noise", "agree", "range_ok", "sentinel_drift")
                            if before_values[k] != after[k]} and "set_from" in changed
    kept = yaml.safe_load(before)
    assert {k: v for k, v in after.items() if k not in changed} == {k: v for k, v in kept.items() if k not in changed}
    text = tol.read_text()
    assert text.count("#") == before.count("#") and len(text.splitlines()) == len(before.splitlines())
    assert re.search(r"^step_noise: 0\.0041\s+# a step time's", tol.read_text(), re.M)  # the comment stays on its line
    other = "raw" if before_values["overlap_model"] == "corrected" else "corrected"
    sug.write_text(yaml.safe_dump({"set_from": "S1-test", "overlap_model": other, "step_noise": 0.0041,
                                   "agree": 0.02, "range_ok": 0.05, "sentinel_drift": 0.0123}))
    assert apply_suggested(sug, tol) == {"overlap_model": (before_values["overlap_model"], other)}  # the chain
    # decided: applied, once
    assert apply_suggested(sug, tol) == {}


def test_the_opening_strip_draws_typical_steps_not_a_pause(tmp_path):
    """On the study's card the strip SKIP steps in held a hiccup of the traced run (vLLM's first drawn step 4.1 ms
    where its others were 2.3): the page's picture draws the window of consecutive steps nearest the capture's median,
    each within 15% of it; the explorer's cell strips keep the cell's own window."""
    from anatomy.strip import SKIP, strip, typical_window
    kernels, t = [], 0
    lengths = [10_000_000] * 30
    lengths[SKIP] = 25_000_000  # the step the fixed cut would draw first pauses for 15 ms
    for i, n in enumerate(lengths):
        kernels += [(t + j * 2_000_000, t + j * 2_000_000 + 1_900_000, "gemm",
                     {"graphNodeId": j, "graphId": 1, "correlationId": i}) for j in range(4)]
        t += n
    path = fake_trace(tmp_path / "node.1.sqlite", kernels=kernels)
    fixed = [r["start"] for r in strip(path) if r["kind"] == "step"]
    assert fixed[1] - fixed[0] == 25_000.0                      # the trace as captured: the pause
    typical = [r["start"] for r in strip(path, typical=True) if r["kind"] == "step"]
    assert [b - a for a, b in zip(typical, typical[1:])] == [10_000.0] * 3   # three steps like the median
    starts = [sum(lengths[:i]) for i in range(len(lengths) + 1)]
    at = typical_window(starts)
    assert at is not None and at >= SKIP and SKIP not in range(at, at + 3)   # clear of the edges and of the pause
    assert typical_window(starts, steps=3, within=0.0) is not None            # exact tens qualify at zero tolerance
    assert typical_window([0, 10, 35, 40, 70], steps=2, margin=0) is None     # nothing within 15% of the median
