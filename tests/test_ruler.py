"""The ruler check against traces shaped like the synthetic workload's (the real ones come from S1, or a validation
run): each nsys mode is judged against the ground truth its own run recorded."""
import json

import pytest

from ruler.validate import check
from tests.nsysfake import fake_trace

STEP_GPU, NODES, NODE_KERNEL = 400_000, 50, 7_000  # 400 us of GPU work per step; 50 nodes of 7 us kernel each
HOSTS = [450, 600, 1200, 2400]  # host waits longer than the GPU work: known busy fractions ~0.89 ... 0.17


def window(host_ns, steps=100, graph=True, node=False):
    """Traces for one window: steps of GPU work, each starting host_ns after the previous one started."""
    kernels, graphs, t = [], [], 0
    for s in range(steps + 1):
        if graph and not node:
            graphs.append((t, t + STEP_GPU, 1))
        elif graph and node:
            gap = (STEP_GPU - NODES * NODE_KERNEL) // NODES
            kernels += [(t + j * (NODE_KERNEL + gap), t + j * (NODE_KERNEL + gap) + NODE_KERNEL, "spin_kernel",
                         {"graphNodeId": j, "graphId": 1, "correlationId": s}) for j in range(NODES)]
        else:
            kernels.append((t, t + STEP_GPU, "void at::cuda::spin_kernel(long)"))
        t += host_ns
    return kernels, graphs


def write_run(folder, mode, variant, hosts=HOSTS, truth_busy=None):
    truth = {"mode": mode, "variant": variant, "nodes": NODES if variant == "graph" else 1, "windows": []}
    for i, h in enumerate(hosts, 1):
        step = h * 1000
        truth["windows"].append({"host_us": h, "steps": 100, "wall_ns": 100 * step, "gpu_ns": 100 * STEP_GPU,
                                 "busy": truth_busy if truth_busy is not None else STEP_GPU / step,
                                 "unix": [1000.0 + 10 * i, 1000.5 + 10 * i]})
        k, g = window(step, graph=variant == "graph", node=mode == "node")
        fake_trace(folder / f"busy-{mode}-{variant}.{i}.sqlite", kernels=k, graphs=g)
    (folder / f"busy-{mode}-{variant}.json").write_text(json.dumps(truth))
    log = folder / "gpu-processes.log"  # one GPU process (the ruler's) sampled through every window
    if not log.exists():
        log.write_text("".join(f"{1000 + 0.2 * k:.1f} 4242,\n" for k in range(300)))


def complete(folder):
    """Every run the ruler makes, faithful: both modes and variants traced, both untraced, the launch latency."""
    for mode in ("graph", "node"):
        for variant in ("eager", "graph"):
            write_run(folder, mode, variant)
    for variant in ("eager", "graph"):
        untraced(folder, variant)
    (folder / "launch_latency.json").write_text(json.dumps({"graph_node_us": 1.0}))


def test_ruler_passes_on_faithful_traces(tmp_path):
    complete(tmp_path)
    report = check(tmp_path)
    assert report["ok"], json.dumps(report, indent=1)
    graph_rows = [r for r in report["variants"]["graph"] if r["mode"] == "graph"]
    assert [round(r["truth_busy"], 2) for r in graph_rows] == [0.89, 0.67, 0.33, 0.17]  # every window informative
    gaps = [r["in_graph_gap_us_per_node"] for r in report["variants"]["graph"] if r["mode"] == "node"]
    assert len(gaps) == 4 and all(abs(g - (STEP_GPU / NODES - NODE_KERNEL) / 1e3) < 0.05 for g in gaps)


def test_ruler_fails_when_the_parser_is_off(tmp_path):
    write_run(tmp_path, "graph", "eager", hosts=[450], truth_busy=0.5)  # true busy is 400/450 = 0.89, not 0.5
    assert check(tmp_path)["ok"] is False


def test_a_window_shared_with_another_process_is_contaminated_not_judged(tmp_path):
    write_run(tmp_path, "graph", "eager", hosts=[450], truth_busy=0.5)  # wrong on purpose
    truth = json.loads((tmp_path / "busy-graph-eager.json").read_text())
    truth["windows"][0]["unix"] = [100.0, 101.0]
    (tmp_path / "busy-graph-eager.json").write_text(json.dumps(truth))
    (tmp_path / "gpu-processes.log").write_text("99.9 4242,\n100.5 4242,5151,\n101.2 4242,\n")
    report = check(tmp_path)
    row = report["variants"]["eager"][0]
    assert row["contaminated"] == "contaminated" and "ok" not in row and report["contaminated_windows"] == 1
    assert not report["ok"]


def test_a_window_no_sample_reached_is_unchecked_not_clean():
    from ruler.validate import contention
    samples = [(100.0, {"1"}), (102.0, {"1"})]
    assert contention(samples, [101.0, 101.15]) == "unchecked"   # 150 ms, nothing within 0.3 s
    assert contention(samples, [100.2, 100.4]) == "clean"        # the 100.0 sample brackets it
    assert contention([(100.5, {"1", "2"})], [100.2, 100.4]) == "contaminated"


def test_each_mode_is_judged_against_its_own_truth(tmp_path):
    """A node-mode run that recorded a different busy fraction cannot make the graph-mode traces pass or fail."""
    write_run(tmp_path, "graph", "eager")
    write_run(tmp_path, "node", "eager", truth_busy=0.1)  # its traces disagree with its own truth: node rows fail
    report = check(tmp_path)
    rows = report["variants"]["eager"]
    assert all(r["ok"] for r in rows if r["mode"] == "graph") and not any(r["ok"] for r in rows if r["mode"] == "node")


def test_dram_counter_calibration(tmp_path):
    """Five 4 GiB reads at 1.4 TB/s on an A100 (1.555 TB/s datasheet) whose counters say 85%: the counter under-reads,
    and the scale says by how much. Samples before each kernel's first period are the ramp and are left out."""
    from ruler.validate import contention, dram_calibration

    nbytes = 4 * 2**30
    dur = int(nbytes / 2 / 1.4e12 * 1e9)  # each call: two 2 GiB reduce kernels (32-bit indexing splits it)
    kernels = [(i * 2 * dur + h * dur, i * 2 * dur + (h + 1) * dur, "reduce_kernel") for i in range(5) for h in (0, 1)]
    def pct(t):  # 85% in steady state, 20% in a kernel's first sample period (the ramp), idle between kernels
        if any(s + 100_000 <= t <= e for s, e, _ in kernels):
            return 85
        return 20 if any(s <= t <= e for s, e, _ in kernels) else 0
    read = [(t, pct(t)) for t in range(0, 10 * dur, 100_000)]
    fake_trace(tmp_path / "dram-calibration.sqlite", kernels=kernels,
               metrics={"DRAM Read Bandwidth [Throughput %]": read})
    (tmp_path / "bandwidth-profiled.json").write_text(json.dumps(
        {"device": "NVIDIA A100-SXM4-40GB", "bytes": nbytes, "calls": 5, "unix": [1000.0, 1001.0]}))
    samples = [(1000.0 + 0.2 * k, {"1"}) for k in range(6)]
    cal = dram_calibration(tmp_path, lambda w: contention(samples, w))
    assert cal["dram_stream_read_pct"] == pytest.approx(85)
    assert cal["dram_counter_scale"] == pytest.approx(1.4e12 / (0.85 * 1.555e12), rel=1e-3)
    assert dram_calibration(tmp_path, lambda w: contention([(1000.5, {"1", "2"})], w)) == \
        {"dram_calibration": "contaminated"}


def test_an_idle_resident_is_tolerated_an_active_one_is_not():
    from ruler.validate import contention
    procs = [(100.0 + 0.2 * k, {"777", "42"}) for k in range(10)]   # 777: another user's server; 42: the ruler
    idle = [(99.5 + k, {"777": 0, "42": 60}) for k in range(3)]
    busy = [(99.5, {"777": 0}), (100.5, {"777": 35})]
    assert contention(procs, [100.4, 100.9]) == "contaminated"                      # no residents: two processes
    assert contention(procs, [100.4, 100.9], frozenset({"777"}), idle) == "clean"
    assert contention(procs, [100.4, 100.9], frozenset({"777"}), busy) == "contaminated"
    assert contention(procs, [100.4, 100.9], frozenset({"777"}), []) == "unchecked"  # no pmon sample near
    third = [(t, p | {"9"}) for t, p in procs]
    assert contention(third, [100.4, 100.9], frozenset({"777"}), idle) == "contaminated"


def test_activity_log_parsing(tmp_path):
    from ruler.validate import gpu_activity
    (tmp_path / "gpu-activity.log").write_text("100.5 40385:-,63321:12,\n101.5 \n")
    assert gpu_activity(tmp_path) == [(100.5, {"40385": 0, "63321": 12}), (101.5, {})]


def test_a_resident_window_in_a_pmon_gap_is_unchecked():
    from ruler.validate import contention
    procs = [(100.0 + 0.2 * k, {"777", "42"}) for k in range(20)]
    gap = [(98.8, {"777": 0}), (101.5, {"777": 0})]   # covers to 99.8, then from 101.5: [100.4, 100.9] unseen
    assert contention(procs, [100.4, 100.9], frozenset({"777"}), gap) == "unchecked"
    assert contention(procs, [100.4, 100.9], frozenset({"777"}), [(100.0, {"777": 0})]) == "clean"


def untraced(folder, variant, hosts=HOSTS, work=1.0):
    """The untraced run's truth for the same windows; work != 1: its GPU work per step differs from the traced run's
    (what tracing must not do, and what the cross check exists to catch)."""
    truth = {"mode": "none", "variant": variant, "windows": [], "nodes": NODES if variant == "graph" else 1,
             "spin_us": NODES * NODE_KERNEL / 1e3}
    for i, h in enumerate(hosts, 1):
        step = h * 1000
        truth["windows"].append({"host_us": h, "steps": 100, "step_ns": step, "busy": STEP_GPU * work / step,
                                 "step_device_ns": STEP_GPU * work, "unix": [1000.0 + 10 * i, 1000.5 + 10 * i]})
    (folder / f"busy-none-{variant}.json").write_text(json.dumps(truth))


def test_the_cells_occupancy_estimator_is_checked_across_runs(tmp_path):
    from ruler.validate import contention, cross_check, gpu_processes
    write_run(tmp_path, "graph", "eager")
    untraced(tmp_path, "eager")
    judge = lambda w: contention(gpu_processes(tmp_path), w)  # noqa: E731
    rows = cross_check(tmp_path, "eager", judge)
    assert len(rows) == 4 and all(r["ok"] for r in rows)
    untraced(tmp_path, "eager", work=0.9)  # tracing added 10% GPU work a step: busy over the step reads high
    assert not any(r["ok"] for r in cross_check(tmp_path, "eager", judge))


def test_a_ruler_without_its_evidence_fails_and_says_what_is_missing(tmp_path):
    """No run at all, truth whose traces are gone, a window count that differs from the traces, or a calibration
    without its keys: each fails with its reason, where each once read ok (variants filled with empty lists)."""
    empty = check(tmp_path)
    assert not empty["ok"] and "no busy-graph-eager.json" in empty["problems"]

    def case(name):
        (tmp_path / name).mkdir()
        complete(tmp_path / name)
        return tmp_path / name

    gone = case("gone")
    for p in gone.glob("busy-graph-eager.*.sqlite"):
        p.unlink()
    report = check(gone)
    assert not report["ok"] and "busy-graph-eager.json: 4 windows against 0 traces" in report["problems"]
    short = case("short")
    (short / "busy-node-graph.4.sqlite").unlink()  # one window's trace missing: zip would have dropped it
    assert "busy-node-graph.json: 4 windows against 3 traces" in check(short)["problems"]
    uncalibrated = case("uncalibrated")
    (uncalibrated / "launch_latency.json").unlink()
    assert check(uncalibrated)["problems"] == ["calibration without graph_node_floor_us, graph_node_floor_range_us"]


def test_a_non_finite_truth_fails_the_ruler(tmp_path):
    complete(tmp_path)
    truth = json.loads((tmp_path / "busy-graph-eager.json").read_text())
    truth["windows"][0]["busy"] = float("nan")
    (tmp_path / "busy-graph-eager.json").write_text(json.dumps(truth))
    report = check(tmp_path)
    assert not report["ok"] and "graph/eager window 1: no finite measurement" in report["problems"]


def test_the_report_names_the_card_it_ran_on(tmp_path):
    complete(tmp_path)
    (tmp_path / "host.json").write_text(json.dumps({"id": "abc", "gpu": {"name": "NVIDIA H100 80GB HBM3"},
                                                    "tools": {"nsys": "2026.4.1"}}))
    (tmp_path / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": 1980}))
    assert check(tmp_path)["card"] == {"host_id": "abc", "gpu": "NVIDIA H100 80GB HBM3", "nsys": "2026.4.1",
                                       "sm_clock_lock_mhz": 1980}
