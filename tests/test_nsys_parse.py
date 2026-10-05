import random

import pytest

from servebench.nsys.parse import Interval, Trace, classify, dominant, step_starts, summarise, union_ns, whole_steps
from tests.nsysfake import fake_trace


def iv(*pairs):
    return [Interval(s, e) for s, e in pairs]


def test_union_overlap_nested_disjoint_and_empty():
    assert union_ns(iv((0, 10), (5, 15), (20, 30), (22, 25), (40, 40))) == 25
    assert union_ns([]) == 0


def test_union_clips_to_the_window():
    assert union_ns(iv((0, 10), (20, 30)), lo=5, hi=25) == 10
    assert union_ns(iv((0, 10)), lo=10, hi=20) == 0


def decode_steps(n=4, period=1000, graph_ns=800, sample_ns=100, node=False, gap=50, nodes=4):
    """n decode steps of `period` ns: a full-graph replay, then an eager sampling kernel, then the GPU idles waiting on
    the host. Graph mode gives one launch row per step; node mode gives one row per node (nodes x kernels with `gap`
    ns between them, sharing the step's correlationId) and no launch rows."""
    kernels, graphs = [], []
    for s in range(n):
        t0 = s * period
        if node:
            k_ns = (graph_ns - (nodes - 1) * gap) // nodes
            for j in range(nodes):
                a = t0 + j * (k_ns + gap)
                kernels.append((a, a + k_ns, f"void gemm_kernel_{j}<float>()",
                                {"graphNodeId": j, "graphId": 1, "correlationId": 100 + s}))
        else:
            graphs.append((t0, t0 + graph_ns, 1))
        kernels.append((t0 + graph_ns, t0 + graph_ns + sample_ns, "argmax_kernel", {"correlationId": 900 + s}))
    return kernels, graphs


def test_graph_mode_occupied_over_whole_steps(tmp_path):
    kernels, graphs = decode_steps()
    s = summarise(fake_trace(tmp_path / "g.sqlite", kernels=kernels, graphs=graphs))
    assert s["steps"] == 3 and s["window_ns"] == [0, 3000] and s["step_ns"] == 1000
    assert s["occupied"] == pytest.approx(0.9)  # 800 graph + 100 sampling of every 1000
    assert s["graph_launches"] == 3 and s["graph_table"] == "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"


def test_node_mode_sees_the_gaps_inside_the_graph(tmp_path):
    kernels, _ = decode_steps(node=True)
    s = summarise(fake_trace(tmp_path / "n.sqlite", kernels=kernels))
    assert s["steps"] == 3 and s["step_ns"] == 1000
    # 4 nodes of (800 - 3*50)/4 = 162 ns, plus 100 ns sampling: in-graph gaps are not kernel time
    assert s["kernel_ns_per_step"] == pytest.approx(4 * 162 + 100)
    assert s["occupied"] == s["kernel_active"]  # no launch rows in node mode: occupied is kernels only
    assert s["graph_node_kernels"] == 12 and s["kernels_per_step"] == 5


def test_occupied_minus_kernel_active_is_the_bubbles(tmp_path):
    k1, g1 = decode_steps()
    k2, _ = decode_steps(node=True)
    graph = summarise(fake_trace(tmp_path / "g.sqlite", kernels=k1, graphs=g1))
    node = summarise(fake_trace(tmp_path / "n.sqlite", kernels=k2))
    bubbles = graph["occupied_ns_per_step"] - node["kernel_ns_per_step"]
    assert bubbles == pytest.approx(900 - 748)  # the three 50 ns in-graph gaps plus rounding of the node split


def test_dominant_graph_anchors_steps(tmp_path):
    kernels, graphs = decode_steps(n=5)
    graphs.append((10_000, 10_400, 99))  # a different graph once (another batch shape): not a step anchor
    with Trace(fake_trace(tmp_path / "d.sqlite", kernels=kernels, graphs=graphs)) as t:
        assert step_starts(t) == [0, 1000, 2000, 3000, 4000]


def test_the_anchor_graph_is_decodes_not_the_most_launched():
    """A whole-wave capture holds the prefill: llama-server's prefill graph outnumbered its decode graph at 256 slots,
    in one unbroken run of 262 before decode's 256. A batch shape met as the batch fills and again as it drains is a
    few launches at either end."""
    prefill = [(t, "prefill") for t in range(0, 262)]
    shapes = [(t, "b8") for t in (262, 263, 900, 901)]
    decode = [(t, "decode") for t in range(264, 520 * 3, 3)]
    assert dominant(prefill + shapes + decode) == [t for t, _ in decode]
    assert dominant([]) == []


def test_overlapping_streams_count_once(tmp_path):
    kernels = [(0, 600, "a", {"streamId": 1}), (300, 900, "b", {"streamId": 2}), (2000, 2100, "a")]
    s = summarise(fake_trace(tmp_path / "o.sqlite", kernels=kernels), anchor="^a$")
    assert s["window_ns"] == [0, 2000] and s["steps"] == 1
    assert s["kernel_active"] == pytest.approx(900 / 2000)


@pytest.mark.parametrize("table", ["CUPTI_ACTIVITY_KIND_GRAPH_TRACE_V2", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE_V3"])
def test_versioned_graph_tables(tmp_path, table):
    kernels, graphs = decode_steps()
    s = summarise(fake_trace(tmp_path / "v.sqlite", kernels=kernels, graphs=graphs, graph_table=table))
    assert s["graph_table"] == table and s["occupied"] == pytest.approx(0.9)


def test_highest_graph_table_version_wins(tmp_path):
    path = fake_trace(tmp_path / "v.sqlite", graphs=[(0, 1, 1)], graph_table="CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    fake_trace(path, graphs=[(0, 1, 1)], graph_table="CUPTI_ACTIVITY_KIND_GRAPH_TRACE_V3")
    with Trace(path) as t:
        assert t.graph_table == "CUPTI_ACTIVITY_KIND_GRAPH_TRACE_V3"


def test_missing_tables_are_empty(tmp_path):
    with Trace(fake_trace(tmp_path / "k.sqlite", kernels=[(0, 5, "k")])) as t:
        assert t.graph_launches() == [] and t.memcpys() == [] and t.memsets() == [] and t.graph_table is None


def test_copies_and_sets_occupy_the_gpu(tmp_path):
    path = fake_trace(tmp_path / "c.sqlite", kernels=[(0, 100, "step"), (1000, 1100, "step")],
                      memcpys=[(100, 300)], memsets=[(300, 400)])
    s = summarise(path, anchor="step")
    assert s["occupied"] == pytest.approx(400 / 1000) and s["memcpy_ns"] == 200 and s["memset_ns"] == 100


def test_no_anchor_falls_back_to_the_activity_span(tmp_path):
    """Not the capture range: in real exports it opens ~30 ms before the first kernel, which would read as idle GPU."""
    path = fake_trace(tmp_path / "a.sqlite", kernels=[(100, 200, "k"), (400, 500, "k")], analysis=(0, 1000))
    s = summarise(path, anchor="never")
    assert s["window_ns"] == [100, 500] and s["window_source"] == "activity" and s["steps"] is None
    assert s["occupied"] == pytest.approx(0.5)


def test_whole_steps_skip_and_count():
    assert whole_steps([0, 10, 20, 30, 40], skip=1, steps=2) == (10, 30, 2)
    assert whole_steps([5]) is None


@pytest.mark.parametrize("name,cls", [
    ("void cutlass::device_kernel<flash::FlashAttnFwdSm90<cutlass::gemm::...>>(...)", "attention"),
    ("BatchDecodeWithPagedKVCacheKernel", "attention"),
    ("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1", "gemm"),
    ("nvjet_tst_64x8_64x16_2x1_v_bz_TNT", "gemm"),
    ("void at::native::reduce_kernel<512, 1, ArgMaxOps<float>>", "sampling"),
    ("fused_add_rms_norm_kernel", "norm_rope_act"),
    ("act_and_mul_kernel<silu>", "norm_rope_act"),
    # FlashInfer's norm kernels, by name in flashinfer::norm, are norms, not attention
    ("void flashinfer::norm::FusedAddRMSNormKernel<(unsigned int)8, __nv_bfloat16>(T2 *, ...)", "norm_rope_act"),
    ("kernel_cutlass_kernel_flashinfernormkernelsfusedaddrmsnorm", "norm_rope_act"),
    ("void flashinfer::BatchDecodeWithPagedKVCacheKernel<...>", "attention"),
    ("void tensorrt_llm::kernels::fusedQKNormRopeKernel<...>", "norm_rope_act"),
    ("mystery", "other"),
])
def test_kernel_classes(name, cls):
    assert classify(name) == cls


def test_matches_synthetic_ground_truth(tmp_path):
    """The S1 ruler in miniature: random steps with known busy time; the parser must recover it exactly."""
    rng = random.Random(0)
    kernels, graphs, t, busy = [], [], 0, 0
    for _ in range(200):
        g = rng.randint(300, 2000)
        graphs.append((t, t + g, 7))
        e = rng.randint(10, 300)
        kernels.append((t + g + rng.randint(0, 50), t + g + 50 + e, "sample", {"streamId": 3}))
        busy += g + (t + g + 50 + e) - max(t + g, kernels[-1][0])  # graph, then the eager kernel after its gap
        t += g + 50 + e + rng.randint(0, 1500)
    s = summarise(fake_trace(tmp_path / "r.sqlite", kernels=kernels, graphs=graphs))
    expected = busy - (graphs[-1][1] - graphs[-1][0]) - (kernels[-1][1] - kernels[-1][0])  # last step is outside
    assert s["steps"] == 199
    assert s["occupied"] * s["wall_ns"] == pytest.approx(expected)


def test_a_real_nsys_2026_5_1_export():
    """A real export (nsys 2026.5.1, NVTX capture range with --capture-range-end=repeat, CPU-only process): the tables
    the parser reads are absent and must read as empty; the capture range comes from ANALYSIS_DETAILS."""
    from pathlib import Path

    real = Path(__file__).parent / "fixtures" / "nsys-2026.5.1-nvtx-window-cpu.sqlite"
    with Trace(real) as t:
        assert {"ANALYSIS_DETAILS", "StringIds", "NVTX_EVENTS"} <= t.tables
        assert t.kernels() == [] and t.graph_launches() == [] and t.graph_table is None
        lo, hi = t.capture_range()
        assert 0.29e9 < hi - lo < 0.31e9  # the 300 ms window the process held open
    assert summarise(real).get("empty")  # no GPU activity at all: nothing to measure, said so


def test_real_export_clocks_and_leaked_warmup_launches(tmp_path):
    """As found in real graph-mode exports on the A100 (nsys 2026.5.1): activity rows count from the session start while
    ANALYSIS_DETAILS is absolute, and the warm-up replays before cudaProfilerStart are exported with negative times.
    The window must come from the 200 launches inside the capture, not stretch back over the 20 warm-ups."""
    base = 638_321_871_400
    warm = [(-471_000_000 + i * 570_000, -471_000_000 + i * 570_000 + 568_000, 1) for i in range(20)]
    real = [(30_000_000 + i * 570_000, 30_000_000 + i * 570_000 + 560_000, 1) for i in range(200)]
    path = fake_trace(tmp_path / "w.sqlite", graphs=warm + real, analysis=(base, base + 150_000_000),
                      session_start=base)
    with Trace(path) as t:
        assert t.capture_range() == (0, 150_000_000)
    s = summarise(path)
    assert s["steps"] == 199 and s["occupied"] == pytest.approx(560 / 570)


def test_dram_throughput_inside_the_window(tmp_path):
    kernels, graphs = decode_steps()
    read = [(t, 80 if t < 3000 else 0) for t in range(50, 4000, 100)]  # 80% inside the 3 whole steps, idle after
    path = fake_trace(tmp_path / "m.sqlite", kernels=kernels, graphs=graphs,
                      metrics={"DRAM Read Bandwidth [Throughput %]": read, "DRAM Write Bandwidth [Throughput %]":
                               [(t, 5) for t, _ in read], "SM Active [Throughput %]": [(t, 99) for t, _ in read]})
    d = summarise(path)["dram"]
    assert d["dram_read_pct"] == 80 and d["dram_write_pct"] == 5 and d["dram_read_samples"] == 30


def test_no_gpu_metrics_is_none(tmp_path):
    kernels, graphs = decode_steps()
    assert summarise(fake_trace(tmp_path / "g.sqlite", kernels=kernels, graphs=graphs))["dram"] is None


def test_metric_set_per_card():
    from servebench.nsys.capture import metric_set, nsys_cmd
    assert metric_set("NVIDIA H100 80GB HBM3") == "gh100" and metric_set("NVIDIA A100-SXM4-40GB") == "ga100"
    assert metric_set("Tesla T4") is None
    assert "--gpu-metrics-set=ga100" in nsys_cmd("/o/m", "metrics", metrics_set="ga100")



# Each interactive command's switches, from nsys 2026.5.1 --help (launch refuses --cpuctxsw)
LAUNCH = {"--command-file", "--cuda-event-trace", "--cuda-flush-interval", "--cuda-graph-trace", "--cuda-memory-usage",
          "--cuda-trace-all-apis", "--cudabacktrace", "--env-var", "--mpi-impl", "--inherit-environment",
          "--nccl-trace",
          "--nvtx-capture", "--osrt-threshold", "--python-backtrace", "--python-functions-trace", "--python-sampling",
          "--python-sampling-frequency", "--pytorch", "--resolve-symbols", "--session", "--session-new", "--trace",
          "-t", "--trace-fork-before-exec", "--show-output", "--wait"}
START = {"--backtrace", "--capture-range", "--capture-range-end", "--cpu-core-events", "--cpuctxsw", "--export",
         "--force-overwrite", "--flush-on-cudaprofilerstop", "--gpu-metrics-devices", "--gpu-metrics-frequency",
         "--gpu-metrics-set", "--gpuctxsw", "--output", "-o", "--sample", "--sampling-period", "--session",
         "--session-new", "--stats", "--stop-on-exit"}


@pytest.mark.parametrize("kind", ["graph", "node", "metrics", "hostload"])
def test_interactive_switches_go_to_the_command_that_takes_them(kind):
    from servebench.nsys.capture import nsys_launch_cmd, nsys_start_cmd
    switches = lambda cmd: {a.split("=")[0] for a in cmd[2:] if a.startswith("-")}  # noqa: E731
    assert switches(nsys_launch_cmd(kind, extra=["--trace-fork-before-exec=true"])) <= LAUNCH
    assert switches(nsys_start_cmd("/out/x", kind, metrics_set="ga100")) <= START


def test_overlaps_are_per_stream():
    from servebench.nsys.parse import Interval, overlaps
    ks = [Interval(0, 100, "a", 7), Interval(90, 200, "b", 7),     # starts 10 ns before a ends: overlapped
          Interval(150, 300, "c", 8),                              # another stream: concurrency, not a dependent launch
          Interval(300, 400, "d", 7)]
    assert overlaps(ks) == [10]


def test_the_logits_copy_anchors_a_step_where_kernels_repeat(tmp_path):
    """llama.cpp with graphs off: every candidate kernel runs twice a step (get_rows: 576 in 288 steps), but the step's
    logits come to the host once, at the full batch's size only while every sequence decodes."""
    step, vocab = 2_000_000, 151_936 * 4
    copies = [(i * step + 1_900_000, i * step + 1_950_000, 2, 8 * vocab) for i in range(40)]   # full batch of 8
    copies += [(40 * step + 1_900_000, 40 * step + 1_950_000, 2, 7 * vocab)]                  # the tail: one done
    copies += [(i * step, i * step + 1_000, 1, 32) for i in range(41)]                        # inputs to the device
    kernels = [(i * step + j * 500_000, i * step + j * 500_000 + 100_000, "k_get_rows_float") for i in range(41)
               for j in (0, 2)]
    path = fake_trace(tmp_path / "t.sqlite", kernels=kernels, memcpys=copies)
    with Trace(path) as t:
        starts = step_starts(t, "memcpy-dtoh")
        assert len(starts) == 40 and starts[1] - starts[0] == step     # one a step, the tail's smaller copy left out
        assert len(step_starts(t, "get_rows")) == 82                    # twice a step: no anchor


def test_scrub_blanks_the_environment_and_leaves_what_the_parser_reads(tmp_path):
    """Exports carry the traced processes' environment (a pod's held its cloud API key): never uploaded."""
    import shutil
    import sqlite3
    from pathlib import Path

    from servebench.nsys import scrub

    path = tmp_path / "t.sqlite"
    shutil.copy(Path(__file__).parent / "fixtures" / "nsys-2026.5.1-nvtx-window-cpu.sqlite", path)
    db = sqlite3.connect(path)
    top = db.execute("select max(id) from StringIds").fetchone()[0] or 0
    db.executemany("insert into StringIds (id, value) values (?, ?)",
                   [(top + 1, "CLOUD_API_KEY=cak_secret"), (top + 2, "PATH=/usr/bin\0SSH_CONNECTION=1.2.3.4 22"),
                    (top + 3, "PTXAS_FLAGS\0SSH_CLIENT=1.2.3.4 1 22"),  # a block whose first entry has no value
                    (top + 4, "void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128>>(Params)")])
    db.commit()
    db.close()
    before = summarise(path)
    assert scrub.leaks(path)                                  # the check sees it before the scrub
    assert scrub.scrub(path) == 3 and not scrub.leaks(path)   # every environment row blanked, nothing left
    names = {v for (v,) in sqlite3.connect(path).execute("select value from StringIds")}
    assert "void flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128>>(Params)" in names  # a kernel name stays
    assert summarise(path) == before                          # and so does everything the parser reads


def test_the_upload_check_sees_keys_tokens_and_compressed_files_and_not_engine_arguments(tmp_path):
    """The external review's canaries: a fake HF_TOKEN as a JSON key and inside a gzip went unseen (the check knew
    only NAME=value in raw bytes). And what real sessions hold must not trip it: vLLM's own lower-case hf_* arguments,
    an empty or null key, task- and disk- names, which a bare hf_ or sk- prefix matched in 15 real files."""
    import gzip
    import json

    from servebench.nsys import scrub

    fake = "hf_" + "Q" * 34  # the shape of a token, a fake one
    canaries = {"env.log": "HF_TOKEN=DELIBERATELY_FAKE\n", "keys.json": json.dumps({"HF_TOKEN": "DELIBERATELY_FAKE"}),
                "keys.yaml": "CLOUD_API_KEY: DELIBERATELY_FAKE\n", "token.txt": f"login with {fake}\n"}
    for name, text in canaries.items():
        (tmp_path / "bad" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "bad" / name).write_text(text)
    (tmp_path / "bad" / "log.gz").write_bytes(gzip.compress(b"HF_TOKEN=DELIBERATELY_FAKE"))
    assert sorted(p.name for p in scrub.leaks(tmp_path / "bad")) == sorted([*canaries, "log.gz"])
    clean = {"args.json": json.dumps({"hf_token": None, "hf_overrides": {}, "hf_config_path": "x", "HF_TOKEN": None}),
             "names.txt": "task-1 disk-0 mask-pad hf_overrides sk-short\n", "args.yaml": "HF_TOKEN:\nhf_token: null\n"}
    for name, text in clean.items():
        (tmp_path / "ok" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "ok" / name).write_text(text)
    (tmp_path / "ok" / "chunks.json.gz").write_bytes(gzip.compress(b"[[[1, 1], [2, 2]]]"))
    assert scrub.leaks(tmp_path / "ok") == []


def test_a_missed_anchor_shows_in_the_rows_of_its_step(tmp_path):
    """An eager step among graph replays has no graph launch: anchored on the graph, its interval holds two steps'
    kernels in a node trace (llama.cpp where its padded KV view grows), and the window would count 65 steps as 64. In
    a graph trace each replay is one row, so the step that ran eagerly between two launches holds many more."""
    kernels, _ = decode_steps(n=8, node=True)
    eager = [(a, b, name, {k: v for k, v in extra.items() if k not in ("graphNodeId", "graphId")})
             if 4000 <= a < 5000 else (a, b, name, extra) for a, b, name, extra in kernels]  # step 4 ran eagerly
    node = summarise(fake_trace(tmp_path / "n.sqlite", kernels=eager))
    assert node["steps"] == 6 and node["step_row_ratio"] == pytest.approx(2.0)
    assert summarise(fake_trace(tmp_path / "m.sqlite", kernels=kernels))["step_row_ratio"] == pytest.approx(1.0)
    graph_kernels, graphs = decode_steps(n=8)                  # a graph trace: one launch row and one kernel a step
    whole = summarise(fake_trace(tmp_path / "g.sqlite", kernels=graph_kernels, graphs=graphs))
    assert whole["step_row_ratio"] == pytest.approx(1.0)
    eager_step = [(4000 + j * 150, 4000 + j * 150 + 100, f"gemm_{j}", {}) for j in range(6)]  # step 4 ran eagerly
    missed = summarise(fake_trace(tmp_path / "e.sqlite", kernels=graph_kernels + eager_step,
                                  graphs=[g for g in graphs if g[0] != 4000]))
    assert missed["step_row_ratio"] >= 2


def test_a_steps_copies_to_the_host_are_one_anchor(tmp_path):
    """Under GPU sampling llama.cpp copies four 4-byte rows per slot each step, back to back: one anchor a step, not
    4 x B (at B = 2: 8 copies a step)."""
    kernels, copies = [], []
    for s in range(10):
        t0 = s * 1000
        kernels.append((t0, t0 + 600, "gemm", {"correlationId": s}))
        copies += [(t0 + 700 + i * 10, t0 + 705 + i * 10, 4) for i in range(8)]
    path = fake_trace(tmp_path / "s.sqlite", kernels=kernels, memcpys=[(a, b, 2, n) for a, b, n in copies])
    with Trace(path) as t:
        starts = step_starts(t, "memcpy-dtoh")
    assert starts == [s * 1000 + 700 for s in range(10)]
