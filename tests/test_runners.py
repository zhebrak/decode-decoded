"""L0 runner outputs, in the formats the pinned images print (one_batch.py's measurement_results; batched-bench's jsonl
printf), into timing cells, and through the tidy table."""
import json

import pytest

from anatomy.metrics import runner_cells
from anatomy.tidy import build
from servebench.results import Unit, unit_dir, write_meta, write_status
from servebench.runners import batched_bench_steps, one_batch_steps

ONE_BATCH = "\n".join(json.dumps({"run_name": "default", "batch_size": b, "input_len": 896, "output_len": 288,
                                  "prefill_latency": 0.05, "median_decode_latency": s, "median_decode_throughput": 1,
                                  "overall_throughput": 1}) for b, s in ((1, 0.0012), (1, 0.0013), (1, 0.0012),
                                                                         (64, 0.0050), (64, 0.0051), (64, 0.0049)))
BENCH_LINE = {"n_kv_max": 319488, "n_batch": 8192, "n_ubatch": 512, "flash_attn": 1, "is_pp_shared": 0,
              "n_gpu_layers": 999, "n_threads": 8, "n_threads_batch": 8, "pp": 896, "tg": 288, "pl": 64, "n_kv": 75776,
              "t_pp": 1.2, "speed_pp": 1, "t_tg": 1.44, "speed_tg": 1, "t": 2.6, "speed": 1}
BATCHED = "\n".join(["llama_context: n_ctx = 319488",
                     "main: n_kv_max = 319488, n_batch = 8192, n_ubatch = 512, flash_attn = 1, is_pp_shared = 0",
                     json.dumps({**BENCH_LINE, "t_tg": 6.8}),  # the adapter's warm-up run: cold, dropped
                     json.dumps(BENCH_LINE), json.dumps({**BENCH_LINE, "t_tg": 1.4688}),
                     "llama_perf_context_print: total time = 5000 ms"])


def test_one_batch():
    steps = one_batch_steps(ONE_BATCH)
    assert steps[1] == pytest.approx([1.2, 1.3, 1.2]) and len(steps[64]) == 3


def test_batched_bench_ignores_log_lines():
    steps = batched_bench_steps(BATCHED)
    assert steps == {64: pytest.approx([5.0, 5.1])}


def test_runner_cells_median_and_range():
    c = runner_cells({1: [1.2, 1.3, 1.2]})[1]
    assert c["step_ms"] == pytest.approx(1.2) and c["repeats"] == 3 and c["rerun"]  # 8% range: listed for rerun


def test_l0_cells_reach_the_table(tmp_path):
    for engine, version, fname, text in (("sglang", "0.5.20", "one_batch.jsonl", ONE_BATCH),
                                         ("llamacpp", "v0.5.0", "container.log", BATCHED)):
        u = Unit(engine=engine, version=version, arm="defaults", level="L0", model="Qwen/Qwen3-0.6B",
                 capture="runner", batches=(64,))
        d = unit_dir("S3-test", u, root=tmp_path)
        write_meta(d, u)
        write_status(d, True, 1.0)
        (d / fname).write_text(text)
    rows = {(r["engine"], r["batch"]): r for r in build([tmp_path / "S3-test"], bw=3.1e12)}
    assert rows[("sglang", 64)]["level"] == "L0" and rows[("sglang", 64)]["step_ms"] == pytest.approx(5.0)
    assert rows[("llamacpp", 64)]["step_ms"] == pytest.approx(5.05)
    assert rows[("llamacpp", 64)]["sources"] == {"runner": f"S3-test/{u.key}"}  # named under its session


def test_a_runner_throttled_by_a_cpu_quota_is_rejected(tmp_path):
    """A runner has no waves to bracket: the grid records its whole run's quota periods in status.json."""
    u = Unit(engine="sglang", version="0.5.20", arm="defaults", level="L0", model="Qwen/Qwen3-0.6B", capture="runner",
             batches=(64,))
    d = unit_dir("S3-test", u, root=tmp_path)
    write_meta(d, u)
    write_status(d, True, 1.0, cpu={"periods": 100, "throttled": 5})
    (d / "one_batch.jsonl").write_text(ONE_BATCH)
    row = {r["batch"]: r for r in build([tmp_path / "S3-test"], bw=3.1e12)}[64]
    assert row["cpu_throttled"] == pytest.approx(0.05) and "quota periods" in row["rejection"]
