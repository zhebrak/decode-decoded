import asyncio
import gzip
import json

from servebench.drivers.common import main, prompts


class FakeEngine:
    """Tokens every `step` seconds after a staggered prefill; batch 3 fails (out of KV, say)."""

    def __init__(self, a, step=0.002):
        self.step, self.calls, self.profile, self.collector_on = step, [], [], {}

    async def stream(self, prompt_ids, gen_len, rid):
        if len(prompt_ids) and rid.startswith("b3"):
            raise RuntimeError("no KV for batch 3")
        import gc
        self.collector_on[rid.rsplit("-", 1)[0]] = gc.isenabled()  # per wave: held off a measured one, on a warm-up
        i = int(rid.rsplit("-", 1)[1])
        await asyncio.sleep(i * self.step)
        for n in range(1, gen_len + 1):
            await asyncio.sleep(self.step)
            yield n

    async def start_profile(self):
        self.profile.append("start")

    async def stop_profile(self):
        self.profile.append("stop")

    def info(self):
        return {"kv_tokens": 1000}

    def close(self):
        self.closed = True


def test_sweep_records(tmp_path):
    out = tmp_path / "driver.jsonl"
    engines = []

    def make(a):
        engines.append(FakeEngine(a))
        return engines[-1]

    main(make, ["--engine-args", '{"x": 1}', "--model", "m", "--batches", "2,3", "--prompt-len", "4", "--gen-len",
                "60", "--short-gen-len", "20", "--repeats", "2", "--warmup", "1", "--margin", "4", "--profile",
                "cuda", "--profile-steps", "8", "--out", str(out)])
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    kinds = [(r["kind"], r.get("batch")) for r in recs]
    assert kinds[0] == ("info", None) and recs[0]["engine_args"] == {"x": 1}
    assert kinds.count(("long", 2)) == 2 and kinds.count(("short", 2)) == 2
    assert ("differential", 2) in kinds and ("profile", 2) in kinds and ("error", 3) in kinds
    long = next(r for r in recs if r["kind"] == "long")
    assert long["tokens"] == [60, 60] and long["cadence"]["step_ns"] > 0
    diff = next(r for r in recs if r["kind"] == "differential")
    assert len(diff["step_ns"]) == 2 and all(d > 0 for d in diff["step_ns"])
    prof = next(r for r in recs if r["kind"] == "profile")
    assert prof["window_from_end"] == 60 - 26 and prof["profile_steps"] == 8  # centred: (60 - 8) // 2 = 26 in
    assert engines[0].profile == ["start", "stop"] and engines[0].closed
    chunks = json.load(gzip.open(tmp_path / "chunks-b2-r0.json.gz", "rt"))
    assert len(chunks) == 2 and chunks[0][-1][1] == 60
    # the garbage collector: the engine's heap frozen out of every collection once it is up, held off during each
    # measured wave (long, short and the profiled one) and back on for the warm-up and after the sweep
    import gc
    assert recs[0]["gc"]["held_during_waves"] and recs[0]["gc"]["frozen"] > 0 and gc.isenabled()
    on = engines[0].collector_on
    assert on["b2w0"] is True and not any(on[k] for k in ("b2long0", "b2short0", "b2long1", "b2short1", "b2p"))


def test_prompts_unique_and_reproducible():
    a, b = prompts(4, 16, 0, "long0"), prompts(4, 16, 0, "long0")
    assert a == b and len({tuple(p) for p in a}) == 4
    assert prompts(4, 16, 0, "long1") != a


def test_tensorrt_llms_kv_pool_is_read_from_its_own_log(tmp_path):
    """The executor logs an estimation pass and then the pool; only the pool's line, never the estimate's alone."""
    from servebench.drivers.trtllm_offline import kv_tokens
    log = tmp_path / "container.log"
    line = "[TensorRT-LLM][INFO] [MemUsageChange] Allocated {} GiB for max tokens in paged KV cache ({}).\n"
    log.write_text(line.format(1.77, 16608))
    got, why = kv_tokens(log, wait_s=0)
    assert got is None and "1 of its two" in why                     # the estimate alone would skip what fits
    with open(log, "a") as f:
        f.write(line.format(71.84, 672544))
    assert kv_tokens(log, wait_s=0)[0] == 672544
    assert kv_tokens(tmp_path / "missing.log", wait_s=0)[0] is None


def test_the_agreement_probe_picks_its_prompts_by_tokenizer_family_and_records_failures():
    """Each family's prompts by the name the engine got (a repo id, or TensorRT-LLM's snapshot path); a model of no
    listed family runs none; a prompt that fails is recorded and never fails the unit."""
    import asyncio
    import json

    from servebench.drivers.common import probe, probe_prompts
    qwen, n = probe_prompts("Qwen/Qwen3-8B")
    assert n == 32 and len(qwen) == 8 and qwen == probe_prompts("/hf/hub/models--Qwen--Qwen3-8B/snapshots/ab")[0]
    llama, _ = probe_prompts("unsloth/Llama-3.2-1B-Instruct")
    assert llama[0][0] == 128000 and probe_prompts("org/other")[0] is None  # Llama 3 starts with its BOS

    class Adapter:
        async def tokens(self, ids, n, rid):
            if rid == "probe-1":
                raise RuntimeError("boom")
            return list(range(n))

    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    out = Path(tempfile.mkdtemp()) / "driver.jsonl"
    asyncio.run(probe(Adapter(), SimpleNamespace(model="Qwen/Qwen3-0.6B", out=str(out))))
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 8 and all(r["kind"] == "greedy" for r in recs)
    assert recs[0]["ids"] == list(range(32)) and "boom" in recs[1]["error"]


def test_the_probe_runs_on_the_sweeps_own_event_loop():
    """vLLM's AsyncLLM ties its output handler to the loop it first ran on; a second asyncio.run finds the engine dead.
    An adapter bound to its first loop must still answer the probe."""
    import asyncio
    import json
    import tempfile
    from pathlib import Path

    from servebench.drivers import common

    class Bound:
        loop = None

        def _check(self):
            loop = asyncio.get_running_loop()
            self.loop = self.loop or loop
            if loop is not self.loop:
                raise RuntimeError("EngineDeadError")

        async def stream(self, ids, n, rid):
            self._check()
            for i in range(1, n + 1):
                yield i

        async def tokens(self, ids, n, rid):
            self._check()
            return [7] * n

        def info(self):
            return {}

    out = Path(tempfile.mkdtemp()) / "driver.jsonl"
    a = common.parse_args(["--engine-args", "{}", "--model", "Qwen/Qwen3-0.6B", "--batches", "1", "--out", str(out),
                           "--prompt-len", "8", "--gen-len", "64", "--short-gen-len", "32", "--warmup", "0",
                           "--repeats", "1"])
    asyncio.run(common.run(Bound(), a))
    recs = [json.loads(line) for line in out.read_text().splitlines() if '"greedy"' in line]
    assert len(recs) == 8 and all(r.get("ids") == [7] * 32 for r in recs)
