"""The L1 offline driver's engine-independent part. It runs inside the engine's image, so it uses the standard library
only (plus servebench.cadence, also standard library), and talks to the engine through a small adapter:

    class Adapter:
        async def stream(self, prompt_ids, gen_len, rid)  -> async iterator of tokens-so-far (int) per output chunk
        async def tokens(self, prompt_ids, n, rid)        -> the n greedy token ids one prompt generates
        async def start_profile(self) / stop_profile(self) -> bracket an Nsight cudaProfilerApi capture range
        def info(self) -> dict                             -> what the engine reports about itself (KV capacity, ...)
        def close(self)

Per batch B, in one process (the model loads once per (engine, arm, model)):
1. warm-up: `warmup` full generations, so graphs, compile caches and allocator pools are hot before anything counts;
2. long and short, alternating, `repeats` of each: generations of gen_len and of short_gen_len tokens, all B prompts
           submitted at once, every output chunk timestamped (chunks-b<B>-r<n> and -s<n>); step time from
           servebench.cadence.estimate inside a long one's all-active window, and the differential estimator from each
           stream's own span (servebench.cadence.stream_span_ns) in a long and a short one;
3. profile (when --profile cuda): one more long generation inside one --capture-range=cudaProfilerApi capture,
           start_profile() before its requests and stop_profile() after its last token (profile_window); the window
           anatomy.tidy reads is `profile_steps` decode steps centred in the generation, so its mean context matches the
           cadence window's and the physics model's (prompt + gen / 2). Timed and profiled runs never share a
           generation.
After the last batch a timing unit runs the agreement probe (probe), in the sweep's own event loop: each of
configs/correctness/prompts.json's prompts for the model's family, alone, up to 32 greedy tokens, stopping at the
model's end of sequence as every engine does alike (ignore_eos would have one engine pick the runner-up there), their
ids recorded (kind "greedy"); anatomy.tidy compares them across engines and arms. It runs after every timed wave, so it
can never change one; a probe that fails is recorded and never fails the unit.
A batch whose B x (prompt + gen) tokens exceed the KV capacity the engine reports (adapter.info()["kv_tokens"]) is
recorded as skipped, never run: an engine would admit it and preempt during decode, and the cell would look plausible
and be wrong. The cadence estimate's stall count catches what the reported capacity misses.
Prompts are random token ids, unique per sequence and per generation, so neither prefix caching (off anyway) nor any
other reuse can shorten a prefill. Records go to --out as JSON lines; raw chunk times go beside it, gzipped.
"""
import argparse
import asyncio
import gc
import gzip
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench.cadence import differential, estimate, stream_span_ns  # noqa: E402
from servebench.cgroup import cpu_stat, since  # noqa: E402

TOKEN_RANGE = (1000, 100_000)  # ordinary tokens in both the Qwen3 (151,936) and Llama 3 (128,256) vocabularies


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-args", required=True, help="JSON: the merged arm's engine arguments")
    p.add_argument("--model", required=True)
    p.add_argument("--batches", required=True, help="comma-separated batch widths, run in this order")
    p.add_argument("--prompt-len", type=int, default=896)
    p.add_argument("--gen-len", type=int, default=288)
    p.add_argument("--short-gen-len", type=int, default=96)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--margin", type=int, default=16)
    p.add_argument("--profile", choices=("none", "cuda"), default="none")
    p.add_argument("--profile-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    a.engine_args = json.loads(a.engine_args)
    a.batches = [int(b) for b in a.batches.split(",")]
    return a


STACK = ("torch", "triton", "flashinfer-python", "nvidia-cublas", "nvidia-cublas-cu12", "vllm", "sglang",
         "sglang-kernel", "tensorrt_llm", "tensorrt-llm")


def stack():
    """The software the engine actually ran with, as the process sees it: an image and a venv at the same engine
    version differ here (TRT-LLM's image runs NVIDIA's torch 2.10 build, its pip install torch 2.9.1)."""
    from importlib import metadata

    out = {}
    for name in STACK:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        out["torch.cuda"] = torch.version.cuda
    except ImportError:
        pass
    # the CUDA user-mode driver this process loaded: an NGC image on an older kernel driver loads its own forward-
    # compatible one (TRT-LLM's on 580: "Using CUDA 13.1 driver version 590.44"), and launches go through it
    try:
        import ctypes

        version = ctypes.c_int()
        if ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(version)) == 0:
            out["cuda.driver_api"] = version.value
    except OSError:
        pass
    return out


def prompts(batch, length, seed, tag):
    rng = random.Random(f"{seed}/{batch}/{tag}")
    return [[rng.randrange(*TOKEN_RANGE) for _ in range(length)] for _ in range(batch)]


async def generation(adapter, ids, gen_len, tag):
    """All prompts at once; returns (t0, t1, [[(ns, tokens), ...] per sequence])."""
    async def one(i, prompt):
        chunks = []
        async for n in adapter.stream(prompt, gen_len, f"{tag}-{i}"):
            chunks.append((time.perf_counter_ns(), n))
        return chunks

    t0 = time.perf_counter_ns()
    seqs = await asyncio.gather(*(one(i, p) for i, p in enumerate(ids)))
    return t0, time.perf_counter_ns(), seqs


def profile_window(gen_len, margin, profile_steps):
    """(steps from the end of the wave to the window's start, steps in it): the window of profile_steps decode steps
    centred where the timed window is, counted back from the wave's last step. A capture spans the whole wave, from
    before its first request to after its last token, and anatomy.tidy cuts this window from the trace's own step
    anchors: no profiler latency (an nsys start, a first CUDA context) can cut it short or move it (TensorRT-LLM's
    captures once opened late and held a fraction of the steps), and the wave's last step is always captured."""
    start = max(margin, (gen_len - profile_steps) // 2)
    if start + profile_steps + margin > gen_len:
        raise ValueError("gen_len too short for the centred profile window plus its margin")
    return gen_len - start, profile_steps


async def profiled(adapter, ids, a, tag):
    """One wave inside one capture range; returns the generation and the capture's (perf_counter, UTC) times."""
    t_start, utc_start = time.perf_counter_ns(), time.time_ns()
    await adapter.start_profile()
    gen = await generation(adapter, ids, a.gen_len, tag)
    await adapter.stop_profile()
    return gen, (t_start, time.perf_counter_ns(), utc_start, time.time_ns())


def record(out, rec):
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


def save_chunks(out, name, seqs):
    with gzip.open(Path(out).parent / f"chunks-{name}.json.gz", "wt") as f:
        json.dump(seqs, f)


async def measured(coro):
    """Await a measured wave with the garbage collector held off (sweep says why); the collector is back on however
    the wave ends, and whatever the wave left behind is collected before the next one starts."""
    gc.disable()
    try:
        return await coro
    finally:
        gc.enable()
        gc.collect(generation=0)


async def sweep(adapter, a):
    out = a.out
    info = adapter.info()
    # The collector is held off a measured wave. This process hosts the engine's frontend (vLLM's AsyncLLM, SGLang's
    # tokenizer manager, TensorRT-LLM's result thread) beside the reader, and allocates a few records per token: at
    # 256 sequences a 288-token wave crosses the oldest generation's threshold about once, and that collection walked
    # the engine's whole heap for 150-400 ms while the engine kept stepping (SMOKE-20261003 and S1-20261003, every
    # engine, some long waves, never a 96-token one; the tokens then arrived in a burst at a third of the cadence,
    # which pulled one SGLang repeat's step 7% low when the burst fell inside the cadence window). Freezing moves what
    # the engine has allocated so far out of every later collection; disabling during a wave leaves no collection at
    # all between its first token and its last. Reference counting still frees the wave's records as it goes.
    gc.collect()
    gc.freeze()
    record(out, {"kind": "info", "engine_args": a.engine_args, "model": a.model, "info": info, "argv": sys.argv[1:],
                 "gen_len": a.gen_len, "short_gen_len": a.short_gen_len, "repeats": a.repeats, "margin": a.margin,
                 "profile_steps": a.profile_steps, "gc": {"frozen": gc.get_freeze_count(), "held_during_waves": True},
                 "stack": stack()})
    capacity = info.get("kv_tokens")
    for b in a.batches:
        if capacity and b * (a.prompt_len + a.gen_len) > capacity:
            record(out, {"kind": "skipped", "batch": b, "reason": f"needs {b * (a.prompt_len + a.gen_len)} KV tokens, "
                         f"the engine reports {capacity}"})
            continue
        try:
            for w in range(a.warmup):
                await generation(adapter, prompts(b, a.prompt_len, a.seed, f"w{w}"), a.gen_len, f"b{b}w{w}")
            spans = {}
            # long and short alternate (L S L S L S), so a drift in the card's state (a power cap's clock) falls on
            # both alike; the differential takes each stream's own first-to-last token span, as at L2, so neither
            # prefill nor its scatter enters it (in validation, both read as a 0.5-1.7% offset at batch 64)
            for r in range(a.repeats):
                for kind, n, tag in (("long", a.gen_len, "r"), ("short", a.short_gen_len, "s")):
                    cpu = cpu_stat()
                    t0, t1, seqs = await measured(generation(adapter, prompts(b, a.prompt_len, a.seed, f"{kind}{r}"),
                                                             n, f"b{b}{kind}{r}"))
                    cpu = since(cpu)  # a CPU quota's throttling during the wave, where there is a quota
                    spans.setdefault(kind, []).append(stream_span_ns(seqs))
                    rec = {"kind": kind, "batch": b, "repeat": r, "gen_len": n, "wall_ns": t1 - t0,
                           "span_ns": spans[kind][-1], "tokens": [s[-1][1] if s else 0 for s in seqs],
                           "cadence": estimate(seqs, a.margin),  # a short wave's own stall and gap checks
                           **({"cpu": cpu} if cpu else {})}
                    if kind == "long":
                        rec["cadence_tail"] = estimate(seqs, a.margin, after_tokens=a.short_gen_len)
                    # the raw arrivals before the record that names them: a kill between the two leaves a chunks
                    # file no record points at, never a record whose chunks are torn (tidy re-estimates from them)
                    save_chunks(out, f"b{b}-{tag}{r}", seqs)
                    record(out, rec)
            if a.repeats:  # a profiled unit runs no timed generations: its timing comes from the timing unit
                diffs = [differential(lw, sw, a.gen_len, a.short_gen_len) if lw and sw else None
                         for lw, sw in zip(spans["long"], spans["short"])]
                record(out, {"kind": "differential", "batch": b, "step_ns": diffs})
            if a.profile == "cuda":
                cpu = cpu_stat()
                (t0, t1, seqs), win = await measured(profiled(adapter, prompts(b, a.prompt_len, a.seed, "p"), a,
                                                              f"b{b}p"))
                cpu = since(cpu)
                from_end, steps = profile_window(a.gen_len, a.margin, a.profile_steps)
                record(out, {"kind": "profile", "batch": b, "wall_ns": t1 - t0, "capture_ns": list(win[:2]),
                             "utc_ns": list(win[2:]), "window_from_end": from_end, "profile_steps": steps,
                             "gen_len": a.gen_len, "cadence": estimate(seqs, a.margin),
                             **({"cpu": cpu} if cpu else {})})
        except Exception as e:  # one batch failing (out of KV, a crash in one shape) must not lose the others
            record(out, {"kind": "error", "batch": b, "error": repr(e), "trace": traceback.format_exc()})


PROMPTS = Path(__file__).resolve().parents[2] / "configs" / "correctness" / "prompts.json"


def probe_prompts(model, path=PROMPTS):
    """(the prompts' ids for the model's tokenizer family, tokens to generate), or (None, 0) for a model of no listed
    family: the family is the key the model's name contains (Qwen3, Llama-3), whatever path the engine was given."""
    if not Path(path).exists():
        return None, 0
    spec = json.loads(Path(path).read_text())
    family = next((f for f in spec["tokens"] if f.lower() in str(model).lower().replace("--", "/")), None)
    return (spec["tokens"][family], spec["new_tokens"]) if family else (None, 0)


PROBE_TIMEOUT_S = 120  # one prompt's 32 greedy tokens take well under a second: a probe that hangs is an error record,
                       # never the unit's whole deadline spent (anatomy.served.greedy bounds its probe the same way)


async def probe(adapter, a, timeout=PROBE_TIMEOUT_S):
    """The agreement probe, after the sweep: one "greedy" record per prompt, or one with its error."""
    ids, n = probe_prompts(a.model)
    for i, prompt in enumerate(ids or []):
        try:
            ids_out = await asyncio.wait_for(adapter.tokens(prompt, n, f"probe-{i}"), timeout)
            record(a.out, {"kind": "greedy", "prompt": i, "ids": ids_out})
        except Exception as e:  # a check beside the measurement: its failure is recorded, never the unit's
            record(a.out, {"kind": "greedy", "prompt": i, "error": repr(e)})


async def run(adapter, a):
    """The sweep, then, for a timing unit, the agreement probe, in one coroutine on one event loop: vLLM's AsyncLLM
    binds its output handler to the loop it first ran on, and a second asyncio.run finds it cancelled and the engine
    dead (EngineDeadError on every request)."""
    await sweep(adapter, a)
    if a.repeats and hasattr(adapter, "tokens"):  # timing units only: a profiled unit's waves are its traces
        await probe(adapter, a)


def main(make_adapter, argv=None):
    """make_adapter(args) -> Adapter. The sweep runs with asyncio.run, or with the adapter's own run(coro) when it has
    one (SGLang's Engine owns its event loop)."""
    a = parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    adapter = make_adapter(a)
    try:
        (getattr(adapter, "run", None) or asyncio.run)(run(adapter, a))
    finally:
        adapter.close()
