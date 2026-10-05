"""A synthetic S1-shaped session for tests (tests/test_session_s1.py, tests/test_gate.py): what the first study session
leaves behind, built from the records the drivers and anatomy.served write, with no GPU. The NVL fixture
(tests/fixtures/cells-nvl) has no noise, sentinel or agreement rows, so every change to how tidy reads the noise block,
the sentinels, the reruns list or the suggested tolerances is checked here, against golden outputs
(tests/golden/s1-fake/, regenerated with UPDATE_GOLDEN=1).

    make_session(root) -> the session directory root/S1-fake, holding:
    - the ruler: bandwidth, launch latency and a passing report with a PDL calibration;
    - noise-start-p1..p3 and noise-end-p1..p3: vLLM's engine API and llama-server at Qwen3-0.6B, batches 1 and 8,
      with the differential, the greedy probe and (vLLM) a graph trace at batch 1; noise-end-p3's vLLM unit records
      an error at batch 8 (its batch-1 reading stands); noise-end-p2's llama-server ran batch 8 in slots that are
      not one run (its batch-1 reading stands, batch 8 is excluded with the reason);
    - sentinels: start and end for vLLM and llama-server, vLLM unlocked, and a redo round's start and end;
    - a probe-slots unit (llama-server, one server for batches 1 and 8), never a cell nor noise;
    - bring-up: vLLM's timing at batch 1 with repeats 8% apart (reruns.csv), its graph and hostload traces, SGLang's
      timing (the greedy-token pair), and a TensorRT-LLM unit that failed.
"""
import json
import time

from servebench.results import Unit, unit_dir, write_meta, write_status
from tests.nsysfake import fake_trace

SESSION = "S1-fake"
STEP = 10_000_000       # 10 ms decode steps
MODEL = "Qwen/Qwen3-0.6B"
VERSIONS = {"vllm": "0.30.0", "sglang": "0.5.20", "llamacpp": "v0.5.0", "trtllm": "1.2.1"}
LEVELS = {"vllm": "L1", "sglang": "L1", "llamacpp": "L2", "trtllm": "L1"}
PROCESS_STEPS = {"start": (1.000, 1.004, 0.997), "end": (1.006, 1.010, 1.003)}  # each process's step, over STEP
GREEDY = {"vllm": [[1, 2, 3, 4]] * 8, "sglang": [[1, 2, 3, 4, 6]] * 8, "llamacpp": [[1, 2, 3, 5]] * 8,
          "trtllm": [[1, 2, 3, 4]] * 8}


def jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def wave(kind, batch, repeat, step_ns, gen=None):
    gen = gen or (288 if kind == "long" else 96)
    cadence = {"step_ns": step_ns, "interval_ns": step_ns}
    r = {"kind": kind, "batch": batch, "repeat": repeat, "gen_len": gen, "tokens": [gen] * batch, "cadence": cadence}
    if kind == "long":
        r["cadence_tail"] = {"step_ns": step_ns}
    return r


def timing_records(engine, batches, step_ns, repeats=3, spread=0.0, errors=(), differential_off=0.0):
    """A timing or served unit's driver.jsonl: the info record, long and short waves alternating per batch, the
    differential (differential_off from the step), the greedy probe. spread: the repeats' range around step_ns.
    errors: batches whose error record replaces their waves."""
    recs = [{"kind": "info", "level": LEVELS[engine], "repeats": repeats, "margin": 16, "short_gen_len": 96,
             "stack": {"engine": engine}}]
    for b in batches:
        if b in errors:
            recs.append({"kind": "error", "batch": b, "error": "RuntimeError('out of KV')"})
            continue
        for i in range(repeats):
            s = int(step_ns * (1 + spread * (i - 1)))
            recs += [wave("long", b, i, s), wave("short", b, i, s)]
        recs.append({"kind": "differential", "batch": b, "step_ns": [int(step_ns * (1 + differential_off))] * repeats})
    recs += [{"kind": "greedy", "prompt": i, "ids": ids} for i, ids in enumerate(GREEDY[engine])]
    return recs


def llama_log(batches, waves=6, steps=(288, 96), fragment=(), probes=0, stall=()):
    """container.log for llama-server, one server per batch: each server's load_model line, then per wave one
    launch_slot_ line per sequence (slots 0..B-1, or, for a batch in `fragment`, slots that are not one run) and its
    print_timing lines (the prompt evaluated in 100 ms, or 2,000 for one request of a wave in `stall`, (batch, wave);
    the graph reused for every step; the wave's eval tokens); then, on the last server, `probes` requests of the greedy
    probe, one at a time."""
    lines, task = [], 0
    for b in batches:
        lines.append(f"srv  load_model: initializing, n_slots = {b}")
        reused = 0
        for w in range(waves):
            k = steps[w % 2]
            slots = [b + 1 + s for s in range(b)] if b in fragment else list(range(b))
            if b in fragment:
                slots[-1] = 2 * b + 7  # one sequence far from the others
            tasks = list(range(task, task + b))
            task += b
            for t, s in zip(tasks, slots):
                lines.append(f"slot launch_slot_: id  {s} | task {t} | processing task")
            reused += k - 1
            for i, (t, s) in enumerate(zip(tasks, slots)):
                prompt = 2000.0 if (b, w) in stall and i == 0 else 100.0
                lines.append(f"slot print_timing: id  {s} | task {t} |  prompt eval time = {prompt:8.2f} ms /   896 "
                             f"tokens ({prompt / 896:.2f} ms per token)")
                lines.append(f"slot print_timing: id  {s} | task {t} |  eval time =   100.00 ms / {k} tokens")
                lines.append(f"slot print_timing: id  {s} | task {t} |  graphs reused = {reused}")
    for i in range(probes):  # the probe's prompts, after the last wave, one request each on the last server
        s = i % batches[-1]
        lines.append(f"slot launch_slot_: id  {s} | task {task + i} | processing task")
        lines.append(f"slot print_timing: id  {s} | task {task + i} |  prompt eval time =   100.00 ms /   896 tokens")
        lines.append(f"slot print_timing: id  {s} | task {task + i} |  eval time =    10.00 ms / 32 tokens")
        lines.append(f"slot print_timing: id  {s} | task {task + i} |  graphs reused = 0")
    return "\n".join(lines) + "\n"


CLOCKS = "timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active\n"


def clock_log(d, mhz=1830, samples=5):
    """clocks.csv as servebench.clocks.ClockLog writes it: the SM clock at mhz, no slowdown."""
    rows = [f"2026/10/02 10:00:{i:02d}.000, {mhz}, 2619, 45, 300.0, 0x0000000000000000" for i in range(samples)]
    (d / "clocks.csv").write_text(CLOCKS + "\n".join(rows) + "\n")


def unit(root, engine, capture, batches, block, *, step_ns=STEP, ok=True, error=None, written=None, spread=0.0,
         errors=(), fragment=(), contention=None, lock=1830, extra_meta=None, level=None, probe=True):
    """One unit directory with its meta, status, clock log and, for a timing or served capture, its records (the
    greedy probe's unless probe is False: a server without a probe endpoint) and (llama.cpp) log. Returns the
    directory."""
    level = level or LEVELS[engine]
    cap = "served" if capture == "timing" and level == "L2" else capture
    u = Unit(engine=engine, version=VERSIONS[engine], arm="defaults", level=level, model=MODEL, capture=cap,
             batches=tuple(batches), block=block)
    d = unit_dir(SESSION, u, root=root)
    write_meta(d, u, git="fake0000", sm_clock_lock_mhz=lock, host={"gpu": {"name": "NVIDIA H100 80GB HBM3"}},
               **(extra_meta or {}))
    meta = json.loads((d / "meta.json").read_text())
    meta["written"] = written if written is not None else time.time()
    if contention is not None:
        meta["contention"] = contention
    (d / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    write_status(d, ok, 60.0, error)
    clock_log(d, mhz=1830 if lock else 1755)
    if not ok:  # a unit that failed left its info record and nothing to read
        jsonl(d / "driver.jsonl", [{"kind": "info", "level": level, "repeats": 3, "margin": 16, "short_gen_len": 96}])
    elif cap in ("timing", "served"):
        recs = timing_records(engine, batches, step_ns, spread=spread, errors=errors)
        jsonl(d / "driver.jsonl", recs if probe else [r for r in recs if r.get("kind") != "greedy"])
        if engine == "llamacpp":
            (d / "container.log").write_text(llama_log([b for b in batches if b not in errors], fragment=fragment))
    return d


def graph_unit(root, engine, block, step_ns, busy_ns, batches=(1,), capture="graph", written=None,
               contention=None, level=None):
    """A graph (or hostload) trace unit: 20 steps of one graph launch of busy_ns each, the window placed as the
    drivers place it (16 steps counted back from 18 steps before the end). An L1 unit's exports are numbered, a
    server's named per batch (<capture>-b<B>.sqlite), as anatomy.served writes them; a llama-server unit keeps its
    log too (one profile wave per batch)."""
    lvl = level or LEVELS[engine]
    d = unit(root, engine, capture, batches, block, written=written, contention=contention, level=lvl)
    jsonl(d / "driver.jsonl", [{"kind": "profile", "batch": b, "window_from_end": 18, "profile_steps": 16}
                               for b in batches])
    graphs = [(j * step_ns, j * step_ns + busy_ns, 1) for j in range(20)]
    for i, b in enumerate(batches, 1):
        name = f"{capture}-b{b}.sqlite" if lvl == "L2" else f"{capture}.{i}.sqlite"
        fake_trace(d / name, graphs=graphs)
    if engine == "llamacpp":
        (d / "container.log").write_text(llama_log(list(batches), waves=1))
    return d


def ruler(session, pdl=True, ok=True):
    r = session / "ruler"
    r.mkdir(parents=True, exist_ok=True)
    (r / "bandwidth.json").write_text(json.dumps({"read_bytes_per_s": 3.03e12, "copy_bytes_per_s": 3.03e12}))
    (r / "launch_latency.json").write_text(json.dumps({"graph_node_us": 0.98, "torch_launch_us": 4.9}))
    cal = {"graph_node_floor_us": 0.98, "graph_node_floor_range_us": [0.79, 0.98], "node_trace_inflation_ns": 105,
           "dram_counter_scale": 1.02}
    if pdl:  # two windows, both preferring the corrected model by more than the chain resolves (8 ns timer, 10 ns
        # spread over three untraced passes): the suggestion is "corrected" (anatomy.decompose.suggested_overlap_model)
        cal |= {"pdl_chain_ok": True, "pdl_exposed_ns_per_node": 310.0, "pdl_model_error_ns": 48.0,
                "pdl_model_corrected_error_ns": 21.0, "pdl_timer_step_ns": 8, "pdl_truth_passes": 3,
                "pdl_truth_spread_ns": 10.0, "pdl_release_ns": 310.0, "pdl_release_tail_ns": 2000,
                "pdl_wide_exposed_ns_per_node": 400.0, "pdl_wide_grid": 264,
                "pdl_windows": [{"tail_ns": 500, "pdl_model_error_ns": 48.0, "pdl_model_corrected_error_ns": 21.0,
                                 "pdl_model_floored_error_ns": 48.0, "pdl_model_floored_corrected_error_ns": 21.0},
                                {"tail_ns": 2000, "pdl_model_error_ns": 61.0, "pdl_model_corrected_error_ns": 12.0,
                                 "pdl_model_floored_error_ns": 61.0, "pdl_model_floored_corrected_error_ns": 12.0}]}
    else:
        cal["pdl"] = "skipped: Triton without launch_pdl"
    report = {"ok": ok, "problems": [] if ok else ["no busy-graph-eager.json"], "calibration": cal,
              "tolerance_pp": 1.0, "card": {"gpu": "NVIDIA H100 80GB HBM3", "sm_clock_lock_mhz": 1830}}
    (r / "ruler_report.json").write_text(json.dumps(report, indent=1))
    (r / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": 1830}))
    return r


def host(session, locked=True, help_files=True):
    """The session's host/ files provision.sh and capture_help.sh write: the clock lock's state and every image's
    --help."""
    h = session / "host"
    h.mkdir(parents=True, exist_ok=True)
    (h / "clock-lock.txt").write_text("locked\n" if locked else "refused\n")
    if help_files:
        for engine, version in (("vllm", "0.30.0"), ("sglang", "0.5.20"), ("trtllm", "1.2.1"),
                                ("llamacpp", "v0.5.0")):
            (h / "help" / engine / version).mkdir(parents=True, exist_ok=True)
            (h / "help" / engine / version / "help.txt").write_text("usage\n")
    return h


def make_session(root, *, pdl=True, clean=False):
    """The whole session under root; returns its directory. clean: no batch error, no fragmented server, no failed
    unit (what a session that passes its gates looks like)."""
    session = root / SESSION
    ruler(session, pdl=pdl)
    t = 1_000_000.0  # provenance times, in order
    for position in ("start", "end"):
        for n, factor in enumerate(PROCESS_STEPS[position], 1):
            block, t = f"noise-{position}-p{n}", t + 10
            step = int(STEP * factor)
            errors = (8,) if (position, n) == ("end", 3) and not clean else ()
            unit(root, "vllm", "timing", (1, 8), block, step_ns=step, written=t, errors=errors)
            graph_unit(root, "vllm", block, step, int(step * 0.9), written=t + 1)
            fragment = (8,) if (position, n) == ("end", 2) and not clean else ()
            unit(root, "llamacpp", "timing", (1, 8), block, step_ns=int(step * 1.2), written=t + 2,
                 fragment=fragment)
    for block, factor, lock in (("sentinel-start", 1.0, 1830), ("sentinel-end", 1.008, 1830),
                                ("sentinel-unlocked", 1.05, None), ("sentinel-redo1-start", 1.002, 1830),
                                ("sentinel-redo1-end", 1.009, 1830), ("sentinel-redo1-unlocked", 1.051, None)):
        t += 10
        unit(root, "vllm", "timing", (1,), block, step_ns=int(STEP * factor), written=t, lock=lock)
        if block in ("sentinel-start", "sentinel-end"):
            unit(root, "llamacpp", "timing", (1,), block, step_ns=int(STEP * 1.2 * factor), written=t + 1)
    t += 10
    unit(root, "llamacpp", "timing", (1, 8), "probe-slots", step_ns=int(STEP * 1.3), written=t)
    t += 10
    unit(root, "vllm", "timing", (1,), "bringup", written=t, spread=0.0 if clean else 0.08)  # 8% apart: a rerun
    graph_unit(root, "vllm", "bringup", STEP, int(STEP * 0.9), written=t + 1)
    # under contention the host runs 30% slower while the GPU's busy time per step stays: the busy-time test holds
    graph_unit(root, "vllm", "bringup", int(STEP * 1.3), int(STEP * 0.9), capture="hostload", written=t + 2,
               contention={"1": {"cpu_share": [0.91, 0.9, 0.92], "cores": [0, 1, 2]}})
    unit(root, "sglang", "timing", (1,), "bringup", written=t + 3, step_ns=int(STEP * 0.95))  # the greedy pair
    unit(root, "vllm", "timing", (1,), "bringup", written=t + 5, level="L2", probe=False)  # a server with no probe
    graph_unit(root, "vllm", "bringup", STEP, int(STEP * 0.9), written=t + 6, level="L2")  # its trace, named per batch
    if clean:
        unit(root, "trtllm", "timing", (1,), "bringup", written=t + 4)
    else:
        unit(root, "trtllm", "timing", (1,), "bringup", written=t + 4, ok=False, error="exit 1: no end id")
    return session
