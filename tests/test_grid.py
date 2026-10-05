import json
from pathlib import Path

import pytest

from anatomy import grid as G
from servebench.containers import Mounts
from servebench.engines.base import render
from servebench.results import is_done
from tests.conftest import linux_affinity

STUDY = Path(__file__).resolve().parents[1] / "configs" / "grid" / "v1.yaml"   # the study: three engines
V1 = Path(__file__).resolve().parents[1] / "configs" / "grid" / "validation" / "four-engines.yaml"   # its design


@pytest.fixture(scope="module")
def v1():
    """The four-engine grid the study was designed as (configs/grid/validation/four-engines.yaml): the harness's
    llama.cpp paths are tested against it; the study itself (configs/grid/v1.yaml) runs three engines, below."""
    return G.load(V1)


def test_the_study_grid_runs_three_engines_and_keeps_the_four_engine_design_as_validation():
    study = G.load(STUDY)
    assert study["engines"] == ["vllm", "sglang", "trtllm"] and not study.get("validation")
    for session in ("S1", "S2", "S3"):
        units = G.session_units(study, session)
        assert units and {u.engine for u in units} == {"vllm", "sglang", "trtllm"}
        assert {u.level for u in units if u.block != "runner" and not u.block.startswith("noise")} <= {"L1", "L2"}
    assert not {"gpu_sampling", "hostload_llama"} & set(study["blocks"])
    assert {u.engine for u in G.expand(study, blocks={"runner"})} == {"sglang"}
    four = G.load(V1)
    assert four["validation"] and "llamacpp" in four["engines"] and "gpu_sampling" in four["blocks"]


PIN_PROBLEMS = G.env.pin_problems  # the real check, before pins_hold stands in for it


@pytest.fixture(autouse=True)
def pins_hold(monkeypatch):
    """Every model at its pin: no test box holds the Hugging Face cache (the check has its own test)."""
    monkeypatch.setattr(G.env, "pin_problems", lambda models: [])


def mounts(out):
    return Mounts(repo=Path("/repo"), out=out, hf=Path("/hf"), cache=Path("/c"), models=Path("/m"), nsys=Path("/n"))


def test_feasibility_drops_and_resolves_max(v1):
    assert G.feasible([1, 64, 256, "max"], "Qwen/Qwen3-8B", 896, 288, v1)[-1] == 256  # max_num_seqs caps "max"
    b32 = G.feasible([1, 2, 4, 8, 16, 32, 64, 128, 256, "max"], "Qwen/Qwen3-32B", 896, 288, v1)
    assert 64 not in b32 and b32[-1] < 64 and b32[:5] == (1, 2, 4, 8, 16)
    assert 256 not in G.feasible([256], "Qwen/Qwen3-14B", 896, 288, v1)


def test_the_kv_bound_keeps_a_reserve_every_engine_holds(v1):
    """The top-end cells are ones vLLM, SGLang, TensorRT-LLM and llama.cpp all hold at 0.9 of the card: the naive
    bound (KV beside the weights) less kv_reserve_gib, so 14B stops at 211 rather than 244 and 32B at 16 rather
    than 36 (measured: vLLM, the tightest, held 218 sequences of 14B and 19 of 32B)."""
    assert G.feasible(["max"], "Qwen/Qwen3-14B", 896, 288, v1) == (211,)
    assert G.feasible(["max"], "Qwen/Qwen3-32B", 896, 288, v1) == (16,)
    assert G.feasible(["max"], "Qwen/Qwen3-32B", 896, 288, {**v1, "kv_reserve_gib": 0}) == (36,)


def test_s1_runs_each_engines_largest_s2_cells(v1):
    """What only breaks at scale (KV at 0.9, llama-server at 256 slots, the largest trace) breaks in S1, not ten hours
    into S2: S1's scale block holds every engine's top-end S2 cells."""
    s1 = G.session_units(v1, "S1")
    s2 = [u for u in G.expand(v1, sessions={"S2"}, blocks={"main"}) if u.capture in ("timing", "served")]
    top = {(u.engine, u.model): max(u.batches) for u in s2 if u.model.endswith(("14B", "32B"))}
    held = {(u.engine, u.model): set(u.batches) for u in s1 if u.capture in ("timing", "served")}
    assert top and all(b in held.get(k, set()) for k, b in top.items())
    assert {u.engine for u in s1 if u.capture == "node" and 256 in u.batches} == {"vllm", "sglang", "trtllm",
                                                                                     "llamacpp"}


def test_main_block(v1):
    units = G.expand(v1, blocks={"main"})
    by = {(u.engine, u.capture) for u in units}
    assert ("llamacpp", "timing") not in by and ("llamacpp", "served") in by  # its headline level is L2
    assert {u.level for u in units if u.engine == "llamacpp"} == {"L2"}
    assert {u.level for u in units if u.engine != "llamacpp"} == {"L1"}
    timing = {(u.engine, u.model): u.batches for u in units if u.capture in ("timing", "served")}
    node = [u for u in units if u.capture == "node"]
    assert node and all(u.batches == timing[(u.engine, u.model)] for u in node)  # kernel time at every timed batch
    assert len({u.model for u in units}) == 6


def test_non_default_arms(v1):
    arms = {(u.engine, u.arm) for u in G.expand(v1, blocks={"mitigations"})}
    assert {a for e, a in arms if e == "vllm"} == {"graphs-off", "async-off", "mrv2-off"}
    assert {a for e, a in arms if e == "sglang"} == {"overlap-off", "graphs-off"}
    assert {a for e, a in arms if e == "llamacpp"} == {"graphs-off", "gpu-sampling"}  # its headline level is L2
    assert {a for e, a in arms if e == "trtllm"} == {"overlap-off", "graphs-off"}


def test_served_and_runner_levels(v1):
    served = G.expand(v1, blocks={"served", "gpu_sampling"})
    assert {u.level for u in served} == {"L2"}
    pairs = {(u.engine, u.arm) for u in served}
    assert ("llamacpp", "gpu-sampling") in pairs and ("vllm", "gpu-sampling") not in pairs
    assert ("llamacpp", "defaults") not in pairs  # its server is its headline: the main block measures it
    assert {(u.engine, u.level) for u in G.expand(v1, blocks={"runner"})} == {("sglang", "L0"), ("llamacpp", "L0")}


def test_long_context(v1):
    lc = G.expand(v1, blocks={"long_context"})
    assert lc and all(u.prompt_len == 8064 and "-p8064g288" in u.key for u in lc)


def test_session_order(v1):
    units = G.session_units(v1, "S2")
    assert units[0].block == "sentinel-start" and units[-1].block == "sentinel-unlocked"
    assert [u.model for u in units if u.block == "sentinel-unlocked"] == ["Qwen/Qwen3-8B"]
    main = [u for u in units if u.block == "main" and u.engine == "vllm" and u.model == "Qwen/Qwen3-0.6B"]
    assert [u.capture for u in main] == ["timing", "graph", "node"]


def test_a_session_runs_each_pair_back_to_back(v1):
    """Everything compared with a model's defaults cells (its mitigation arms, its server, its runner) runs right after
    them, engine by engine, model by model: a pair's two sides never sit hours apart in a long session."""
    units = [u for u in G.session_units(v1, "S2") if not u.block.startswith("sentinel-")]
    stretch = [(u.model, u.engine) for u in units]
    runs = [k for i, k in enumerate(stretch) if i == 0 or stretch[i - 1] != k]
    assert len(runs) == len(set(runs))  # each (model, engine) is one contiguous run of units
    vllm = [u for u in units if (u.model, u.engine) == ("Qwen/Qwen3-0.6B", "vllm")]
    assert [u.block for u in vllm][:3] == ["main"] * 3
    assert {u.block for u in vllm} == {"main", "mitigations", "served", "hostload_arms", "hostload_serial",
                                       "hostload_defaults"}  # the busy-time checks beside the cells they test
    assert [u.model for u in units][0] == "Qwen/Qwen3-0.6B" and [u.model for u in units][-1] == "Qwen/Qwen3-32B"
    assert {u.block for u in G.session_units(v1, "S3")} >= {"long_context", "second_family", "dram"}
    assert not {"mitigations", "served", "runner"} & {u.block for u in G.session_units(v1, "S3")}


@pytest.mark.parametrize("session", ["S1", "S2", "S3"])
def test_every_unit_of_a_session_has_its_own_directory(v1, session):
    """Resume skips a unit whose status says done: two units sharing a key would make the second never run (the start
    and end sentinels did, before their block became part of the key)."""
    keys = [u.key for u in G.session_units(v1, session)]
    assert len(keys) == len(set(keys))


def test_commands_shapes(v1):
    units = G.expand(v1, blocks={"main"})
    timing = next(u for u in units if u.engine == "vllm" and u.capture == "timing")
    graph = next(u for u in units if u.engine == "sglang" and u.capture == "graph")
    l2 = next(u for u in units if u.engine == "llamacpp")
    t = render(G.commands(timing, mounts(Path("/o")), "0-11")["run"])
    assert "--cpuset-cpus 0-11" in t and "--entrypoint python3" in t
    assert "--profile none" in t and "nsys profile" not in t
    g = render(G.commands(graph, mounts(Path("/o")))["run"])
    assert "--entrypoint /opt/nsys/bin/nsys" in g and "--trace-fork-before-exec=true" in g and "--repeats 0" in g
    assert "--cuda-graph-trace=graph" in g and "--capture-range-end=repeat" in g and "--profile cuda" in g
    s = G.commands(l2, mounts(Path("/o")))
    assert set(s["servers"]) == {str(b) for b in l2.batches}  # llama.cpp: a server per batch
    assert all("/models/gguf/llamacpp-v0.5.0/Qwen3-0.6B-c1899de2-bf16.gguf" in render(x) for x in s["servers"].values())
    assert s["ready"].endswith("/health")
    trt = next(u for u in units if u.engine == "trtllm" and u.capture == "timing")
    assert "--entrypoint /opt/nvidia/nvidia_entrypoint.sh" in render(G.commands(trt, mounts(Path("/o")))["run"])


def test_overrides_reach_args_env_and_request_fields(v1):
    lc = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "llamacpp")
    over = {"llamacpp": {"args": {"-t": 2}, "env": {"X": "1"}, "request": {"timings_per_token": False}}}
    c = G.commands(lc, mounts(Path("/o")), overrides=over)
    assert all("-t 2" in render(x) and "X=1" in render(x) for x in c["servers"].values())
    assert c["request"]["timings_per_token"] is False and c["request"]["top_k"] == 1  # the release's others kept


def test_llamacpp_slot_context_follows_the_unit(v1):
    lc = next(u for u in G.expand(v1, blocks={"long_context"}) if u.engine == "llamacpp")
    servers = {b: render(x) for b, x in G.commands(lc, mounts(Path("/o")))["servers"].items()}
    assert f"-np 8 -c {8 * (8064 + 288 + 64)}" in servers["8"] and f"-np 1 -c {8064 + 288 + 64}" in servers["1"]


def test_llamacpp_threads_follow_the_engine_cores(v1):
    """llama.cpp's own default counts every CPU the host shows; its pool is sized to the engine's cpuset instead."""
    units = G.expand(v1, blocks={"main", "runner"})
    server = next(u for u in units if u.engine == "llamacpp" and u.level == "L2")
    runner = next(u for u in units if u.engine == "llamacpp" and u.level == "L0")
    assert "-t 22 -tb 22" in render(G.commands(server, mounts(Path("/o")), "0-21")["servers"]["1"])
    assert "-t 6 -tb 6" in render(G.commands(runner, mounts(Path("/o")), "0-3,8-9")["run"])
    assert " -t " not in render(G.commands(server, mounts(Path("/o")))["servers"]["1"])  # no cpuset: its own default


def test_cpusets():
    assert G.cpusets(range(26)) == ("0-21", "22-25")
    assert G.cpusets(range(4)) == (None, None)
    assert G.cpusets(range(16)) == ("0-11", "12-15")
    # a two-socket host: GPU 0's node (0-55, 112-167) gives the study VM's 22 engine cores and 4 client cores
    assert G.cpusets([*range(56), *range(112, 168)]) == ("0-21", "22-25")
    assert G.cpu_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"


def test_dry_run_prints_every_unit(v1, capsys):
    assert G.main([str(V1), "--session", "S3", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert out.count("# ") == len(G.session_units(v1, "S3"))


def test_profiled_l0_captures_are_refused(v1):
    import copy

    g = copy.deepcopy(v1)
    g["blocks"] = {"bad": {"engines": ["sglang"], "arms": ["defaults"], "levels": ["L0"],
                           "models": ["Qwen/Qwen3-0.6B"], "batches": [1], "captures": {"graph": "all"}}}
    with pytest.raises(ValueError, match="L0 runners are not profiled"):
        G.expand(g)


def test_grid_overrides_reach_the_command_and_are_recorded(tmp_path):
    """A validation grid's per-engine overrides (a shared card's smaller memory fraction, as the A100 box needed) land
    in the engine's own arguments."""
    spec = tmp_path / "shared.yaml"
    spec.write_text("extends: configs/grid/v1.yaml\nvalidation: true\n"
                    "overrides: {sglang: {args: {mem_fraction_static: 0.45}}}\n")
    g = G.load(spec)
    u = next(x for x in G.expand(g, blocks={"bringup"}) if x.engine == "sglang" and x.capture == "timing")
    cmd = render(G.commands(u, mounts(Path("/o")), overrides=g["overrides"])["run"])
    assert '"mem_fraction_static":0.45' in cmd


def test_a_unit_whose_batches_failed_is_not_ok(tmp_path, monkeypatch, v1):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)

    def fake_run(argv, log, timeout, **_):  # the driver exits 0 having recorded a skip and a failure
        (log.parent / "driver.jsonl").write_text(
            json.dumps({"kind": "skipped", "batch": 256, "reason": "KV"}) + "\n"
            + json.dumps({"kind": "error", "batch": 1, "error": "ValueError('end_id')"}) + "\n")
        return 0
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "trtllm" and u.capture == "timing")
    ok, _ = G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)
    status = json.loads((tmp_path / "S2-test" / unit.key / "status.json").read_text())
    assert not ok and status["error"].startswith("errors in batches [1]")
    assert G.batch_errors(tmp_path / "missing.jsonl") is None


def test_a_failed_unit_reruns_in_an_empty_directory(tmp_path, monkeypatch, v1):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "graph")
    attempt = iter([1, 0])

    def fake_run(argv, log, timeout, **_):
        n = next(attempt)
        (log.parent / "driver.jsonl").write_text(
            json.dumps({"kind": "error", "batch": 8, "error": "OOM"}) + "\n" if n
            else json.dumps({"kind": "profile", "batch": 1}) + "\n")  # one window opened, one export below
        (log.parent / "graph.1.sqlite").write_text(f"attempt {n}")
        return 0
    assert not G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)[0]
    assert G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)[0]   # the old errors are not read
    out = tmp_path / "S2-test" / unit.key
    assert (out / "graph.1.sqlite").read_text() == "attempt 0"
    assert (out.with_name(out.name + ".attempt-1") / "graph.1.sqlite").read_text() == "attempt 1"


def test_the_activity_idle_gate_is_refused_on_a_study_grid(monkeypatch):
    monkeypatch.setenv("DA_NSYS", "/n")
    monkeypatch.setattr(G.gpu, "IDLE_MODE", "activity")
    monkeypatch.setattr(G, "run_unit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a unit")))
    with pytest.raises(SystemExit, match="validation"):
        G.main([str(STUDY), "--session", "S1", "--run", "--no-gpu"])


def test_native_commands_are_the_container_commands_on_host_paths(v1, tmp_path, monkeypatch):
    monkeypatch.setattr("servebench.containers.ENVS", tmp_path / "envs")
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "graph")
    m = Mounts(repo=Path("/srv/repo"), out=tmp_path / "out", hf=Path("/srv/hf"), cache=Path("/srv/cache"),
               models=Path("/srv/models"), nsys=Path("/srv/nsys"))
    docker = G.commands(unit, m, cpuset="0-123", runtime="docker")["run"]
    native = G.commands(unit, m, cpuset="0-123", runtime="native")
    argv = native["run"]
    assert argv[:2] == ["env", "-i"] and native["runtime"] == "native"
    assert any(a.startswith(f"PATH={tmp_path}/envs/vllm-0.30.0/bin:") for a in argv)
    assert "HF_HOME=/srv/hf" in argv and "CUDA_VISIBLE_DEVICES=0" in argv
    assert argv[argv.index("taskset") + 2] == "0-123"
    cmd = argv[argv.index("taskset") + 3:]
    assert cmd[0] == "/srv/nsys/bin/nsys" and f"{tmp_path}/out/graph" in cmd
    tail = docker[docker.index("--entrypoint") + 3:]  # the container's command after the entrypoint and image
    moved = [a.replace("/work", "/srv/repo").replace("/out", f"{tmp_path}/out") for a in tail]
    assert cmd[-len(tail):][1:] == moved[1:]


def test_native_runs_use_the_engines_own_cuda_toolkit(tmp_path):
    from servebench.containers import bundled_cuda, native_run
    env_dir = tmp_path / "sglang-0.5.20"
    nvcc = env_dir / "lib/python3.12/site-packages/nvidia/cu13/bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.touch()
    assert bundled_cuda(env_dir) == nvcc.parent.parent and bundled_cuda(tmp_path / "none") is None
    m = Mounts(repo=Path("/r"), out=Path("/o"), hf=Path("/h"), cache=Path("/c"), models=Path("/m"))
    argv = native_run("img", ["python3", "-V"], name="x", mounts=m, env_dir=env_dir, cuda="bundled")
    assert f"CUDA_HOME={nvcc.parent.parent}" in argv
    assert "CUDA_HOME=/usr/local/cuda" in native_run("img", ["python3"], name="x", mounts=m, env_dir=env_dir)
    assert next(a for a in argv if a.startswith("PATH=")).split(":")[1] == f"{nvcc.parent}"


def test_a_profiled_unit_missing_an_export_fails(tmp_path, monkeypatch, v1):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "graph")

    def fake_run(argv, log, timeout, **_):  # three windows opened, nsys exported two
        (log.parent / "driver.jsonl").write_text("".join(json.dumps({"kind": "profile", "batch": b}) + "\n"
                                                         for b in (1, 8, 64)))
        for i in (1, 2):
            (log.parent / f"graph.{i}.sqlite").write_text("")
        return 0
    ok, _ = G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)
    status = json.loads((tmp_path / "S2-test" / unit.key / "status.json").read_text())
    assert not ok and status["error"] == "3 profile windows but 2 trace exports"


def test_profiled_units_without_nsys_are_refused(monkeypatch):
    monkeypatch.delenv("DA_NSYS", raising=False)
    monkeypatch.setattr(G, "run_unit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran a unit")))
    with pytest.raises(SystemExit, match="DA_NSYS"):
        G.main([str(V1), "--session", "S1", "--run", "--no-gpu"])



def test_teardown_ends_the_group_even_a_child_that_ignores_term():
    import subprocess

    from servebench.containers import teardown
    proc = subprocess.Popen(["bash", "-c", "trap '' TERM; sleep 60 & wait"], start_new_session=True)
    teardown(proc, grace_s=0.2)
    assert proc.poll() is not None


def test_strays_are_the_run_s_own_never_another_run_s(tmp_path):
    """By the run's DA_UNIT marker, or from its venv with no marker at all; never a process marked for another unit
    (another grid on the same box), never one that merely mentions the venv elsewhere."""
    from servebench.containers import strays
    venv = "/root/da-envs/trtllm-1.2.1"
    for pid, environ, cmdline in ((101, "DA_UNIT=u1", "python3 worker.py"),
                                  (102, "DA_UNIT=u2", f"{venv}/bin/python3 -m driver"),
                                  (103, "PATH=/usr/bin", f"{venv}/bin/python3 -m mpi_worker"),
                                  (104, "PATH=/usr/bin", "bash -c sleep"),
                                  # a compiler building in the venv's tree, a conversion in its own env: never strays
                                  (105, "PATH=/usr/bin", f"/usr/local/cuda/bin/nvcc -c {venv}/src/mmf.cu"),
                                  (106, "PATH=/usr/bin", f"{venv}/py/bin/python convert_hf_to_gguf.py")):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "environ").write_bytes(f"HOME=/root\0{environ}\0".encode())
        (d / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode())
    assert sorted(strays("u1", venv, proc=tmp_path)) == [101, 103]
    assert strays("u1", proc=tmp_path) == [101]


def test_s1_opens_and_closes_with_its_noise_block(v1):
    """Every engine's 0.6B at batch 1, 8 and 64 (engine API and server), its 8B at batch 1, 64 and 256, its most
    host-bound arm timed and traced (that arm's own occupancy spread), a graph trace of its 0.6B at batch 1 and 64,
    llama-server's GPU sampling, and the sentinels, as three separate processes at each end of S1, each in its own
    directory (a second process must never find the first's status and skip)."""
    units = G.session_units(v1, "S1")
    noise = [u for u in units if u.block.startswith("noise-")]
    assert {u.block for u in noise} == {f"noise-{t}-p{n}" for t in ("start", "end") for n in (1, 2, 3)}
    assert len({u.key for u in noise}) == len(noise) == 162   # 27 cells x 3 processes x 2 ends
    assert {u.engine for u in noise if u.level == "L0"} == {"sglang", "llamacpp"}  # the runners' own noise
    assert {u.engine for u in noise} == {"vllm", "sglang", "trtllm", "llamacpp"}
    assert {(u.engine, u.arm) for u in noise if u.arm != "defaults"} == {
        ("vllm", "async-off"), ("sglang", "overlap-off"), ("trtllm", "overlap-off"), ("llamacpp", "graphs-off"),
        ("llamacpp", "gpu-sampling")}
    assert {(u.engine, u.arm) for u in noise if u.arm != "defaults" and u.capture == "graph"} == {
        ("vllm", "async-off"), ("sglang", "overlap-off"), ("trtllm", "overlap-off"), ("llamacpp", "graphs-off"),
        ("llamacpp", "gpu-sampling")}
    large = {u.engine: u.batches for u in noise if u.model.endswith("8B") and u.capture in ("timing", "served")}
    assert set(large) == {"vllm", "sglang", "trtllm", "llamacpp"} and all(b == (1, 64, 256) for b in large.values())
    traced = {(u.engine, u.batches) for u in noise if u.capture == "graph" and u.arm == "defaults"}
    assert traced == {(e, (1, 64)) for e in ("vllm", "sglang", "trtllm", "llamacpp")}
    assert all(u.capture in ("timing", "served", "runner", "graph") for u in noise)
    blocks = [u.block for u in units]
    assert blocks.index("noise-start-p1") < blocks.index("bringup") < blocks.index("noise-end-p1")
    assert not any(u.block.startswith("noise-") for u in G.session_units(v1, "S2"))


def test_a_unit_runs_once_resumes_as_done_and_a_redo_keeps_its_first_attempt(tmp_path, monkeypatch, v1):
    """A finished unit is skipped on resume; reruns.csv names finished units whose repeats spread too far, and --redo
    runs them again, the first set aside."""
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "timing")
    calls = []

    def fake_run(argv, log, timeout, **_):
        calls.append(argv)
        return 0

    assert G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)[0]
    out = tmp_path / "S2-test" / unit.key
    assert is_done(out) and json.loads((out / "meta.json").read_text())["unit"]["key"] == unit.key
    G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None)
    assert len(calls) == 1                                          # done: skipped
    G.run_unit(unit, "S2-test", mounts, run=fake_run, idle=lambda: None, redo=True)
    assert len(calls) == 2 and (tmp_path / "S2-test" / f"{unit.key}.attempt-1").is_dir()
    from anatomy.tidy import unit_key
    assert unit_key(tmp_path / "S2-test" / unit.key, "S2-test") == unit.key


def runs_of(monkeypatch, argv):
    """The units main() would run (run_unit replaced), with whether each was a redo."""
    ran = []
    monkeypatch.setenv("DA_NSYS", "/n")
    monkeypatch.setattr(G.env, "host", lambda: {})
    monkeypatch.setattr(G, "run_unit", lambda u, *a, redo=False, **k: ran.append((u.key, redo)) or (True, 0.0))
    G.main(argv)
    return ran


def test_redo_runs_only_the_listed_units_and_the_sentinels(tmp_path, monkeypatch, v1):
    """On a fresh box nothing is done, so --redo must narrow the session itself, never fall back to all of it."""
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "timing")
    reruns = tmp_path / "reruns.csv"
    reruns.write_text(f"session,unit\nS2-x,{unit.key}\nS1-x,something/else\n")
    ran = runs_of(monkeypatch, [str(V1), "--session", "S2", "--session-id", "S2-x", "--run", "--no-gpu",
                                "--redo", str(reruns)])
    assert [k for k, redo in ran if redo] == [unit.key]
    # the sentinels run anew, in their own directories: a same-session redo must not find the first run's done
    assert all(k.startswith("sentinel-redo1-") for k, redo in ran if not redo) and len(ran) == 1 + 13
    with pytest.raises(SystemExit, match="no unit of session S2-y"):
        runs_of(monkeypatch, [str(V1), "--session", "S2", "--session-id", "S2-y", "--run", "--no-gpu",
                              "--redo", str(reruns)])
    # on another host the reruns go under a new session id, read from the old session's rows
    ran = runs_of(monkeypatch, [str(V1), "--session", "S2", "--session-id", "S2-z", "--run", "--no-gpu",
                                "--redo", str(reruns), "--redo-session", "S2-x"])
    assert [k for k, redo in ran if redo] == [unit.key]


def test_a_resumed_redo_does_not_pay_twice_for_the_units_it_already_redid(tmp_path, monkeypatch, v1):
    import os
    import time
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path / "runs")
    units = [u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "timing"][:2]
    reruns = tmp_path / "reruns.csv"
    reruns.write_text("session,unit\n" + "".join(f"S2-x,{u.key}\n" for u in units))
    os.utime(reruns, (time.time() - 60, time.time() - 60))
    G.run_unit(units[0], "S2-x", mounts, run=lambda *a, **k: 0, idle=lambda: None, redo=True)  # redone, then lost
    ran = runs_of(monkeypatch, [str(V1), "--session", "S2", "--session-id", "S2-x", "--run", "--no-gpu",
                                "--redo", str(reruns)])
    assert [k for k, redo in ran if redo] == [units[1].key]  # the first, finished after reruns.csv, stays done


def test_a_unit_that_raises_is_recorded_and_the_session_goes_on(tmp_path, monkeypatch):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setenv("DA_NSYS", "/n")
    monkeypatch.setattr(G.env, "host", lambda: {})
    monkeypatch.setattr(G.containers, "RUNTIME", "native")
    calls = []

    def flaky(u, *a, **k):
        calls.append(u.key)
        if len(calls) == 1:
            raise RuntimeError("GPU still busy after 180 s")
        return True, 0.0
    monkeypatch.setattr(G, "run_unit", flaky)
    monkeypatch.setattr(G.containers, "RUNTIME", "native")
    code = G.main([str(V1), "--session", "S3", "--session-id", "S3-x", "--run", "--no-gpu"])
    assert code == 1 and len(calls) == len(G.session_units(G.load(V1), "S3"))  # every unit after it still ran
    status = json.loads((tmp_path / "S3-x" / calls[0] / "status.json").read_text())
    assert not status["ok"] and "GPU still busy" in status["error"]


class FakeContention:
    seen = []

    def __init__(self, cpus, container=None):
        self.cpus, self.container = cpus, container

    def __enter__(self):
        FakeContention.seen.append((self.cpus, self.container))
        return self

    def __exit__(self, *exc):
        FakeContention.seen.append(("released", self.cpus))

    def problem(self):
        return None

    def record(self):
        return {"spinners": 4, "cpu_share": [0.5] * 4}


def test_a_hostload_unit_runs_under_contention_inside_its_container(tmp_path, monkeypatch, v1):
    from servebench.containers import cores
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    unit = next(u for u in G.expand(v1, blocks={"bringup"}) if u.engine == "vllm" and u.capture == "hostload")
    assert G.commands(unit, mounts(Path("/o")), "0-3")["contend"] == "0-3"
    graph = next(u for u in G.expand(v1, blocks={"bringup"}) if u.engine == "vllm" and u.capture == "graph")
    assert "contend" not in G.commands(graph, mounts(Path("/o")), "0-3")
    FakeContention.seen = []
    monkeypatch.setattr(G.containers, "Contention", FakeContention)
    G.run_unit(unit, "S1-x", mounts, cpuset="0-3", run=lambda *a, **k: 0, idle=lambda: None)
    meta = json.loads((tmp_path / "S1-x" / unit.key / "meta.json").read_text())
    # under Docker the loops run in the engine's own container: CFS shares cores between cgroups, not processes
    assert FakeContention.seen == [("0-3", unit.name), ("released", "0-3")]
    assert meta["contention"] == {"spinners": 4, "cpu_share": [0.5] * 4} and cores("0-2,5") == [0, 1, 2, 5]
    lines = render(G.commands(unit, mounts(Path("/o")), "0-3")["run"])
    assert "--cuda-graph-trace=graph" in lines and "osrt" not in lines and "--sample=none" in lines


@linux_affinity
def test_contention_spins_one_loop_per_core_measures_them_and_ends_them():
    import os

    from servebench.containers import Contention, spinners
    cpus = sorted(os.sched_getaffinity(0))[:2]
    with Contention(",".join(map(str, cpus)), period_s=0.2) as spin:
        import time
        time.sleep(0.5)
        assert len(spinners()) == len(cpus)
    assert spinners() == [] and spin.record()["spinners"] == len(cpus)
    assert len(spin.record()["cpu_share"]) == len(cpus) and all(0 < x <= 1.05 for x in spin.record()["cpu_share"])
    with Contention(None) as spin:
        assert spin.record() == {"spinners": 0, "cpu_share": []}


@linux_affinity
def test_native_spinners_die_with_the_grid_however_it_dies():
    """kill -9 of the grid must not leave busy loops slowing every unit after it (they carry PR_SET_PDEATHSIG)."""
    import os
    import signal
    import subprocess
    import sys
    import time

    from servebench.containers import spinners
    code = (f"import time\nfrom servebench.containers import Contention\n"
            f"with Contention('{sorted(os.sched_getaffinity(0))[0]}'):\n    time.sleep(30)")
    child = subprocess.Popen([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1])
    for _ in range(50):
        if spinners():
            break
        time.sleep(0.1)
    assert spinners()
    child.send_signal(signal.SIGKILL)
    child.wait()
    for _ in range(50):
        if not spinners():
            break
        time.sleep(0.1)
    assert spinners() == []


def test_spinners_are_found_by_their_name(tmp_path):
    from servebench.containers import SPIN_LOOP, SPIN_TAG, spinners

    def proc(pid, argv):
        d = tmp_path / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    proc(11, ["sh", "-c", SPIN_LOOP, SPIN_TAG])
    proc(13, ["sh", "-c", "while :; do :; done; other", "x"])  # not ours
    assert spinners(tmp_path) == [11]


def test_a_hostload_unit_whose_loops_never_ran_fails(tmp_path, monkeypatch, v1):
    """A hostload unit with no contention tested nothing: it fails (and goes to the reruns), never passes quietly."""
    class Idle(FakeContention):
        def problem(self):
            return "contention: no busy loop ran on the engine's cores"
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr(G.containers, "Contention", Idle)
    unit = next(u for u in G.expand(v1, blocks={"bringup"}) if u.engine == "vllm" and u.capture == "hostload")
    ok, _ = G.run_unit(unit, "S1-x", mounts, cpuset="0-3", run=lambda *a, **k: 0, idle=lambda: None)
    status = json.loads((tmp_path / "S1-x" / unit.key / "status.json").read_text())
    assert not ok and "no busy loop" in status["error"]


def test_the_contention_watcher_reports_a_docker_that_will_not_start_its_loops():
    import subprocess

    from servebench.containers import Contention

    def sh(argv, **kw):
        if "inspect" in argv:
            raise subprocess.TimeoutExpired(argv, 30)
        return subprocess.CompletedProcess(argv, 0, "", "")
    with Contention("0", container="da-x", sh=sh, period_s=0.05) as spin:
        assert not spin.wait_ready(0.3)  # inspect keeps timing out: never ready, and the watcher is still alive
        assert spin.thread.is_alive()
    assert spin.problem().startswith("contention: no busy loop ran")


def test_a_command_that_times_out_ends_with_its_whole_group(tmp_path):
    import os
    import subprocess

    from servebench.containers import run
    pidfile = tmp_path / "child"
    with pytest.raises(subprocess.TimeoutExpired):
        run(["sh", "-c", f"sleep 30 & echo $! > {pidfile}; wait"], timeout=0.5)
    child = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):  # AIPerf's workers, hf upload under sync_raw.sh: gone with it
        for _ in range(20):
            os.kill(child, 0)
            import time
            time.sleep(0.05)


def test_remove_is_bounded_in_time_and_a_hung_daemon_raises():
    import subprocess

    from servebench.containers import DockerHung, remove
    calls = []
    remove("da-x", sh=lambda argv, **kw: calls.append((argv, kw.get("timeout"))))
    assert calls == [(["docker", "rm", "-f", "da-x"], 120)]

    def hung(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])
    with pytest.raises(DockerHung):  # a daemon that cannot remove a container: the box is broken, the grid counts it
        remove("da-x", sh=hung)


def test_a_gpu_that_cannot_be_queried_is_not_idle():
    from servebench import gpu
    with pytest.raises(gpu.GPUBusy, match="no answer"):
        gpu.wait_for_idle_gpu(state=lambda: None)


def main_with(monkeypatch, tmp_path, flaky, argv):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setenv("DA_NSYS", "/n")
    monkeypatch.setattr(G.env, "host", lambda: {})
    monkeypatch.setattr(G.containers, "RUNTIME", "native")
    monkeypatch.setattr(G, "run_unit", flaky)
    return G.main([str(V1), *argv, "--run", "--no-gpu"])


def test_a_busy_gpu_is_cleared_and_the_unit_tried_once_more(tmp_path, monkeypatch, capsys):
    from servebench import gpu
    calls, cleared = [], []
    monkeypatch.setattr(G.containers, "clear", lambda: cleared.append(1))

    def flaky(u, *a, **k):
        calls.append(u.key)
        if len(calls) == 1:
            raise gpu.GPUBusy("GPU still busy after 180 s")
        return True, 0.0
    code = main_with(monkeypatch, tmp_path, flaky, ["--session", "S3", "--session-id", "S3-x"])
    assert code == 0 and cleared == [1] and calls[0] == calls[1]   # the same unit, once more, after the clearing


def test_a_box_whose_units_all_raise_ends_the_session(tmp_path, monkeypatch, capsys):
    calls = []

    def broken(u, *a, **k):
        calls.append(u.key)
        raise RuntimeError("the GPU has fallen off the bus")
    monkeypatch.setattr(G.containers, "clear", lambda: None)
    code = main_with(monkeypatch, tmp_path, broken, ["--session", "S3", "--session-id", "S3-x"])
    out = capsys.readouterr().out
    assert code == 2 and len(calls) == G.GIVE_UP_AFTER and '"ended_early"' in out and "units in a row" in out


def test_sigterm_unwinds_the_grid_ends_the_running_engine_and_restores_the_handlers(tmp_path, monkeypatch):
    """A SIGTERM while a unit's engine runs (in its own session, so it gets no signal) must end that engine too."""
    import os
    import signal
    import threading
    import time
    before = signal.getsignal(signal.SIGTERM)
    pidfile = tmp_path / "engine"

    def engine(u, *a, **k):  # the real execute, as run_unit calls it, on a child that would outlive the grid
        threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        return G.execute(["sh", "-c", f"echo $$ > {pidfile}; exec sleep 60"], tmp_path / "log", timeout=30), 0.0
    with pytest.raises(SystemExit) as e:
        main_with(monkeypatch, tmp_path, engine, ["--session", "S3", "--session-id", "S3-x"])
    assert e.value.code == 128 + signal.SIGTERM and signal.getsignal(signal.SIGTERM) == before
    pid = int(pidfile.read_text())
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):  # the engine went with the grid
        os.kill(pid, 0)


def test_units_that_run_out_of_time_in_a_row_end_the_session(tmp_path, monkeypatch, capsys):
    calls = []

    def stuck(u, *a, **k):
        calls.append(u.key)
        out = G.unit_dir("S3-x", u)
        out.mkdir(parents=True, exist_ok=True)
        G.write_status(out, False, 3600.0, "exit 124")
        return False, 3600.0
    code = main_with(monkeypatch, tmp_path, stuck, ["--session", "S3", "--session-id", "S3-x"])
    out = capsys.readouterr().out
    # S3 opens on three vLLM sentinels: one engine's run of timeouts does not end it; the fourth, SGLang's, does
    assert code == 2 and len(calls) == G.GIVE_UP_AFTER + 1 and "ran out of time" in out


def test_one_engine_timing_out_unit_after_unit_does_not_end_the_session(tmp_path, monkeypatch, capsys):
    calls = []

    def stuck_trtllm(u, *a, **k):  # one engine hanging on everything: the others and the end sentinels still run
        calls.append(u.key)
        out = G.unit_dir("S3-x", u)
        out.mkdir(parents=True, exist_ok=True)
        hung = u.engine == "trtllm"
        G.write_status(out, not hung, 3600.0 if hung else 1.0, "exit 124" if hung else None)
        return not hung, 3600.0 if hung else 1.0
    main_with(monkeypatch, tmp_path, stuck_trtllm, ["--session", "S3", "--session-id", "S3-x"])
    assert "the session ends here" not in capsys.readouterr().out
    assert any("sentinel" in k and "end" in k for k in calls)


def test_a_mistyped_model_is_refused_by_name(v1, capsys):
    with pytest.raises(SystemExit, match="Qwen3-32b"):
        G.main([str(V1), "--session", "S2", "--models", "Qwen/Qwen3-14B", "Qwen/Qwen3-32b", "--list"])


def test_failed_uploads_are_retried_and_units_that_did_not_run_are_not_uploaded(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DA_SYNC", "1")
    monkeypatch.setenv("DATASET", "x/y")
    uploads, healthy = [], [False]
    monkeypatch.setattr(G, "SYNC_RETRY_EVERY", 2)

    def sync(key, session, versions):
        uploads.append(key)
        return healthy[0]
    monkeypatch.setattr(G, "sync", sync)
    ran = []

    def unit(u, *a, **k):
        ran.append(u.key)
        if len(ran) == 3:
            healthy[0] = True                       # the Hub comes back
        return True, (0.0 if len(ran) == 1 else 5.0)  # the first was done already: nothing to upload
    main_with(monkeypatch, tmp_path, unit, ["--session", "S3", "--session-id", "S3-x"])
    assert ran[0] not in uploads                                       # done before: never uploaded again
    assert set(ran[1:]) <= set(uploads) and uploads.count(ran[1]) >= 2  # the failed one, tried again and landed


def test_models_narrow_a_session_to_a_lost_hosts_unfinished_models(v1):
    units = G.session_units(v1, "S2", models=["Qwen/Qwen3-14B", "Qwen/Qwen3-32B"])
    blocks = [u for u in units if not u.block.startswith("sentinel-")]
    assert blocks and {u.model for u in blocks} == {"Qwen/Qwen3-14B", "Qwen/Qwen3-32B"}
    assert [u for u in units if u.block.startswith("sentinel-")] == \
        [u for u in G.session_units(v1, "S2") if u.block.startswith("sentinel-")]  # the sentinels stay whole


def test_engines_take_turns_both_ways_round(v1):
    """Drift over a long session must not side with one engine at every model: the order flips model by model."""
    units = [u for u in G.session_units(v1, "S2") if not u.block.startswith("sentinel-")]
    order = {}
    for u in units:
        order.setdefault(u.model, [])
        if u.engine not in order[u.model]:
            order[u.model].append(u.engine)
    first, second = order["Qwen/Qwen3-0.6B"], order["Qwen/Qwen3-1.7B"]
    assert first == sorted(first) and second == sorted(second, reverse=True)


def test_a_refresh_is_s2_again_for_the_engines_that_changed(tmp_path):
    spec = tmp_path / "r.yaml"
    spec.write_text("versions: 2026-09\nrefresh: {grid: configs/grid/validation/four-engines.yaml, session: S2, as: R, "
                    "engines: [llamacpp]}\n")
    r = G.load(spec)
    units = G.session_units(r, "R")
    work = [u for u in units if not u.block.startswith("sentinel-")]
    assert {u.engine for u in work} == {"llamacpp"} and {u.block for u in work} >= {"main", "mitigations", "runner"}
    assert {u.model for u in work if u.block == "main"} == set(r["qwen3"])                  # every model and batch
    assert {u.engine for u in units if u.block == "sentinel-start"} == {"vllm", "sglang", "trtllm", "llamacpp"}
    assert not any(u.block.startswith("noise-") for u in units)
    spec.write_text("versions: 2026-09\nrefresh: {grid: configs/grid/validation/pod.yaml, session: S2, as: R, "
                    "engines: [vllm]}\n")  # the pod grid extends the study's, which runs no llama.cpp
    assert {u.engine for u in G.session_units(G.load(spec), "R") if u.block == "main"} == {"vllm"}  # one loader


def test_a_gpu_that_never_idles_leaves_no_attempt_behind(tmp_path, monkeypatch, v1):
    from servebench import gpu
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    unit = next(u for u in G.expand(v1, blocks={"main"}) if u.engine == "vllm" and u.capture == "timing")

    def busy():
        raise gpu.GPUBusy("GPU still busy after 180 s")
    with pytest.raises(gpu.GPUBusy):
        G.run_unit(unit, "S2-x", mounts, run=lambda *a, **k: 0, idle=busy)
    assert not (tmp_path / "S2-x").exists()  # nothing written: the retry after clearing starts clean


def test_an_unlocked_clock_stops_the_study_unless_every_session_runs_unlocked(tmp_path, monkeypatch):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setenv("DA_NSYS", "/n")
    monkeypatch.setattr(G.env, "gpu_query", lambda *a, **k: {"name": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setattr(G.gpu, "lock_clocks", lambda mhz: False)
    ran = []
    monkeypatch.setattr(G, "run_unit", lambda u, *a, **k: ran.append(u.key) or (True, 1.0))
    monkeypatch.setattr(G.containers, "RUNTIME", "native")
    argv = [str(STUDY), "--session", "S3", "--session-id", "S3-x", "--run"]
    with pytest.raises(SystemExit, match="could not be locked"):
        G.main(argv)
    assert ran == []
    monkeypatch.setenv("DA_ALLOW_UNLOCKED", "1")
    monkeypatch.setattr(G.env, "host", lambda: {})
    monkeypatch.setattr(G.containers, "RUNTIME", "native")
    G.main(argv)
    assert ran


def test_an_engine_hanging_on_a_model_has_its_other_units_there_skipped(tmp_path, monkeypatch):
    called = []

    def hangs(u, *a, **k):  # llama.cpp times out on every grid unit; everything else, sentinels included, works
        called.append(u.key)
        out = G.unit_dir("S2-x", u)
        out.mkdir(parents=True, exist_ok=True)
        hung = u.engine == "llamacpp" and not u.block.startswith(("sentinel-", "noise-"))
        G.write_status(out, not hung, 3600.0 if hung else 1.0, "exit 124" if hung else None)
        return not hung, 3600.0 if hung else 1.0
    main_with(monkeypatch, tmp_path, hangs, ["--session", "S2", "--session-id", "S2-x", "--models", "Qwen/Qwen3-0.6B"])
    grid_llama = [k for k in called if k.startswith("llamacpp")]
    assert len(grid_llama) == G.HUNG_AFTER                                   # two hours spent, not one per unit
    units = [u for u in G.session_units(G.load(V1), "S2", models=["Qwen/Qwen3-0.6B"])
             if u.engine == "llamacpp" and not u.block.startswith(("sentinel-", "noise-"))]
    skipped = [u for u in units if u.key not in called]
    assert skipped and all("skipped" in json.loads((G.unit_dir("S2-x", u) / "status.json").read_text())["error"]
                           for u in skipped)
    assert not any(G.is_done(G.unit_dir("S2-x", u)) for u in skipped)       # failed, not done: a resume runs them
    assert any(k.startswith("sentinel-end/llamacpp") for k in called)        # the closing sentinels still run


def test_progress_names_the_models_a_new_host_must_finish(tmp_path, monkeypatch, v1, capsys):
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    for u in G.session_units(v1, "S2"):
        if u.model == "Qwen/Qwen3-0.6B" and not u.block.startswith(("sentinel-", "noise-")):
            out = G.unit_dir("S2-x", u)
            out.mkdir(parents=True, exist_ok=True)
            G.write_status(out, True, 1.0)
    G.main([str(V1), "--session", "S2", "--session-id", "S2-x", "--progress"])
    out = capsys.readouterr().out
    assert "--models Qwen/Qwen3-1.7B" in out and "Qwen/Qwen3-0.6B " not in out.splitlines()[-1]


def test_noise_and_sentinel_units_are_never_rerun(tmp_path, v1):
    units = G.session_units(v1, "S1")
    noisy = next(u for u in units if u.block.startswith("noise-"))
    grid_unit = next(u for u in units if u.block == "bringup")
    csv = tmp_path / "reruns.csv"
    csv.write_text(f"session,unit,batches,ranges\nS1-x,{noisy.key},1,x\nS1-x,{grid_unit.key},1,x\n")
    picked, listed = G.narrow_to_reruns(units, csv, "S1-x")
    assert listed == {grid_unit.key} and noisy.key not in {u.key for u in picked}


def test_a_cpu_quota_caps_the_cores_used(tmp_path):
    assert G.cpusets(range(64), quota=6) == ("0-3", "4-5")       # a pod: 64 CPUs seen, 6.8 paid for


def test_a_validation_grid_can_extend_the_study_grid_unchanged():
    pod, study = G.load(G.ROOT / "configs/grid/validation/pod.yaml"), G.load(STUDY)
    assert pod["validation"] is True and pod["clock_mhz"] is None
    assert all(pod[k] == study[k] for k in ("blocks", "noise", "sentinels"))
    assert [u.key for u in G.session_units(pod, "S2")] == [u.key for u in G.session_units(study, "S2")]


def test_a_grid_extending_an_extension_gets_both_and_a_loop_is_refused(tmp_path):
    probe = tmp_path / "probe.yaml"
    probe.write_text("extends: configs/grid/validation/pod.yaml\nblocks: {}\n")
    grid = G.load(probe)
    assert grid["validation"] is True and grid["versions"] == G.load(V1)["versions"] and grid["blocks"] == {}
    (tmp_path / "a.yaml").write_text(f"extends: {tmp_path / 'b.yaml'}\n")
    (tmp_path / "b.yaml").write_text(f"extends: {tmp_path / 'a.yaml'}\n")
    with pytest.raises(ValueError, match="loop"):
        G.load(tmp_path / "a.yaml")


def test_engines_narrows_the_units_and_refuses_an_unknown_one(capsys):
    G.main([str(V1), "--block", "scale", "--engines", "vllm", "sglang", "--list"])
    out = capsys.readouterr().out
    assert "vllm-" in out and "sglang-" in out and "trtllm-" not in out and "llamacpp-" not in out
    with pytest.raises(SystemExit, match="vlm"):
        G.main([str(V1), "--block", "scale", "--engines", "vlm", "--list"])


def test_every_validation_probe_grid_loads_and_has_units():
    for path in sorted((G.ROOT / "configs/grid/validation/probes").glob("*.yaml")):
        grid = G.load(path)
        blocks = set(grid["blocks"])
        assert grid["validation"] is True and G.expand(grid, blocks=blocks), path.name


def test_a_probe_grid_is_validation_data_and_keeps_its_units_apart():
    """A probe grid marks every unit validation, its overrides reach the command, and its units never take an S1
    unit's directory."""
    slots = G.load(G.ROOT / "configs/grid/validation/probes/llamacpp-slots.yaml")
    assert slots.get("validation")  # a probe is never study data
    (u,) = [u for u in G.expand(slots, blocks={"probe-slots"})]
    c = G.commands(u, mounts(Path("/o")), overrides=slots["overrides"])
    assert list(c["servers"]) == ["1,8,64"] and "-np 64" in render(c["servers"]["1,8,64"])  # one server, as before
    # S1 runs the same model's llama.cpp unit in its own blocks: the probes' units must never take its directory (a
    # probe run after S1's grid would find status.ok and do nothing; run before, it would skip S1's own cells)
    nosim = G.load(G.ROOT / "configs/grid/validation/probes/llamacpp-slots-no-similarity.yaml")
    (w,) = G.expand(nosim, blocks={"probe-slots-no-similarity"})
    s1 = {x.key for x in G.expand(G.load(G.ROOT / "configs/grid/v1.yaml"), sessions={"S1"})}
    assert u.key != w.key and not {u.key, w.key} & s1


def test_a_release_that_asks_gets_a_server_per_batch(tmp_path, monkeypatch, v1):
    """llama.cpp's server slowed at later batches after serving batch 1: one unit, one fresh server per batch, one
    deadline across them; the other engines keep one server for the unit."""
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    calls = []

    def served(unit, cmds, out, timeout=None, probe=True):
        calls.append((unit.engine, unit.batches, timeout, cmds["server"], probe))
        return "B=8: boom" if unit.batches == (8,) else None
    units = [u for u in G.expand(v1, blocks={"main"}) if u.capture == "served"]
    lc = next(u for u in units if u.engine == "llamacpp")
    ok, _ = G.run_unit(lc, "S2-test", mounts, idle=lambda: None, served=served, timeout=600)
    assert [c[1] for c in calls] == [(b,) for b in lc.batches] and all(0 < c[2] <= 600 for c in calls)
    assert [c[4] for c in calls] == [False] * (len(calls) - 1) + [True]  # the agreement probe on the last server
    for (_, (b,), _, argv, _) in calls:  # each server a slot per sequence of its own batch, the same context per slot
        assert argv[argv.index("-np") + 1] == str(b) and argv[argv.index("-c") + 1] == str(b * (896 + 288 + 64))
    meta = json.loads((tmp_path / "S2-test" / lc.key / "meta.json").read_text())
    assert set(meta["commands"]["servers"]) == {str(b) for b in lc.batches}  # every server's command recorded
    assert not ok and "B=8: boom" in json.loads((tmp_path / "S2-test" / lc.key / "status.json").read_text())["error"]
    calls.clear()
    vllm = next(u for u in G.expand(v1, blocks={"served"}) if u.engine == "vllm" and u.capture == "served")
    G.run_unit(vllm, "S2-test", mounts, idle=lambda: None, served=served)
    assert [c[1] for c in calls] == [vllm.batches]


def test_a_study_grid_does_not_start_on_a_model_off_its_pin(monkeypatch, tmp_path):
    """Every engine loads what refs/main names; a unit on another snapshot would be rejected after it was paid for."""
    import servebench.env as E
    hub = tmp_path / "hub" / "models--Qwen--Qwen3-0.6B" / "refs"
    hub.mkdir(parents=True)
    (hub / "main").write_text("0" * 40)
    monkeypatch.setattr(E, "hub_dir", lambda repo: tmp_path / "hub" / f"models--{repo.replace('/', '--')}")
    (problem,) = PIN_PROBLEMS(["Qwen/Qwen3-0.6B"])
    assert problem.startswith("Qwen/Qwen3-0.6B: refs/main is 0000") and E.pinned("Qwen/Qwen3-0.6B") in problem
    monkeypatch.setattr(G.env, "pin_problems", lambda models: [f"{m}: off" for m in models])
    monkeypatch.setenv("DA_NSYS", "/n")
    with pytest.raises(SystemExit, match="off their pins"):
        G.main([str(G.ROOT / "configs/grid/v1.yaml"), "--session", "S1", "--run", "--no-gpu"])


def test_uploads_can_run_every_few_units_and_always_after_the_last(tmp_path, monkeypatch):
    """One HF commit per unit idles the GPU for its duration: DA_SYNC_EVERY batches the per-unit uploads, and the last
    unit's upload always runs, so a session ends with everything it ran on the Hub (session.sh syncs once more)."""
    monkeypatch.setenv("DA_SYNC", "1")
    monkeypatch.setenv("DATASET", "x/y")
    monkeypatch.setattr(G, "SYNC_EVERY", 4)
    ran, uploaded, at = [], [], []

    def sync(key, session, versions):
        uploaded.append(key)
        at.append(len(ran))  # how many units had run when this upload happened
        return True
    monkeypatch.setattr(G, "sync", sync)
    main_with(monkeypatch, tmp_path, lambda u, *a, **k: ran.append(u.key) or (True, 5.0),
              ["--session", "S3", "--session-id", "S3-x"])
    assert set(uploaded) == set(ran)                                   # every unit that ran was uploaded
    assert all(n % 4 == 0 or n == len(ran) for n in at) and at[-1] == len(ran)   # in rounds of four, and at the end


def test_spinners_are_found_whatever_shell_runs_the_loop(tmp_path):
    """TensorRT-LLM's image: `sh` is a wrapper that execs /bin/bash, so the loops read `/bin/bash -c <loop> da-spin`
    on the host; a match on the shell's name saw none, and the hostload unit failed for want of proof while its loops
    ran (an H100 PCIe VM)."""
    from servebench.containers import SPIN_LOOP, SPIN_TAG, spinners
    proc = tmp_path / "proc"
    (proc / "uptime").parent.mkdir()
    for pid, argv in ((101, ["taskset", "-c", "3", "sh", "-c", SPIN_LOOP, SPIN_TAG]),
                      (102, ["sh", "-c", SPIN_LOOP, SPIN_TAG]),
                      (103, ["/bin/bash", "-c", SPIN_LOOP, SPIN_TAG]),
                      (104, ["sh", "-c", "sleep 1", SPIN_TAG]),  # another command carrying the tag: not a loop
                      (105, ["sh", "-c", SPIN_LOOP])):  # the loop without the tag: not ours
        (proc / str(pid)).mkdir()
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    assert spinners(proc) == [101, 102, 103]
