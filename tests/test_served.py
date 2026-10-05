import json
import shutil
import subprocess
from pathlib import Path

import pytest

from anatomy.served import run_served, wait_ready
from servebench.aiperf import aiperf_cmd
from servebench.results import Unit
from tests.conftest import linux_affinity

FIXTURE = Path(__file__).parent / "fixtures" / "aiperf_raw_llamacpp_cpu.jsonl"
CMDS = {"server": ["docker", "run", "--rm", "--name", "srv", "-d", "img"], "ready": "http://127.0.0.1:9/health",
        "files": {"opts.yaml": "a: 1\n"}}


def unit(capture="served", batches=(4,)):
    """A served unit as the AIPerf fixture ran it (48 tokens); a profiled one at the grid's 288 for its window."""
    return Unit(engine="llamacpp", version="v0.5.0", arm="defaults", level="L2", model="Qwen/Qwen3-0.6B",
                capture=capture, batches=batches, prompt_len=64, gen_len=48 if capture == "served" else 288)


class Shell:
    """Fake subprocess.run: docker succeeds, aiperf writes the real CPU-rehearsal raw export into its artifact dir."""

    def __init__(self, start_code=0, aiperf_code=0, running=True):
        self.calls, self.start_code, self.aiperf_code, self.running = [], start_code, aiperf_code, running

    def __call__(self, argv, **kw):
        self.calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(argv, self.start_code, "", "boom" if self.start_code else "")
        if argv[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "true\n" if self.running else "false\n", "")
        if "profile" in argv and any("aiperf" in a for a in argv):
            art = Path(argv[argv.index("--artifact-dir") + 1])
            art.mkdir(parents=True)
            shutil.copy(FIXTURE, art / "profile_export_raw.jsonl")
            return subprocess.CompletedProcess(argv, self.aiperf_code)
        return subprocess.CompletedProcess(argv, 0, "", "")


class TimedShell(Shell):
    """A shell that honours timeouts as subprocess does (one of zero expires at once) and keeps each call's beside it:
    the plain Shell ignores them, which hid a cleanup bounded by a deadline already passed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeouts = []

    def __call__(self, argv, **kw):
        self.timeouts.append((tuple(argv[:3]), kw.get("timeout")))
        if kw.get("timeout") is not None and kw["timeout"] <= 0:
            raise subprocess.TimeoutExpired(argv, kw["timeout"])
        return super().__call__(argv, **kw)


def records(out):
    return [json.loads(line) for line in (out / "driver.jsonl").read_text().splitlines()]


def test_a_server_part_with_no_time_left_is_a_timeout_and_never_touches_docker(tmp_path):
    """llama.cpp's later servers (one per batch) start with what the earlier ones left; with nothing left the part's
    batches are deadline errors (what anatomy.grid.timed_out reads), not a docker call with a zero timeout that the
    grid would count as a broken box."""
    sh = TimedShell()
    err = run_served(unit(batches=(4,)), CMDS, tmp_path, sh=sh, ready=lambda *a, **k: True, timeout=0.0)
    assert err == "B=4: the unit's deadline passed before its server started" and sh.calls == []
    assert [(r["kind"], r["batch"]) for r in records(tmp_path)] == [("error", 4)]


def test_cleanup_and_liveness_keep_their_own_bounds_whatever_the_deadline(tmp_path):
    from anatomy.served import CLEANUP_TIMEOUT_S
    sh = TimedShell()
    err = run_served(unit(batches=(4,)), CMDS, tmp_path, sh=sh, ready=lambda url, wait, alive: alive() and False,
                     timeout=1.0)
    assert err.startswith("server not ready")
    bounds = dict(sh.timeouts)
    assert bounds[("docker", "rm", "-f")] == bounds[("docker", "run", "--rm")] == CLEANUP_TIMEOUT_S
    assert bounds[("docker", "inspect", "-f")] == 30


def test_served_sweep_writes_l1_shaped_records(tmp_path):
    sh = Shell()
    err = run_served(unit(), CMDS, tmp_path, sh=sh, ready=lambda *a, **k: True, repeats=2, margin=4,
                     client_cpuset="12-15")
    assert err is None and (tmp_path / "opts.yaml").read_text() == "a: 1\n"
    recs = records(tmp_path)
    longs = [r for r in recs if r["kind"] == "long"]
    assert len(longs) == 2 and all(r["tokens"] == [48] * 4 and r["cadence"]["step_ns"] > 0 for r in longs)
    aiperf = [c for c in sh.calls if "profile" in c]
    assert aiperf[0][:3] == ["taskset", "-c", "12-15"] and "--concurrency" in aiperf[0]
    assert ["docker", "rm", "-f", "srv"] in sh.calls  # the server is always removed
    kinds = [(r["kind"], r.get("batch")) for r in recs]
    assert kinds.count(("short", 4)) == 2 and ("differential", 4) in kinds  # long and short waves: a differential
    assert all("cadence_tail" in r for r in longs)  # the differential is judged over its own tokens, as at L1


def test_server_that_never_starts(tmp_path):
    assert "did not start" in run_served(unit(), CMDS, tmp_path, sh=Shell(start_code=125))


def test_server_that_exits_fails_fast(tmp_path):
    sh = Shell(running=False)
    err = run_served(unit(), CMDS, tmp_path, sh=sh, ready=lambda url, t, alive: wait_ready(url, t, alive=alive))
    assert "not ready" in err and ["docker", "rm", "-f", "srv"] in sh.calls


def test_one_failing_batch_does_not_stop_the_sweep(tmp_path):
    err = run_served(unit(batches=(1, 4)), CMDS, tmp_path, sh=Shell(aiperf_code=3), ready=lambda *a, **k: True,
                     repeats=1)
    assert "B=1" in err and "B=4" in err
    assert [r["batch"] for r in records(tmp_path) if r["kind"] == "error"] == [1, 4]


class FakeWave:
    made = []

    def __init__(self, url, prompts, gen_len, model, extra=None):
        self.chunks, self.errors = [[] for _ in prompts], []
        self.epoch_offset_ns, self.gen_len = 0, gen_len
        FakeWave.made.append(extra)

    def start(self, timeout=None):
        return self

    def join(self):
        return self


def test_profile_wave_opens_and_closes_an_nsys_window(tmp_path):
    sh = Shell()
    err = run_served(unit(capture="graph"), CMDS, tmp_path, sh=sh, ready=lambda *a, **k: True, wave_cls=FakeWave)
    assert err is None
    execs = [c for c in sh.calls if c[:2] == ["docker", "exec"]]
    assert [c[4] for c in execs] == ["start", "stop"] and "/out/graph-b4" in execs[0]  # one capture around the wave
    rec = records(tmp_path)[-1]
    assert rec["kind"] == "profile" and rec["window_from_end"] == 288 - 112 and rec["profile_steps"] == 64


def test_request_fields_reach_both_clients(tmp_path):
    """A release's request fields (llama.cpp: cache_prompt false) go to AIPerf and to the profile wave alike."""
    cmds = {**CMDS, "request": {"cache_prompt": False}}
    run_served(unit(capture="graph"), cmds, tmp_path, sh=Shell(), ready=lambda *a, **k: True, wave_cls=FakeWave)
    assert FakeWave.made[-1] == {"cache_prompt": False}
    cmd = aiperf_cmd("http://x", "m", "m", 4, 896, 288, "/a", extra_inputs=cmds["request"])
    extras = json.loads(cmd[cmd.index("--extra-inputs") + 1])
    assert extras == {"ignore_eos": True, "temperature": 0, "cache_prompt": False}


def test_llama_server_metrics_give_an_independent_step():
    from servebench.aiperf import server_metrics, server_step_ns
    m = server_metrics(Path(__file__).parent / "fixtures" / "aiperf-0.13.0-server-metrics-llamacpp-b8.csv")
    assert m["llamacpp:n_decode"]["total"] == 288                        # 8 slots in lockstep: one decode per step
    assert server_step_ns(m) / 1e6 == pytest.approx(13.9851 / 2304 * 1e3)  # 6.07 ms; the cadence read 6.09
    assert server_step_ns({}) is None


def test_client_whole_request_time_per_token():
    from servebench.aiperf import client_whole_ns
    seqs = [[(0, 1), (100, 2), (300, 4)], [(50, 1), (250, 3)], [(7, 1)]]   # (ns, tokens so far); a 1-chunk stream
    assert client_whole_ns(seqs) == pytest.approx((300 + 200) / (3 + 2))


def test_vllm_server_metrics_give_its_inter_token_latency():
    from servebench.aiperf import server_metrics, server_step_ns
    m = server_metrics(Path(__file__).parent / "fixtures" / "aiperf-0.13.0-server-metrics-vllm-b8.csv")
    itl = m["vllm:inter_token_latency_seconds"]
    assert itl["count"] == 2296                                   # 8 streams x 287 intervals
    assert server_step_ns(m) / 1e6 == pytest.approx(itl["sum"] / 2296 * 1e3)   # 3.480 ms; the CSV's avg says 3.5


def test_token_counting_across_server_chunk_shapes():
    from servebench.stream_client import advance
    merged = {"choices": [{"text": "ab", "token_ids": [11, 12]}]}           # vLLM a step behind: two tokens, one chunk
    assert advance(5, merged) == 7
    usage = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 9}}  # TRT-LLM continuous usage: absolute
    assert advance(7, usage) == 9
    closing = {"choices": [], "usage": {"completion_tokens": 9}}             # the usage-only last message: no token
    assert advance(9, closing) == 9
    assert advance(3, {"choices": [{"text": "", "finish_reason": "length"}]}) == 3   # llama.cpp's closing chunk
    assert advance(3, {"choices": [{"text": "y"}]}) == 4
    held = {"choices": [{"text": "é"}], "timings": {"predicted_n": 12}}      # llama-server: two tokens, one chunk
    assert advance(10, held) == 12


def test_request_objects_reach_aiperf_as_json():
    cmd = aiperf_cmd("http://x", "m", "m", 4, 896, 288, "/a",
                     extra_inputs={"stream_options": {"include_usage": True, "continuous_usage_stats": True}})
    assert cmd.count("--extra-inputs") == 1  # one JSON object: AIPerf splits key:value pairs at commas
    extras = json.loads(cmd[cmd.index("--extra-inputs") + 1])
    assert extras["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}


def test_native_server_lifecycle(tmp_path):
    from anatomy.served import NativeServer
    cmds = {"server": ["python3", "-c", "import time; print('up', flush=True); time.sleep(60)"],
            "paths": [("/out", str(tmp_path))]}
    calls = []
    s = NativeServer(cmds, tmp_path, lambda argv, **k: calls.append(argv))
    assert s.start() is None and s.alive()
    s.exec(["/opt/x", "-o", "/out/graph-b1"])
    import time
    for _ in range(100):  # the server has started when its first line is in the log
        if "up" in (tmp_path / "container.log").read_text():
            break
        time.sleep(0.05)
    assert calls[-1] == ["/opt/x", "-o", f"{tmp_path}/graph-b1"]           # container paths become host paths
    s.stop()
    assert not s.alive() and "up" in (tmp_path / "container.log").read_text()


def test_a_failing_nsys_command_is_a_batch_error(tmp_path):
    class NsysFails(Shell):
        def __call__(self, argv, **kw):
            if "exec" in argv and "start" in argv:
                self.calls.append(argv)
                return subprocess.CompletedProcess(argv, 1, "", "no such session")
            return super().__call__(argv, **kw)
    err = run_served(unit(capture="graph"), CMDS, tmp_path, sh=NsysFails(), ready=lambda *a, **k: True,
                     wave_cls=FakeWave)
    assert err and "exit 1" in err  # nsys start failed: a batch error, never a silent profile with no trace


def test_a_native_server_under_nsys_launch_has_its_session_shut_down(tmp_path):
    """The application of `nsys launch` belongs to nsys's session, not our process group: stop ends the session."""
    from anatomy.served import NativeServer
    sleep = ["python3", "-c", "import time; time.sleep(60)"]
    cmds = {"server": sleep, "paths": [("/opt/nsys", "/srv/nsys")], "port": None, "nsys_session": True}
    calls = []
    s = NativeServer(cmds, tmp_path, lambda argv, **k: calls.append(argv))
    s.start()
    s.stop()
    assert calls and calls[-1][:2] == ["/srv/nsys/bin/nsys", "shutdown"] and "--kill=sigkill" in calls[-1]


def test_a_port_held_by_time_wait_is_not_free():
    """A killed server's accepted connections linger in TIME_WAIT with nothing listening: the port only counts as free
    once it binds (in validation, vLLM's bind failed for a minute after TensorRT-LLM's server was killed)."""
    import socket

    from anatomy.served import port_free
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    assert not port_free(port, wait_s=1, sleep=lambda s: None)  # listening
    listener.close()
    conn.close()  # the server side closes first: its end of the connection waits in TIME_WAIT on the port
    client.close()
    assert not port_free(port, wait_s=1, sleep=lambda s: None)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        unused = s.getsockname()[1]
    assert port_free(unused, wait_s=1, sleep=lambda s: None) and port_free(None)


def test_a_profile_wave_whose_streams_failed_fails_its_unit(tmp_path):
    """The trace of a wave that never decoded is no window (in validation, TRT-LLM answered 400 to every request, the
    node and hostload units read ok, and their cells had no kernel time)."""
    class Refused(FakeWave):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if self.gen_len == 288:  # the warm-up succeeded, the profiled wave failed
                self.errors = [(i, "RuntimeError('400 tokenizer')") for i in range(len(self.chunks))]
    err = run_served(unit(capture="node"), CMDS, tmp_path, sh=Shell(), ready=lambda *a, **k: True, wave_cls=Refused)
    assert err and "4 of 4 streams failed" in err and "400 tokenizer" in err
    assert [r["kind"] for r in records(tmp_path)][-2:] == ["profile", "error"]  # the trace is still described


def test_a_unit_past_its_deadline_records_its_batches_as_errors(tmp_path):
    """A hung client or server cannot stall an unattended session: what is left of the unit's deadline bounds every
    call, and a batch that would start after it is an error, not a wait."""
    now = [0.0]

    def ready(*args, **kwargs):
        now[0] = 100.0
        return True
    sh = Shell()
    err = run_served(unit(batches=(1, 4)), CMDS, tmp_path, sh=sh, ready=ready, timeout=50,
                     clock=lambda: now[0])
    assert err.count("deadline passed") == 2 and not [c for c in sh.calls if "profile" in c]
    assert ["docker", "rm", "-f", "srv"] in sh.calls


def test_a_hostload_unit_contends_inside_the_server_once_it_is_ready(tmp_path, monkeypatch):
    order = []

    class Contention:
        def __init__(self, cpus, container=None):
            self.cpus, self.container = cpus, container

        def __enter__(self):
            order.append(("contend", self.cpus, self.container))
            return self

        def __exit__(self, *exc):
            order.append(("released", self.cpus))

        def wait_ready(self, timeout):
            order.append(("loops up",))
            return True

        def problem(self):
            return None

        def record(self):
            return {"spinners": 2, "cpu_share": [0.5, 0.5]}
    monkeypatch.setattr("anatomy.served.Contention", Contention)
    cmds = {**CMDS, "contend": "0-1"}
    err = run_served(unit(capture="hostload"), cmds, tmp_path, sh=Shell(), wave_cls=FakeWave,
                     ready=lambda *a, **k: order.append(("ready",)) or True)
    assert err is None and order == [("ready",), ("contend", "0-1", "srv"), ("loops up",), ("released", "0-1")]
    # recorded per server under its batches: a server per batch (llama.cpp) keeps every server's loops, not the last's
    contention = json.loads((tmp_path / "meta.json").read_text())["contention"]
    assert contention == {"4": {"spinners": 2, "cpu_share": [0.5, 0.5]}}
    run_served(unit(capture="hostload", batches=(64,)), cmds, tmp_path, sh=Shell(), wave_cls=FakeWave,
               ready=lambda *a, **k: True)
    contention = json.loads((tmp_path / "meta.json").read_text())["contention"]
    assert sorted(contention) == ["4", "64"]
    from servebench.results import contention_for
    assert contention_for(contention, 64)["cpu_share"] == [0.5, 0.5] and contention_for(contention, 8) is None
    assert contention_for({"spinners": 1, "cpu_share": [0.9]}, 8) == {"spinners": 1, "cpu_share": [0.9]}  # one record


def test_a_hostload_unit_runs_no_wave_until_its_loops_are_up(tmp_path, monkeypatch):
    """An uncontended first wave would test nothing and pass: no loops within the wait, no batch, a failed unit."""
    class Late:
        def __init__(self, cpus, container=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def wait_ready(self, timeout):
            return False

        def problem(self):
            return "contention: no busy loop ran on the engine's cores"

        def record(self):
            return {"spinners": 2, "cpu_share": []}
    monkeypatch.setattr("anatomy.served.Contention", Late)
    made = len(FakeWave.made)
    err = run_served(unit(capture="hostload"), {**CMDS, "contend": "0-1"}, tmp_path, sh=Shell(), wave_cls=FakeWave,
                     ready=lambda *a, **k: True)
    assert err == "contention: no busy loop ran on the engine's cores" and len(FakeWave.made) == made


def test_a_docker_run_that_never_returns_fails_the_start(tmp_path):
    class Hung(Shell):
        def __call__(self, argv, **kw):
            if argv[:2] == ["docker", "run"]:
                assert kw.get("timeout")  # bounded: a wedged daemon must not hold the session
                self.calls.append(argv)
                raise subprocess.TimeoutExpired(argv, kw["timeout"])
            return super().__call__(argv, **kw)
    shell = Hung()
    err = run_served(unit(), CMDS, tmp_path, sh=shell, ready=lambda *a, **k: True)
    assert err.startswith("server did not start: docker run -d did not return")
    runs = [i for i, c in enumerate(shell.calls) if c[:2] == ["docker", "run"]]
    assert any(c[:3] == ["docker", "rm", "-f"] for c in shell.calls[runs[0]:])  # the daemon may have started it


def test_a_slow_removal_after_a_complete_unit_keeps_its_data(tmp_path):

    class SlowRm(Shell):
        def __call__(self, argv, **kw):
            if argv[:3] == ["docker", "rm", "-f"] and any(c[:2] == ["docker", "run"] for c in self.calls):
                raise subprocess.TimeoutExpired(argv, kw.get("timeout") or 1)
            return super().__call__(argv, **kw)
    err = run_served(unit(), CMDS, tmp_path, sh=SlowRm(), ready=lambda *a, **k: True)
    assert err is None and "did not return" in (tmp_path / "container.log").read_text()


@linux_affinity
def test_the_profiled_waves_client_runs_on_the_client_cores(tmp_path):
    import os
    seen = []

    class Where(FakeWave):
        def start(self, timeout=None):
            seen.append(os.sched_getaffinity(0))
            return self
    before = os.sched_getaffinity(0)
    core = sorted(before)[-1]
    run_served(unit(capture="graph"), CMDS, tmp_path, sh=Shell(), ready=lambda *a, **k: True, wave_cls=Where,
               client_cpuset=str(core))
    # the untimed warm-up wave and the profiled one, both pinned; restored after
    assert seen == [{core}, {core}] and os.sched_getaffinity(0) == before


def test_jit_and_triton_caches_outlive_a_container():
    """The pinned llama.cpp image carries Hopper as PTX only, and Triton keys its cache on HOME: without these under the
    mounted /cache, every container would compile its kernels again (inside a profiled capture, at that)."""
    from servebench.containers import Mounts, docker_run
    argv = docker_run("img", ["run"], name="da-x", mounts=Mounts(repo=Path("/r"), out=Path("/o"), hf=Path("/h"),
                                                                  cache=Path("/c"), models=Path("/m"), nsys=None))
    env = {argv[i + 1].split("=", 1)[0]: argv[i + 1].split("=", 1)[1] for i, a in enumerate(argv) if a == "-e"}
    assert env["CUDA_CACHE_PATH"].startswith("/cache") and env["TRITON_CACHE_DIR"].startswith("/cache")
    assert int(env["CUDA_CACHE_MAXSIZE"]) >= 2 ** 31


def test_a_call_cut_off_by_the_deadline_counts_as_the_unit_timing_out(tmp_path):
    """llama-server hanging at its largest batch: AIPerf runs out the unit's time. The status the grid writes must then
    read as a timeout, so the hung-engine skip and the give-up rule see it."""
    from anatomy import grid as G
    now = [0.0]

    class Hangs(Shell):
        def __call__(self, argv, **kw):
            if any("aiperf" in a for a in argv) and "profile" in argv:
                now[0] = 100.0  # the call used up the deadline
                raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
            return super().__call__(argv, **kw)
    err = run_served(unit(batches=(4,)), CMDS, tmp_path, sh=Hangs(), ready=lambda *a, **k: True, timeout=50,
                     clock=lambda: now[0])
    G.write_status(tmp_path, False, 50.0, err)
    assert G.timed_out(tmp_path)


def test_llama_servers_greedy_probe_sends_the_ids_and_records_the_tokens(tmp_path):
    import io
    import json

    from anatomy.served import greedy
    sent = []

    class Reply(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body)
        return Reply(json.dumps({"tokens": list(range(body["n_predict"]))}).encode())
    greedy("http://127.0.0.1:8000/completion", "Qwen/Qwen3-0.6B", {"top_k": 1, "samplers": ["top_k"]}, tmp_path,
           opener=opener)
    recs = [json.loads(line) for line in (tmp_path / "driver.jsonl").read_text().splitlines()]
    assert len(recs) == 8 and recs[0]["ids"] == list(range(32))
    assert sent[0]["return_tokens"] and "ignore_eos" not in sent[0] and sent[0]["top_k"] == 1 and isinstance(
        sent[0]["prompt"][0], int)


def test_the_greedy_probe_never_runs_past_the_units_deadline(tmp_path):
    import json

    from anatomy.served import greedy
    calls = []
    greedy("http://127.0.0.1:8000/completion", "Qwen/Qwen3-0.6B", {}, tmp_path, deadline=100.0, clock=lambda: 101.0,
           opener=lambda req, timeout=None: calls.append(timeout))
    recs = [json.loads(line) for line in (tmp_path / "driver.jsonl").read_text().splitlines()]
    assert calls == [] and len(recs) == 8 and all(r["error"] == "the unit's deadline passed" for r in recs)
