"""anatomy.grid end to end with the real run_unit and execute, nothing of the grid replaced: a tiny grid whose engine
processes are fakes (natively, a python3 in each engine's environment; under Docker, a docker command on PATH) that
finish, fail, or hang ignoring SIGTERM. What a paid session relies on when an engine misbehaves: the statuses, the
timeout, the hung engine's other units skipped, its processes and containers ended, the grid unwinding on a signal."""
import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from anatomy import grid as G
from servebench import containers

GRID = """versions: 2026-09
gpu: NVIDIA H100 80GB HBM3
memory_fraction: 0.9
kv_reserve_gib: 4
max_batch: 256
validation: true
context: {prompt_len: 896, gen_len: 288}
blocks:
  e2e:
    engines: [vllm, trtllm]
    arms: [defaults, non-default]
    levels: [headline]
    models: [Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B]
    batches: [1]
    captures: {timing: all}
"""
FAKE_ENGINE = """#!/usr/bin/env bash
here=$(dirname "$0"); echo $$ >> "$here/pids"
out=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out=$a; prev=$a; done
case "$*" in *Qwen3-0.6B*) mode=$(cat "$here/mode-small") ;; *) mode=$(cat "$here/mode-large") ;; esac
case $mode in
  hang) trap '' TERM; sleep 600 ;;
  fail) exit 1 ;;
  ok) printf '{"kind": "long", "batch": 1, "repeat": 0, "cadence": {"step_ns": 10000000}}\\n' > "$out" ;;
esac
"""
FAKE_DOCKER = """#!/usr/bin/env bash
echo "$*" >> {log}
case "$1" in run) trap '' TERM; sleep 600 ;; *) exit 0 ;; esac
"""


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A raw root, a grid file and engine environments under tmp_path, the box's own probes stubbed."""
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path / "runs")
    monkeypatch.setattr(G.env, "host", lambda: {})
    monkeypatch.setattr(containers, "ENVS", tmp_path / "envs")
    monkeypatch.setattr(containers, "RUNTIME", "native")
    (tmp_path / "grid.yaml").write_text(GRID)
    return tmp_path


def engine(box, name, small, large):
    bin_ = box / "envs" / name / "bin"
    bin_.mkdir(parents=True)
    (bin_ / "python3").write_text(FAKE_ENGINE)
    (bin_ / "python3").chmod(0o755)
    (bin_ / "mode-small").write_text(small)
    (bin_ / "mode-large").write_text(large)
    return bin_


def statuses(box, session):
    return {str(p.parent.relative_to(box / "runs" / session)): json.loads(p.read_text())
            for p in (box / "runs" / session).rglob("status.json") if ".attempt-" not in str(p)}


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return Path(f"/proc/{pid}/stat").exists() and Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"


def test_native_units_finish_fail_and_hang_and_a_hung_engine_is_skipped_on_its_model(box):
    vllm = engine(box, "vllm-0.30.0", small="hang", large="ok")
    engine(box, "trtllm-1.2.1", small="ok", large="fail")
    t0 = time.monotonic()
    G.main([str(box / "grid.yaml"), "--run", "--no-gpu", "--session-id", "E", "--timeout", "1"])
    assert time.monotonic() - t0 < 15
    st = statuses(box, "E")
    small_vllm = {k: s for k, s in st.items() if k.startswith("vllm") and "Qwen3-0.6B" in k}
    assert sorted(s["error"].split(":")[0] for s in small_vllm.values()) == ["exit 124", "exit 124", "skipped",
                                                                            "skipped"]
    assert all(G.timed_out(box / "runs" / "E" / k) for k, s in small_vllm.items() if s["error"] == "exit 124")
    assert all(s["ok"] for k, s in st.items() if k.startswith("vllm") and "Qwen3-1.7B" in k)   # its next model runs
    assert all(s["ok"] for k, s in st.items() if k.startswith("trtllm") and "Qwen3-0.6B" in k)
    assert all(s["error"] == "exit 1" for k, s in st.items() if k.startswith("trtllm") and "Qwen3-1.7B" in k)
    pids = [int(p) for p in (vllm / "pids").read_text().split()]
    time.sleep(0.2)
    assert not [p for p in pids if alive(p)]                            # nothing the hung engine started is left


def docker_box(box, monkeypatch):
    log = box / "docker.log"
    (box / "bin").mkdir()
    (box / "bin" / "docker").write_text(FAKE_DOCKER.format(log=log))
    (box / "bin" / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{box / 'bin'}:{os.environ['PATH']}")
    monkeypatch.setattr(containers, "DOCKER", ["docker"])
    monkeypatch.setattr(containers, "RUNTIME", "docker")
    return log


def test_a_container_that_runs_out_of_time_is_removed(box, monkeypatch):
    log = docker_box(box, monkeypatch)
    G.main([str(box / "grid.yaml"), "--run", "--no-gpu", "--session-id", "D",
            "--models", "Qwen/Qwen3-0.6B", "--engines", "trtllm", "--timeout", "1"])
    calls = log.read_text().splitlines()
    runs = [c.split("--name ")[1].split()[0] for c in calls if c.startswith("run ")]
    assert len(runs) == 2 and all(f"rm -f {name}" in calls for name in runs)  # every timed-out container removed
    errors = sorted(s["error"].split(":")[0] for s in statuses(box, "D").values())
    assert errors == ["exit 124", "exit 124", "skipped"]               # two timeouts, then its third unit skipped


def test_sigterm_mid_unit_removes_the_container_and_unwinds_the_grid(box, monkeypatch):
    log = docker_box(box, monkeypatch)
    before = signal.getsignal(signal.SIGTERM)
    threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    with pytest.raises(SystemExit) as e:
        G.main([str(box / "grid.yaml"), "--run", "--no-gpu", "--session-id", "D",
                "--models", "Qwen/Qwen3-0.6B", "--engines", "vllm", "--timeout", "30"])
    assert e.value.code == 128 + signal.SIGTERM and signal.getsignal(signal.SIGTERM) == before
    calls = log.read_text().splitlines()
    name = next(c.split("--name ")[1].split()[0] for c in calls if c.startswith("run "))
    assert f"rm -f {name}" in calls                                     # the engine went with the grid
