"""ruler.run builds its commands the way the grid builds an engine's (servebench.containers, nsys_cmd)."""
import json

import pytest

from ruler import run as R
from servebench.nsys.capture import nsys_cmd


@pytest.mark.parametrize("runtime", ["docker", "native"])
def test_the_ruler_runs_the_engines_commands(monkeypatch, capsys, runtime, tmp_path):
    monkeypatch.setattr("servebench.containers.RUNTIME", runtime)
    monkeypatch.setattr("servebench.containers.ENVS", tmp_path / "envs")
    monkeypatch.setattr("servebench.env.gpu_query", lambda *f, device="0": {"name": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setenv("SESSION", "S1-test")
    monkeypatch.setenv("VERSIONS", "2026-09")
    monkeypatch.setenv("DA_NSYS", "/opt/nsys-host")
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path / "runs")
    assert R.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    labels = [line[2:] for line in out.splitlines() if line.startswith("# ")]
    assert labels[:4] == ["launch latency", "bandwidth", "empty nodes traced", "DRAM counters"] and len(labels) == 12
    busy = next(line for line in out.splitlines() if "busy-graph-eager" in line)
    expected = nsys_cmd("/out/busy-graph-eager", "graph")[1:]  # the grid's switches, after the nsys binary
    if runtime == "native":
        assert busy.startswith("env -i") and f"{tmp_path}/envs/vllm-0.30.0/bin" in busy
        assert "/opt/nsys-host/bin/nsys " + " ".join(e.replace("/out", f"{tmp_path}/runs/S1-test/ruler")
                                                     for e in expected) in busy
    else:
        assert "docker run" in busy and " ".join(expected) in busy and "--gpus device=0" in busy
    dram = next(line for line in out.splitlines() if "dram-calibration" in line)
    assert "--gpu-metrics-set=gh100" in dram and "--capture-range-end=repeat" in dram  # the metrics captures' switches


def test_the_ruler_needs_its_session_and_nsys(monkeypatch):
    monkeypatch.delenv("SESSION", raising=False)
    monkeypatch.setenv("DA_NSYS", "/opt/nsys-host")
    with pytest.raises(SystemExit, match="SESSION"):
        R.main(["--dry-run"])


def test_a_ruler_step_that_hangs_is_ended_and_its_container_removed():
    import subprocess
    removed = []

    def hung(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    code = R.step(["docker", "run", "--name", R.NAME], sh=hung, remove=lambda *a, **k: removed.append((a, k)))
    assert code == 124 and removed and removed[0][0] == (R.NAME,)


@pytest.mark.parametrize("grid", ["configs/grid/v1.yaml", "configs/grid/refresh.yaml"])
def test_the_ruler_runs_at_the_cells_locked_clock(monkeypatch, tmp_path, grid):
    """The launch costs the ruler measures price every cell's launch term: they are measured at the cells' clock,
    read as the grid reads it (anatomy.grid.load), so a refresh grid with no clock of its own runs at its base's."""
    events = []
    monkeypatch.setenv("SESSION", "S1-test")
    monkeypatch.setenv("GRID", grid)
    monkeypatch.delenv("VERSIONS", raising=False)  # the version set is the grid's too
    monkeypatch.setenv("DA_NSYS", "/opt/nsys-host")
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr("servebench.env.gpu_query", lambda *f, device="0": {"name": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setattr(R.gpu, "lock_clocks", lambda mhz: events.append(("lock", mhz)) or True)
    monkeypatch.setattr(R.gpu, "unlock_clocks", lambda: events.append(("unlock",)))
    monkeypatch.setattr(R, "measure", lambda steps, out: events.append(("measure", len(steps))) or 0)
    monkeypatch.setattr("servebench.env.host", lambda: {"id": "abc", "gpu": {"name": "NVIDIA H100 80GB HBM3"}})
    assert R.main([]) == 0
    assert events == [("lock", 1830), ("measure", 12), ("unlock",)]
    assert (tmp_path / "S1-test" / "ruler" / "clock.json").read_text().strip() == '{"sm_clock_lock_mhz": 1830}'
    assert json.loads((tmp_path / "S1-test" / "ruler" / "host.json").read_text())["id"] == "abc"  # the card, beside


def test_the_rulers_caches_are_the_boxs_not_the_sessions(monkeypatch, tmp_path):
    """Triton's and Inductor's cache files once landed in the session's ruler/ and their *.json went up with it: the
    ruler mounts the box's caches and HF home (anatomy.grid.default_mounts), never a directory under the session."""
    seen = {}
    monkeypatch.setenv("SESSION", "S1-test")
    monkeypatch.setenv("DA_NSYS", "/opt/nsys-host")
    monkeypatch.delenv("VERSIONS", raising=False)
    monkeypatch.setattr("servebench.results.RAW_ROOT", tmp_path)
    monkeypatch.setattr("servebench.env.gpu_query", lambda *f, device="0": {"name": "NVIDIA H100 80GB HBM3"})
    monkeypatch.setattr(R.gpu, "lock_clocks", lambda mhz: True)
    monkeypatch.setattr(R.gpu, "unlock_clocks", lambda: None)
    monkeypatch.setattr(R, "measure", lambda steps, out: 0)
    monkeypatch.setattr("servebench.env.host", lambda: {"id": "abc", "gpu": {"name": "NVIDIA H100 80GB HBM3"}})

    def runner_for(release, mounts, runtime=None):
        seen["mounts"] = mounts
        return (lambda cmd, **k: list(cmd)), {}
    monkeypatch.setattr(R.containers, "runner_for", runner_for)
    assert R.main([]) == 0
    m = seen["mounts"]
    session = tmp_path / "S1-test"
    assert m.out == session / "ruler" and str(m.nsys) == "/opt/nsys-host"
    assert not str(m.cache).startswith(str(session)) and not str(m.hf).startswith(str(session))
