"""The whole command every unit of the smoke run, a DRAM unit and a served unit runs, under Docker and natively,
against golden files: the docker run template (seccomp, IPC, shared memory, mounts, the offline cache, the pinned
image, the counters' capability) and the native translation of it, not only the engine's own argv
(tests/test_engines.py). A flag dropped from either fails here, not on the rented box.
Regenerate after an intended change: UPDATE_GOLDEN=1 pytest tests/test_run_templates.py, then review the diff."""
import os
from pathlib import Path

import pytest

from anatomy import grid as G
from servebench import containers

GOLDEN = Path(__file__).parent / "golden"
V1 = str(G.ROOT / "configs" / "grid" / "validation" / "four-engines.yaml")  # every engine's template, llama.cpp's too
SELECTIONS = (["--session", "S1", "--block", "bringup", "--block", "scale", "--models", "Qwen/Qwen3-0.6B"],
              ["--block", "dram", "--models", "Qwen/Qwen3-0.6B", "--engines", "vllm"],
              ["--block", "served", "--models", "Qwen/Qwen3-0.6B", "--engines", "vllm"])


def rendered(monkeypatch, capsys, runtime):
    """Every selection's dry run, on a box pinned to the study VM's shape: 26 CPUs, no quota, fixed home and caches."""
    for k in (*containers.PASS_THROUGH, "DA_DOCKER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", "/home/box")
    monkeypatch.setenv("HF_HOME", "/home/box/.cache/huggingface")
    monkeypatch.setenv("DA_NSYS", "/opt/nsys-host")
    monkeypatch.setattr(containers, "ENVS", Path("/home/box/da-envs"))
    monkeypatch.setattr(containers, "RUNTIME", runtime)
    monkeypatch.setattr(G.env, "gpu_local_cpus", lambda *a, **k: list(range(26)))
    monkeypatch.setattr(G.cgroup, "cpu_quota", lambda *a, **k: None)
    monkeypatch.setattr(G.env, "model_revision", lambda model: "0123abcd")
    out = []
    for sel in SELECTIONS:
        assert G.main([V1, *sel, "--dry-run"]) == 0
        out.append(capsys.readouterr().out)
    return "".join(out).replace(str(G.ROOT), "/REPO")


@pytest.mark.parametrize("runtime", ["docker", "native"])
def test_run_templates_match_their_golden_files(monkeypatch, capsys, runtime):
    text = rendered(monkeypatch, capsys, runtime)
    path = GOLDEN / f"runs-{runtime}.txt"
    if os.environ.get("UPDATE_GOLDEN"):
        path.write_text(text)
    assert text == path.read_text()


def test_the_docker_template_keeps_what_the_engines_and_nsys_need(monkeypatch, capsys):
    """Named, so a failure says which: nsys inside a container needs seccomp unconfined; vLLM and SGLang need host IPC
    and a large /dev/shm; every unit reads its weights offline from the cache, the code read-only, a pinned image; the
    DRAM counters need SYS_ADMIN."""
    runs = [line for line in rendered(monkeypatch, capsys, "docker").splitlines() if line.startswith("run: docker")]
    assert runs
    for flag in ("--security-opt seccomp=unconfined", "--ipc=host", "--shm-size=16g", "--network=host",
                 "-e HF_HUB_OFFLINE=1", "/REPO:/work:ro", "--gpus device=0", "--cpuset-cpus 0-21", "@sha256:"):
        assert all(flag in r for r in runs), flag
    assert any("--cap-add SYS_ADMIN" in r for r in runs if "/metrics" in r)
