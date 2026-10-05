"""Every command the grid can run, rendered against a golden file per release, so a flag never changes unnoticed.
Regenerate after an intended change: UPDATE_GOLDEN=1 pytest tests/test_engines.py, then review the diff."""
import os
from pathlib import Path

import pytest
import yaml

from servebench.engines.adapters import engine_for
from servebench.engines.base import Plan, Release, merge, render

GOLDEN = Path(__file__).parent / "golden"
RELEASES = [("vllm", "0.30.0"), ("sglang", "0.5.20"), ("trtllm", "1.2.1"), ("llamacpp", "v0.5.0")]
MODEL = {"llamacpp": "/models/gguf/Qwen3-8B-bf16.gguf"}


def rendered(engine, version):
    rel = Release.load(engine, version)
    eng = engine_for(rel)
    model = MODEL.get(engine, "Qwen/Qwen3-8B")
    lines = [f"# {engine} {version}  image {rel.image}"]
    for name in rel.arm_names:
        for profile in (False, True):
            arm = rel.arm(name, profile=profile)
            plan = Plan(model=model, batches=(1, 8, 64, 256), profile="cuda" if profile else "none")
            tag = f"{name}{' +profile' if profile else ''}"
            env = " ".join(f"{k}={v}" for k, v in sorted(arm.env.items()))
            if "L0" in arm.levels and not profile:  # the grid refuses a profiled L0 runner
                lines.append(f"[{tag} L0] {env} {render(eng.runner_cmd(arm, plan))}".replace("]  ", "] "))
            if "L1" in arm.levels:
                lines.append(f"[{tag} L1] {env} {render(eng.offline_cmd(arm, plan))}".replace("]  ", "] "))
            if "L2" in arm.levels:
                lines.append(f"[{tag} L2] {env} {render(eng.server_cmd(arm, plan, 8000))}".replace("]  ", "] "))
                if hasattr(eng, "server_files"):
                    for fname, body in eng.server_files(arm).items():
                        lines.append(f"    {fname}: {yaml.safe_load(body)}")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("engine,version", RELEASES)
def test_golden(engine, version):
    path = GOLDEN / f"{engine}-{version}.txt"
    text = rendered(engine, version)
    if os.environ.get("UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    assert text == path.read_text()


def test_merge_deep_and_replacing():
    assert merge({"a": {"x": 1, "y": 2}, "b": [1]}, {"a": {"y": 3}, "b": [2]}) == {"a": {"x": 1, "y": 3}, "b": [2]}
    assert merge({"a": {"x": 1}}, {"a": None}) == {"a": None}


def test_vllm_conventions():
    rel = Release.load("vllm", "0.30.0")
    eng = engine_for(rel)
    cmd = render(eng.server_cmd(rel.arm("graphs-off"), Plan(model="m", batches=(1,)), 1))
    assert "--no-enable-prefix-caching" in cmd and "--compilation-config '{\"cudagraph_mode\":\"NONE\"}'" in cmd
    assert "--no-async-scheduling" in render(eng.server_cmd(rel.arm("async-off"), Plan(model="m", batches=(1,)), 1))
    assert rel.arm("mrv2-off").env == {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    assert rel.arm("defaults", profile=True).env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_sglang_conventions():
    rel = Release.load("sglang", "0.5.20")
    cmd = render(engine_for(rel).server_cmd(rel.arm("defaults"), Plan(model="m", batches=(1,)), 1))
    assert "--disable-radix-cache" in cmd and "--cuda-graph-max-bs-decode 256" in cmd
    assert "--cuda-graph-max-bs " not in cmd  # removed in 0.5.20
    assert rel.arm("defaults", profile=True).nsys == ("--trace-fork-before-exec=true",)


def test_trtllm_graphs_off_passes_null():
    rel = Release.load("trtllm", "1.2.1")
    assert rel.arm("graphs-off").args["cuda_graph_config"] is None
    files = engine_for(rel).server_files(rel.arm("overlap-off"))
    assert "disable_overlap_scheduler: true" in files["llm_api_options.yaml"]


def test_llamacpp_levels_and_gpu_sampling():
    rel = Release.load("llamacpp", "v0.5.0")
    assert rel.arm("gpu-sampling").levels == ("L2",)
    assert rel.arm("graphs-off").env == {"GGML_CUDA_DISABLE_GRAPHS": "1"}
    cmd = render(engine_for(rel).server_cmd(rel.arm("defaults"), Plan(model="x.gguf", batches=(1, 64)), 1))
    assert "-np 64" in cmd and f"-c {64 * (896 + 288 + 64)}" in cmd and "-fa on" in cmd
