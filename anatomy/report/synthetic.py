"""Synthetic cells for the page's layout preview: the page is designed and tested before the measurements exist.
NOT A PREDICTION and never published as data: a toy host/device model that gives every chart realistic shapes, so
layout, scales, labels and interactions are settled before S2. The page shows a banner whenever it renders these.

Toy model, per engine: host time per step H(B) = h0 + h1 B; kernel time = floor / efficiency(model size) plus each
kernel's launch cost (kernels per step x per-launch cost, inside the kernel as CUPTI records it); an overlapping
engine's step is the larger of host and GPU time, a serial one's their sum.
"""
from anatomy.decompose import decompose
from anatomy.metrics import SERIAL_ARMS, physics
from servebench.engines.base import version_set
from servebench.physics import load_shape

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
VERSIONS = version_set("2026-09")[1]  # the release the layout preview is drawn for
ENGINES = {  # level, (h0 ms, h1 ms per sequence), overlapped, per-launch us
    "vllm": ("L1", (1.45, 0.0060), True, 1.3),
    "sglang": ("L1", (1.20, 0.0050), True, 1.3),
    "trtllm": ("L1", (1.70, 0.0045), True, 1.3),
    "llamacpp": ("L2", (0.55, 0.0200), False, 1.6),
}
ARMS = {"vllm": {"graphs-off": (1.0, 2.6), "async-off": (1.0, None), "mrv2-off": (1.25, None)},
        "sglang": {"graphs-off": (1.0, 2.6), "overlap-off": (1.0, None)},
        "trtllm": {"graphs-off": (1.0, 2.6), "overlap-off": (1.0, None)},
        "llamacpp": {"graphs-off": (1.0, 2.8), "gpu-sampling": (0.7, None)}}
BW = 3.08e12


def cell(engine, model, batch, prompt=896, gen=288, host_scale=1.0, launch_us=None, overlapped=None, arm="defaults"):
    level, (h0, h1), ov, lu = ENGINES[engine]
    version = VERSIONS[engine]
    ov = ov if overlapped is None else overlapped
    lu = launch_us or lu
    shape = load_shape(model)
    ph = physics(model, batch, prompt, gen, None, BW)
    eff = 0.82 - 0.38 / (1 + shape.params / 2.5e9)
    kern = ph["floor_ms"] / eff
    n_kernels = 9 * shape.layers + 12
    kern += n_kernels * lu / 1e3
    host = (h0 + h1 * batch) * host_scale
    gpu = kern
    step = max(host, gpu) * 1.02 if ov else host + gpu
    occupied = gpu / step
    ph = physics(model, batch, prompt, gen, step, BW)
    return {"engine": engine, "version": version, "arm": arm, "level": level, "model": model, "prompt_len": prompt,
            "gen_len": gen, "batch": batch, "step_ms": step, "occupied": occupied, "busy_ms": gpu, "kernel_ms": kern,
            "params": shape.params, "kernels_per_step": n_kernels, **ph,
            **decompose(step, occupied, kern, ph["floor_ms"], kernels_per_step=n_kernels, node_floor_us=lu),
            "synthetic": True}


def cells():
    rows = []
    for engine in ENGINES:
        for model in MODELS:
            limit = load_shape(model).max_batch(1184, 0.9 * 85.5e9)
            for b in [b for b in BATCHES if b <= limit]:
                rows.append(cell(engine, model, b))
                if model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B") and b in (1, 8, 64, 256):
                    for arm, (scale, launch) in ARMS[engine].items():
                        serial = arm in SERIAL_ARMS
                        rows.append(cell(engine, model, b, host_scale=scale, launch_us=launch,
                                         overlapped=False if serial else None, arm=arm))
        for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"):
            for b in (1, 16, 64):
                base = cell(engine, model, b)
                # the levels around the headline one (that row is above): runner below, server above
                for level, extra in (("L0", -0.25), ("L2", 0.08 if model.endswith("0.6B") else 0.02)):
                    if (engine, level) in {("vllm", "L0"), ("trtllm", "L0")} or level == base["level"]:
                        continue
                    rows.append(dict(base, level=level, arm="defaults", step_ms=base["step_ms"] * (1 + extra)))
    return rows


def trace_strip(model="Qwen/Qwen3-0.6B", batch=1, steps=3):
    """GPU activity over `steps` decode steps per engine, as rows of kind "kernel" (start, end in us: one block per
    layer, small gaps between them) and kind "step" (where each step starts). The real strip is cut from a
    node-mode trace of the same cell; this is the layout data."""
    rows = []
    for engine in ENGINES:
        c = cell(engine, model, batch)
        step_us, kern_us, bub_us = c["step_ms"] * 1e3, c["kernel_ms"] * 1e3, 0.02 * c["kernel_ms"] * 1e3
        layers = load_shape(model).layers
        for s in range(steps):
            t = s * step_us
            rows.append({"engine": engine, "kind": "step", "start": t, "step": s})
            for k in range(layers):
                a = t + k * (kern_us + bub_us) / layers
                rows.append({"engine": engine, "kind": "kernel", "start": a, "end": a + kern_us / layers, "step": s})
    return rows
