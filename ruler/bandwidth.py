"""Achievable HBM bandwidth on this card: the denominator of MBU and SOL beside the datasheet's 3.35 TB/s (streaming
reads reach about 91-94% of datasheet on H100 SXM in published measurements). Runs on the GPU box inside any engine
image with torch (S1):

    python3 /work/ruler/bandwidth.py --out /out      (ruler.run runs it in the vLLM image, or vLLM's venv)

Three kernels, each timed with CUDA events over `iters` launches after warm-up, median reported:
- read:  x.sum() over a 4 GiB bf16 tensor (a streaming read, negligible arithmetic): read_bytes_per_s, the MBU
         denominator, since a decode step is read-dominated;
- copy:  y.copy_(x): read + write;
- gemv:  a [1, K] x [K, N] bf16 matmul over a weight matrix the size of a 4B model's layer stack slice, the shape a
         batch-1 decode step's weight matmuls have: what a vendor GEMV reaches, for the kernel-excess discussion.

With --profile (ruler.run, under the metrics capture) only the read runs, bracketed by
cudaProfilerStart/Stop, and bandwidth-profiled.json records its bytes and wall-clock span: ruler.validate calibrates
the DRAM throughput counters against it (a stream of known bytes).
"""
import argparse
import json
import statistics
import time
from pathlib import Path


def timed(fn, iters, torch):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    return statistics.median(times), min(times)


def main(argv=None):
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gib", type=float, default=4.0)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", action="store_true", help="the read only, inside a cudaProfilerApi capture range")
    a = p.parse_args(argv)
    n = int(a.gib * 2**30 // 2)
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda").uniform_()
    y = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()
    if a.profile:
        torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.cudart().cudaProfilerStart()
        read_med, _ = timed(lambda: x.sum(), a.iters, torch)
        torch.cuda.cudart().cudaProfilerStop()
        # calls: every x.sum() inside the capture, timed()'s 3 warm-ups included; a 4 GiB reduction runs as more
        # than one kernel (32-bit indexing splits it), so the calibration divides bytes by all of their time
        out = {"device": torch.cuda.get_device_properties(0).name, "bytes": nbytes, "calls": a.iters + 3,
               "read_bytes_per_s": nbytes / read_med, "unix": [t0, time.time()]}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "bandwidth-profiled.json").write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out))
        return
    read_med, read_best = timed(lambda: x.sum(), a.iters, torch)
    copy_med, copy_best = timed(lambda: y.copy_(x), a.iters, torch)
    y = None  # free the copy target before the GEMV weights are allocated
    k, m = 8192, 32768  # 512 MiB of bf16 weights
    w = torch.empty(k, m, dtype=torch.bfloat16, device="cuda").uniform_()
    v = torch.empty(1, k, dtype=torch.bfloat16, device="cuda").uniform_()
    gemv_med, _ = timed(lambda: v @ w, a.iters, torch)
    props = torch.cuda.get_device_properties(0)
    out = {"device": props.name, "bytes": nbytes,
           "read_bytes_per_s": nbytes / read_med, "read_best_bytes_per_s": nbytes / read_best,
           "copy_bytes_per_s": 2 * nbytes / copy_med, "copy_best_bytes_per_s": 2 * nbytes / copy_best,
           "gemv_bytes_per_s": w.numel() * 2 / gemv_med, "torch": torch.__version__, "cuda": torch.version.cuda}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "bandwidth.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: (f"{v / 1e12:.3f} TB/s" if k.endswith("per_s") else v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
