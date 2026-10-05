"""The host fingerprint's GPU half (S1, and the first minute of every session): what one kernel launch costs on this
host and card, eager and inside a CUDA graph, plus a single-thread Python microbenchmark. Hazy Research measured about
2.1 us per launch on a stream and about 1.3 us inside graphs; these numbers say what this rental's CPU and
driver give, so a second host shows up as a different fingerprint rather than as noise.

    python3 /work/ruler/launch_latency.py --out /out      (ruler.run runs it in the vLLM image, or vLLM's venv)

- eager_host_us: host time per launch of a 1-element add, launches queued back to back: a torch op's whole cost on
  the CPU (dispatch included), 3.4-6 us on the validation hosts' CPUs, not the bare cudaLaunchKernel;
- eager_gpu_us:  GPU time per kernel for the same stream of launches (each launch's device-side spacing);
- graph_node_us: GPU time per node replaying a captured graph of `nodes` such kernels (in-graph per-node cost);
- python_loop_ms: a fixed pure-Python workload (dict and list churn), single thread: the interpreter's speed here.

A split-GEMV slope (the same bytes in more, smaller GEMVs) is not a launch cost: 7.1 us a node in validation (H100
NVL) is small kernels failing to fill the memory system, which is kernel excess.
"""
import argparse
import json
import time
from pathlib import Path


def python_loop(n=2_000_000):
    t0 = time.perf_counter()
    d, acc = {}, 0
    for i in range(n):
        d[i & 1023] = i
        acc += d.get((i * 7) & 1023, 0) & 3
    return (time.perf_counter() - t0) * 1e3, acc


def main(argv=None):
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--launches", type=int, default=20_000)
    p.add_argument("--nodes", type=int, default=1000)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--profile", action="store_true",
                   help="only the graph's replays, inside a cudaProfilerApi capture: ruler.run traces them node by "
                        "node to split an empty node's cost between its kernel's duration and the gap before the next")
    a = p.parse_args(argv)
    x = torch.zeros(1, device="cuda")
    for _ in range(1000):
        x.add_(1)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(a.launches):
        x.add_(1)
    end.record()
    host_s = time.perf_counter() - t0
    end.synchronize()
    gpu_s = start.elapsed_time(end) / 1e3

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(a.nodes):
            x.add_(1)
    g.replay()
    torch.cuda.synchronize()
    if a.profile:
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    reps, times = 20, []
    for _ in range(reps):
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    times.sort()
    loop_ms, _ = python_loop()
    out = {"device": torch.cuda.get_device_name(0), "eager_host_us": host_s / a.launches * 1e6,
           "eager_gpu_us": gpu_s / a.launches * 1e6, "graph_node_us": times[len(times) // 2] / a.nodes * 1e6,
           "python_loop_ms": loop_ms, "launches": a.launches, "nodes": a.nodes, "torch": torch.__version__}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "launch_latency.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
