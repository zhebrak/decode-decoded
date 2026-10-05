"""Measuring the ruler: a workload whose GPU-busy fraction is known, run under the same nsys command lines the engines
get, so the parser (servebench.nsys.parse) is checked against ground truth before it measures any engine. Gate G1
requires agreement within 1 percentage point.

    nsys profile <graph-mode args> --capture-range=cudaProfilerApi --capture-range-end=repeat \
        -o /out/busy-graph-graph python3 /work/ruler/synthetic_busy.py --mode graph --variant graph --out /out
    (and --mode node, and --variant eager: ruler.run runs all four)
    python -m ruler.validate ~/decode-anatomy-runs/<session>/ruler      # host side: parse and compare

Each "step" is: the GPU work (one spin kernel of `spin_us` in the eager variant; a captured graph of `nodes` spin
kernels in the graph variant), launched asynchronously, then the host busy-waits `host_us` before launching the next
step: a decode step's shape, with a controlled host stall. The host waits are all longer than the GPU work, so the
launch queue drains every step and each window has a known busy fraction well below 1 (about spin / host_us: 0.89,
0.67, 0.33, 0.17 at the defaults); a wait shorter than the GPU work would leave the GPU always busy and test nothing.

Ground truth: the device time of one step's work, calibrated by running it back to back between two CUDA events (the
launch queue stays full, so no launch latency is counted), over the step: the span from the first step's completion to
the last one's (CUDA events recorded after each step's work), over the whole steps between them. Whole steps, as the
parser's anchors bound, so a host hiccup at a window's edge counts in neither (in validation, a 5 ms stall inside the
host's wall time and outside the whole steps read as a 4.1-point parser error); and completions rather than launches,
so where the work outlasts the host's wait and launches run ahead of the GPU the step is the GPU's (launch to launch
read 1.08 busy there). The events before each step's launch are recorded too (busy_events) but are not the truth: in
validation they read 1-6 points high, because the GPU reaches the first event and then idles until the host's launch
call (cudaGraphLaunch especially) delivers the work. Written to busy-<mode>-<variant>.json: each nsys mode gets the
truth of its own run, since node-mode tracing slows launches and changes the very fraction being checked.
"""
import argparse
import json
import time
from pathlib import Path

SPIN = None  # the Triton kernel, built on first use


def globaltimer_spin():
    """A one-thread kernel that spins until `ns` nanoseconds of the GPU's %globaltimer have passed: its duration does
    not depend on the SM clock. torch.cuda._sleep counts SM cycles instead, and without a clock lock (a shared box) the
    clock moves with the load pattern, so a cycle-based spin lasts longer in some windows than in its calibration: in
    validation that alone put one eager window 7 points off. None when Triton is not installed."""
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return None

    @triton.jit
    def _spin(ptr, ns):
        x = tl.load(ptr)
        t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                       pack=1)
        now = t0
        while now - t0 < ns:
            now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [x], dtype=tl.int64, is_pure=False,
                                            pack=1)
        tl.store(ptr, x + 1)

    return _spin


def spin(torch, us, clock_hz, buf=None):
    global SPIN
    if SPIN is None:
        SPIN = globaltimer_spin() or False
    if SPIN:
        SPIN[(1,)](buf, int(us * 1000))
    else:
        torch.cuda._sleep(int(us * 1e-6 * clock_hz))


def busy_wait(us):
    end = time.perf_counter_ns() + int(us * 1e3)
    while time.perf_counter_ns() < end:
        pass


def main(argv=None):
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=("eager", "graph"), required=True)
    p.add_argument("--mode", choices=("graph", "node", "none"), required=True,
                   help="the nsys --cuda-graph-trace this run is under; none: no profiler, the timing run the cells' "
                        "occupancy divides by (ruler.validate's cross check)")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--spin-us", type=float, default=400.0)
    p.add_argument("--nodes", type=int, default=50)
    p.add_argument("--host-us", default="380,420,450,600,1200,2400",
                   help="host wait per step, one capture window each; below --spin-us the queue fills and the GPU runs "
                        "back to back (truth near 1, where decode cells sit)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    clock_hz = torch.cuda.get_device_properties(0).clock_rate * 1e3  # kHz -> Hz: the _sleep fallback counts cycles
    buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    node_us = a.spin_us / a.nodes

    if a.variant == "graph":
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                spin(torch, node_us, clock_hz, buf)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(a.nodes):
                spin(torch, node_us, clock_hz, buf)
        work = graph.replay
    else:
        def work():
            spin(torch, a.spin_us, clock_hz, buf)

    def device_time_per_step(n=100):
        """Back-to-back: the queue never drains, so the elapsed time is the work's own device time."""
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            work()
        e1.record()
        e1.synchronize()
        return e0.elapsed_time(e1) * 1e6 / n

    windows = []
    for host_us in map(float, a.host_us.split(",")):
        for _ in range(20):  # warm-up, outside the capture window
            work()
            busy_wait(host_us)
        torch.cuda.synchronize()
        step_device_ns = device_time_per_step()
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.steps)]
        torch.cuda.cudart().cudaProfilerStart()
        u0 = time.time()
        t0 = time.perf_counter_ns()
        for e0, e1 in events:
            e0.record()
            work()
            e1.record()
            busy_wait(host_us)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        u1 = time.time()
        torch.cuda.cudart().cudaProfilerStop()
        span_ns = events[0][1].elapsed_time(events[-1][1]) * 1e6  # first completion to last: whole steps
        step_ns = span_ns / (a.steps - 1)
        events_ns = sum(e0.elapsed_time(e1) for e0, e1 in events[1:]) * 1e6
        windows.append({"host_us": host_us, "steps": a.steps, "wall_ns": t1 - t0, "gpu_ns": step_device_ns * a.steps,
                        "step_device_ns": step_device_ns, "busy": step_device_ns / step_ns,
                        "busy_events": events_ns / span_ns, "step_ns": step_ns,
                        "unix": [u0, u1]})  # for the contention check against the host's GPU process log
    a.out.mkdir(parents=True, exist_ok=True)
    rec = {"mode": a.mode, "variant": a.variant, "spin_us": a.spin_us, "nodes": a.nodes if a.variant == "graph" else 1,
           "clock_hz": clock_hz, "spin": "globaltimer" if SPIN else "cycles", "windows": windows}
    (a.out / f"busy-{a.mode}-{a.variant}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
