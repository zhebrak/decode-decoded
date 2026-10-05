"""The nsys command lines, one per capture kind. One pinned Nsight Systems install on the host is mounted read-only
into every engine container at /opt/nsys, so every engine is traced by the same nsys version.

Kinds:
- graph:   occupancy. --cuda-graph-trace=graph (one row per graph launch, the documented low-overhead mode), CUDA and
           NVTX only: OS-runtime tracing and CPU sampling add host work to exactly the steps whose host stall is being
           measured, so they are left out.
- node:    kernel durations inside graphs (--cuda-graph-trace=node, "significant runtime overhead": its gaps are never
           used, only kernel times).
- metrics: graph mode plus GPU-metrics sampling (DRAM read/write throughput %) for the MBU cross-check; the metric
           set is the card's (metric_set: gh100 on H100/H200, ga100 on A100).
           Needs perf-counter access (root or CAP_PERFMON; the container gets SYS_ADMIN).
- hostload: the graph trace again, run while spinner processes contend for the engine's cores (anatomy.grid: one per
           core, servebench.containers.Contention): far more host cost than tracing adds, the same GPU work. Occupancy
           is the graph trace's busy time per step over the unprofiled step, which assumes a slower host leaves the
           GPU's busy time alone; this capture tests that, cell by cell (anatomy.tidy: busy_invariant). The switches
           are graph's: the extra host cost comes from the contention, not from the trace.

Windows: --capture-range=cudaProfilerApi with --capture-range-end=repeat, so every cudaProfilerStart/Stop pair the
driver opens (one per batch width) becomes its own report; the default, stop-shutdown, would end the session at the
first stop. Servers have no profiler API: `nsys launch`, then start and stop around each wave.
"""
NSYS = "/opt/nsys/bin/nsys"  # servebench.containers.NSYS_ROOT, with its bin: the one place nsys is called from
KINDS = ("graph", "node", "metrics", "hostload")
GRAPH_MODE = ("graph", "metrics", "hostload")  # --cuda-graph-trace=graph; node traces every node
METRIC_SETS = (("H100", "gh100"), ("H200", "gh100"), ("A100", "ga100"))  # the cards the harness has run on


def metric_set(gpu_name):
    """The nsys --gpu-metrics-set for a card, from its nvidia-smi name; None for a card with no set listed here."""
    return next((s for key, s in METRIC_SETS if key in (gpu_name or "")), None)


def traced(kind):
    """What is traced: the switches `nsys launch` takes, and `nsys profile` with them."""
    if kind not in KINDS:
        raise ValueError(f"capture kind {kind!r} is not one of {KINDS}")
    return ["-t", "cuda,nvtx", f"--cuda-graph-trace={'graph' if kind in GRAPH_MODE else 'node'}"]


def collected(kind, metrics_set="gh100", metrics_hz=10_000):
    """How it is collected and written: the switches `nsys start` takes, and `nsys profile` with them."""
    out = ["--sample=none", "--cpuctxsw=none", "--export=sqlite", "--force-overwrite=true"]
    if kind == "metrics":
        out += ["--gpu-metrics-devices=0", f"--gpu-metrics-set={metrics_set}", f"--gpu-metrics-frequency={metrics_hz}"]
    return out


def nsys_cmd(out_base, kind="graph", extra=(), metrics_set="gh100"):
    """L1: the driver under `nsys profile`, one export per cudaProfilerStart/Stop range."""
    return [NSYS, "profile", *traced(kind), *collected(kind, metrics_set), "-o", out_base,
            "--capture-range=cudaProfilerApi", "--capture-range-end=repeat", *extra]


# L2: the server runs under `nsys launch`, and the runner opens and closes collection from outside with `nsys start` /
# `nsys stop`, around each wave (anatomy.served). The same mechanism for all four servers, none of which needs a
# profiler endpoint. Which switches belong where, from nsys 2026.5.1's own --help (launch refused --cpuctxsw): launch
# takes what is traced, start how it is collected and written; the same two helpers build nsys_cmd, so L1 and L2 are
# traced alike.
SESSION = "da"


def nsys_launch_cmd(kind="graph", extra=()):
    return [NSYS, "launch", f"--session-new={SESSION}", *traced(kind), *extra]


def nsys_start_cmd(out_base, kind="graph", metrics_set="gh100"):
    return [NSYS, "start", f"--session={SESSION}", "-o", out_base, *collected(kind, metrics_set)]


def nsys_shutdown_cmd():
    """Ends the launch session and kills its application: a server started by `nsys launch` belongs to nsys's session,
    not to the launching process's group, and outlives it (natively, the next server found its port taken; in a
    container, removing the container did this)."""
    return [NSYS, "shutdown", f"--session={SESSION}", "--kill=sigkill"]


def nsys_stop_cmd():
    return [NSYS, "stop", f"--session={SESSION}"]
