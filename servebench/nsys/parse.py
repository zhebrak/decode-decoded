"""GPU activity from an Nsight Systems SQLite export (nsys profile --export=sqlite), and the two fractions this project
reports for a window of decode steps.

Tables read (nsys exporter schema; a table exists only when the capture holds rows of its kind, so a missing one is
an empty list, never an error):
- CUPTI_ACTIVITY_KIND_KERNEL: one row per kernel; start/end in ns; shortName/demangledName are StringIds keys;
  graphNodeId is set for a kernel that ran inside a CUDA graph traced with --cuda-graph-trace=node.
- CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET: start/end; a copy's copyKind (2: device to host) and bytes
  for the memcpy step anchor.
- CUPTI_ACTIVITY_KIND_GRAPH_TRACE (newer exporters: _V2, _V3, the highest version present wins): one row per graph
  launch under --cuda-graph-trace=graph, from its first node's start to its last node's end, with no kernel rows for
  the nodes.
- GPU_METRICS (timestamp, metricId, value) with names in TARGET_INFO_GPU_METRICS (metricId, metricName): the samples
  of --gpu-metrics-devices captures. Only DRAM read and write throughput (% of the card's peak) are read, for the
  measured-bytes cross-check on the byte model behind MBU.

Two fractions, because a graph-mode trace cannot see inside a graph:
- occupied: |union of kernels, memcpys, memsets and graph launches| / window. A graph launch counts as busy from its
  first node to its last, gaps between nodes included, so this is the time the GPU had work queued; 1 - occupied is
  the time it waited on the host. Read from a graph-mode trace, whose overhead is low.
- kernel-active: |union of kernels| / window from a node-mode trace, where every kernel inside a graph has a row.
  Node tracing slows the host, so only kernel durations are taken from it, never its gaps.
Intervals are united by sort-and-sweep, never summed: streams overlap.

Clocks (checked on real nsys 2026.5.1 exports from an A100): activity rows (kernels, graph launches, runtime calls) are
in ns relative to the session start, while ANALYSIS_DETAILS gives the capture range in absolute time; the session
start is TARGET_INFO_SESSION_START_TIME.systemClockNs. And under --cuda-graph-trace=graph, launches that ran before
cudaProfilerStart (warm-up replays) are exported too, with negative timestamps: step anchors are kept only inside the
capture range, or they would stretch the window back over the warm-up.

A window is a whole number of decode steps: from the start of one step's anchor to the start of a later one's, so no
partial step biases either fraction. The anchor is a graph launch of the dominant graph, the one launched most in the
capture's later launches, where decode runs (graph mode: one full-graph replay per decode step), or the first kernel
of each launch of it (node mode: the kernels of one launch share a correlationId), or the dominant device-to-host copy
(memcpy-dtoh: the step's logits brought to the host, whose size is the full batch's only while every sequence
decodes), or a kernel-name regex for eager runs, chosen per engine from S1's traces.
"""
import re
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import median

KERNEL = "CUPTI_ACTIVITY_KIND_KERNEL"
MEMCPY = "CUPTI_ACTIVITY_KIND_MEMCPY"
MEMSET = "CUPTI_ACTIVITY_KIND_MEMSET"
GRAPH_TRACE = re.compile(r"^CUPTI_ACTIVITY_KIND_GRAPH_TRACE(?:_V(\d+))?$")
GPU_METRICS = "GPU_METRICS"
DRAM = {"read": re.compile(r"^DRAM Read Bandwidth", re.I), "write": re.compile(r"^DRAM Write Bandwidth", re.I)}

# First match wins, on the demangled name: FlashAttention-3 kernels are cutlass templates, so attention is tested
# before GEMM; the norm, rope and activation kernels before attention, since FlashInfer's live in flashinfer::norm
# (flashinfer::norm::FusedAddRMSNormKernel, TensorRT-LLM's and SGLang's RMSNorm: 136 and 91 us a step at 0.6B batch 1
# on the study card, once counted as attention). Written against the kernel names in the validation and study traces;
# "other" is reported, never hidden.
KERNEL_CLASSES = (
    ("norm_rope_act", r"norm|rope|rotary|silu|act_and_mul|gelu|swiglu|activation"),
    ("attention", r"flash|fmha|attn|attention|paged|mha|decode.*kv|kv.*decode|xqa|mla"),
    ("sampling", r"argmax|sampl|topk|top_k|topp|top_p|gumbel|multinomial|radix_sort|cub::.*Sort|log_softmax|softmax"),
    ("gemm", r"gemm|cutlass|cublas|xmma|nvjet|matmul|wgmma|mul_mat|splitk|gemv"),
    ("copy_cast", r"copy|cast|convert|fill|index|gather|scatter|cat_|elementwise|reduce"),
)


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    name: str = ""
    stream: int = 0
    graph_node: int | None = None
    graph: int | None = None
    correlation: int | None = None

    @property
    def ns(self):
        return self.end - self.start


class Trace:
    """One SQLite export. Rows are read once, on first use."""

    def __init__(self, path):
        self.path = Path(path)
        self.con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.tables = {r[0] for r in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self._cache = {}

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _columns(self, table):
        return {r[1] for r in self.con.execute(f"PRAGMA table_info({table})")}

    def _rows(self, table, named=False):
        if table not in self.tables:
            return []
        cols = self._columns(table)
        col = lambda c: f"k.{c}" if c in cols else "NULL"  # noqa: E731  exporter versions differ in optional columns
        names, joins = [], ""
        if named and "StringIds" in self.tables:
            for alias, c in (("d", "demangledName"), ("s", "shortName")):
                if c in cols:
                    names.append(f"{alias}.value")
                    joins += f" LEFT JOIN StringIds {alias} ON {alias}.id = k.{c}"
        name = f"COALESCE({', '.join(names)}, '')" if names else "''"
        sql = (f"SELECT k.start, k.end, {name}, {col('streamId')}, {col('graphNodeId')}, {col('graphId')}, "
               f"{col('correlationId')} FROM {table} k{joins} ORDER BY k.start")
        return [Interval(*r) for r in self.con.execute(sql)]

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def graph_table(self):
        """The GRAPH_TRACE table with the highest version suffix, or None."""
        found = [(int(m.group(1) or 1), t) for t in self.tables if (m := GRAPH_TRACE.match(t))]
        return max(found)[1] if found else None

    def kernels(self):
        return self._get("kernels", lambda: self._rows(KERNEL, named=True))

    def memcpys(self):
        return self._get("memcpys", lambda: self._rows(MEMCPY))

    def memsets(self):
        return self._get("memsets", lambda: self._rows(MEMSET))

    def dtoh_copies(self):
        """[(start, bytes)] of the device-to-host copies (copyKind 2), in start order."""
        def load():
            if MEMCPY not in self.tables or not {"copyKind", "bytes"} <= self._columns(MEMCPY):
                return []
            return list(self.con.execute(f"SELECT start, bytes FROM {MEMCPY} WHERE copyKind = 2 ORDER BY start"))
        return self._get("dtoh", load)

    def graph_launches(self):
        return self._get("graphs", lambda: self._rows(self.graph_table) if self.graph_table else [])

    def activity(self):
        """Everything that occupies the GPU: kernels, copies, sets and graph launches."""
        return [*self.kernels(), *self.memcpys(), *self.memsets(), *self.graph_launches()]

    def gpu_metrics(self):
        """{metric name: [(timestamp, value)]} from GPU-metrics sampling, sorted by time; {} when the capture has none.
        Timestamps are on the activity rows' clock, ns from the session start (checked on the A100's ga100 exports:
        samples span 0 to the capture's end, as the kernels do)."""
        def load():
            if GPU_METRICS not in self.tables or "TARGET_INFO_GPU_METRICS" not in self.tables:
                return {}
            names = dict(self.con.execute("SELECT DISTINCT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
            out = {}
            sql = f"SELECT timestamp, metricId, value FROM {GPU_METRICS} ORDER BY timestamp"
            for ts, mid, v in self.con.execute(sql):
                out.setdefault(names.get(mid, str(mid)), []).append((ts, v))
            return out
        return self._get("metrics", load)

    def session_start(self):
        """The absolute time activity rows count from (TARGET_INFO_SESSION_START_TIME.systemClockNs), or None."""
        if "TARGET_INFO_SESSION_START_TIME" in self.tables and \
                "systemClockNs" in self._columns("TARGET_INFO_SESSION_START_TIME"):
            row = self.con.execute("SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()
            return row[0] if row else None
        return None

    def capture_range(self):
        """(start, stop) of the capture on the activity rows' clock: ANALYSIS_DETAILS (absolute) less the session start
        (its own start when the session table is absent), else the span of activity."""
        if "ANALYSIS_DETAILS" in self.tables and {"startTime", "stopTime"} <= self._columns("ANALYSIS_DETAILS"):
            row = self.con.execute("SELECT startTime, stopTime FROM ANALYSIS_DETAILS").fetchone()
            if row and row[0] is not None and row[1] is not None:
                base = self.session_start()
                base = row[0] if base is None else base
                return row[0] - base, row[1] - base
        act = self.activity()
        return (min(i.start for i in act), max(i.end for i in act)) if act else None


def union_ns(intervals, lo=None, hi=None):
    """Length of the union of intervals, clipped to [lo, hi]: sort by start, sweep, never sum overlaps."""
    spans = sorted((max(i.start, lo) if lo is not None else i.start, min(i.end, hi) if hi is not None else i.end)
                   for i in intervals)
    total, cur_s, cur_e = 0, None, None
    for s, e in spans:
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (cur_e - cur_s if cur_e is not None else 0)


def node_launches(kernels):
    """[(start, graph id)] per graph launch in a node-mode trace: the kernels of one launch share the correlationId of
    the cudaGraphLaunch that ran them, and the launch starts with the first of them."""
    groups = {}
    for k in kernels:
        if k.graph_node is not None:
            start, graph = groups.get(k.correlation, (k.start, k.graph))
            groups[k.correlation] = (min(start, k.start), graph)
    return list(groups.values())


def dominant(pairs):
    """The starts of the (start, key) pairs whose key is the most frequent among the later half of them in time (ties:
    among all), sorted. A whole-wave capture opens on the prefill and decode runs to its end, and the prefill can launch
    another graph more often in all, even in one unbroken run (llama-server at 256 slots of 0.6B: its prefill graph 262
    times in the first 7 s, its decode graph 256 times in the next 52); a batch shape met as the batch fills and again
    as it drains is a few launches at either end."""
    ordered = sorted(pairs, key=lambda p: p[0])
    counts, late = {}, {}
    for i, (_, k) in enumerate(ordered):
        counts[k] = counts.get(k, 0) + 1
        late[k] = late.get(k, 0) + (2 * i >= len(ordered))
    top = max(counts, key=lambda k: (late[k], counts[k]), default=None)
    return [s for s, k in ordered if k == top]


DTOH = "memcpy-dtoh"
FULL_BATCH = ("graph", DTOH)  # anchors that mark only the full batch's steps: the last is the first sequence's end


def step_starts(trace, anchor="graph"):
    """Start times of decode steps in the trace.
    anchor="graph":       launches of the dominant graph (dominant: decode's): in a fixed-batch decode window, one
                          full-graph replay per step. Graph mode reads GRAPH_TRACE rows, node mode the kernels' launch
                          groups;
    anchor="memcpy-dtoh": runs of device-to-host copies of the dominant size: the logits of every decoding sequence
                          brought to the host once a step, or under GPU sampling the sampled rows, four a slot
                          (every llama.cpp arm: its graph launch misses the eager step where the KV view grows, and
                          with graphs off get_rows runs twice a step);
    anchor=<regex>:       kernels whose name matches (eager runs).
    Only those inside the capture range count, before the dominant one is chosen: warm-up launches before
    cudaProfilerStart are exported too, with negative times (two thirds of an L1 graph export's launches)."""
    rng = trace.capture_range()
    inside = (lambda pairs: [p for p in pairs if rng[0] <= p[0] <= rng[1]]) if rng else (lambda pairs: pairs)
    if anchor == "graph":
        return dominant(inside([(g.start, g.graph) for g in trace.graph_launches()] or node_launches(trace.kernels())))
    if anchor == DTOH:
        return dominant(inside(copy_runs(trace.dtoh_copies(), [*trace.kernels(), *trace.graph_launches()])))
    rx = re.compile(anchor)
    return [s for s, _ in inside([(k.start, k.name) for k in trace.kernels() if rx.search(k.name)])]


def copy_runs(copies, work):
    """[(start, total bytes)] per run of device-to-host copies with no kernel or graph launch starting between them: a
    step's copies to the host as one anchor. llama.cpp copies its logits once a step, but under GPU sampling four
    4-byte rows per slot (the sampled token, its probability, the candidate and its logit), 4 x B copies a step."""
    starts = sorted(w.start for w in work)
    runs = []
    for start, size in sorted(copies):
        if runs and bisect_left(starts, start) == bisect_left(starts, runs[-1][2]):
            runs[-1][1] += size
            runs[-1][2] = start
        else:
            runs.append([start, size, start])
    return [(start, size) for start, size, _ in runs]


def whole_steps(starts, skip=0, steps=None):
    """(lo, hi, n): a window of n whole steps, from the start of step `skip` to the start of step skip + n (all
    remaining when steps is None). None when fewer than two anchors remain."""
    starts = sorted(starts)[skip:]
    if steps is not None:
        starts = starts[:steps + 1]
    return (starts[0], starts[-1], len(starts) - 1) if len(starts) >= 2 else None


def dram(trace, spans):
    """Mean DRAM read and write throughput (% of the card's peak) over the samples inside the spans [(lo, hi)], with
    the sample counts; None when the trace has no DRAM metrics. A sample averages the period before it, so a caller
    that needs a clean average trims a period off each span's start."""
    metrics = trace.gpu_metrics()
    out = {}
    for key, rx in DRAM.items():
        name = next((n for n in metrics if rx.search(n)), None)
        vals = [v for t, v in metrics.get(name, ()) if any(lo <= t <= hi for lo, hi in spans)]
        out[f"dram_{key}_pct"] = sum(vals) / len(vals) if vals else None
        out[f"dram_{key}_samples"] = len(vals)
    return out if out["dram_read_pct"] is not None else None


def classify(name, classes=KERNEL_CLASSES):
    for cls, rx in classes:
        if re.search(rx, name, re.IGNORECASE):
            return cls
    return "other"


def summarise(path, anchor="graph", skip=0, steps=None):
    """One trace's numbers for its window of whole steps: `steps` steps from anchor `skip` (negative: counted back from
    the last anchor, as servebench.drivers.common.profile_window places a window); with no anchors found it falls back
    to the activity span with n=None (fractions only)."""
    with Trace(path) as t:
        rng = t.capture_range()
        starts = step_starts(t, anchor)
        win = whole_steps(starts, skip, steps)
        source = "anchors" if win else None
        if win is None:
            # no step anchors: the first to the last activity inside the capture range, not the range itself (which
            # starts ~30 ms before the first kernel in real exports); fractions only, and the caller flags it
            act = [i for i in t.activity() if rng is None or (i.end > rng[0] and i.start < rng[1])]
            if not act:
                return {"path": str(path), "empty": True}
            win, source = (min(i.start for i in act), max(i.end for i in act), None), "activity"
        lo, hi, n = win
        wall = hi - lo
        kernels = t.kernels()
        occupied_ns = union_ns(t.activity(), lo, hi)
        kernel_ns = union_ns(kernels, lo, hi)
        in_win = [k for k in kernels if k.end > lo and k.start < hi]
        by_class = {}
        for k in in_win:
            by_class.setdefault(classify(k.name), []).append(k)
        graph_node_kernels = sum(1 for k in in_win if k.graph_node is not None)
        under = overlaps(in_win)
        launches = [g for g in t.graph_launches() if g.end > lo and g.start < hi]
        per_step = (lambda x: x / n) if n else (lambda x: None)
        # where every kernel is a row (node mode, eager runs) a step holding twice the kernels is two steps; in a graph
        # trace anchored on its launches each replay is one row, so a step holding a burst of eager kernels between
        # two launches is one whose launch never came; under any other anchor an eager step there is just a step
        if source == "anchors" and (graph_node_kernels or not launches):
            rows = kernels
        elif source == "anchors" and anchor == "graph":
            rows = [*kernels, *t.graph_launches()]
        else:
            rows = None
        crowded = step_row_ratio([x for x in starts if lo <= x <= hi], rows) if rows is not None else None
        return {
            "path": str(path), "graph_table": t.graph_table, "window_ns": [lo, hi], "window_source": source,
            "anchors": len(starts), "steps": n, "wall_ns": wall, "step_row_ratio": crowded,
            "step_ns": per_step(wall),
            "occupied": occupied_ns / wall if wall else None,
            "occupied_ns_per_step": per_step(occupied_ns),
            "kernel_active": kernel_ns / wall if wall and kernels else None,
            "kernel_ns_per_step": per_step(kernel_ns) if kernels else None,
            "kernels": len(in_win), "kernels_per_step": per_step(len(in_win)),
            "overlapped_per_step": per_step(len(under)), "overlap_ns_mean": sum(under) / len(under) if under else 0.0,
            "overlaps_ns": under,  # each overlapped kernel's, for the launch term's per-kernel clamp (anatomy.tidy)
            # copies and sets that no kernel covers: memory the GPU moves outside any kernel (anatomy.decompose)
            "copy_ns_per_step": per_step(union_ns([*kernels, *t.memcpys(), *t.memsets()], lo, hi) - kernel_ns)
            if kernels else None,
            "graph_node_kernels": graph_node_kernels, "graph_launches": len(launches),
            "memcpy_ns": union_ns(t.memcpys(), lo, hi), "memset_ns": union_ns(t.memsets(), lo, hi),
            "classes_ns_per_step": {c: per_step(union_ns(ks, lo, hi)) for c, ks in sorted(by_class.items())},
            "top_kernels": top_kernels(in_win, lo, hi),
            "dram": dram(t, [(lo, hi)]),
        }


def step_row_ratio(starts, rows):
    """The most activity rows (kernels; in a graph trace, kernels and launches) any step of the window holds, over the
    median step's: about 2 in a node trace where an anchor never fired and two steps were counted as one (llama.cpp's
    graph anchor missed the eager step where its padded KV view grows: 1,460 kernels in one interval against 730),
    hundreds in a graph trace; about 1 where every step was marked, a slow step included (it runs the same kernels).
    None with fewer than two steps or no rows; anatomy.tidy judges it (missed_anchor)."""
    starts = sorted(starts)
    rows = sorted(r.start for r in rows)
    counts = [bisect_left(rows, b) - bisect_left(rows, a) for a, b in zip(starts, starts[1:])]
    typical = median(counts) if len(counts) >= 2 else 0
    return max(counts) / typical if typical else None


def overlaps(kernels):
    """How far each kernel that starts before its same-stream predecessor ends runs under it, ns: a programmatic
    dependent launch (or a concurrent branch), whose launch and prologue the predecessor's tail hides. The launch term
    charges such a kernel only the node cost its overlap leaves exposed (in validation: vLLM's 28 a step at batch 1,
    0.75 us each; llama.cpp's nearly every kernel)."""
    ends, out = {}, []
    for k in sorted(kernels, key=lambda k: k.start):
        prev = ends.get(k.stream)
        if prev is not None and k.start < prev:
            out.append(prev - k.start)
        ends[k.stream] = max(prev or k.end, k.end)
    return out


def numbered_exports(folder, base):
    """<base>.<n>.sqlite in n order: nsys numbers the exports of --capture-range-end=repeat from 1."""
    def index(p):
        m = re.search(r"\.(\d+)\.sqlite$", p.name)
        return int(m.group(1)) if m else 0
    return sorted(Path(folder).glob(f"{base}.*.sqlite"), key=index)


def top_kernels(kernels, lo, hi, n=12):
    """[(name, count, clipped ns)] of the n kernels with the most time in the window: the attribution tables."""
    agg = {}
    for k in kernels:
        a = agg.setdefault(k.name, [0, 0])
        a[0] += 1
        a[1] += max(0, min(k.end, hi) - max(k.start, lo))
    return [(name, c, ns) for name, (c, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:n]]
