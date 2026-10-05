"""L2 units: one server container per unit, its batch sweep driven from the host.

- served (the timing capture at L2): per batch B, `repeats` pairs of AIPerf invocations, long (gen_len) and short
  (SHORT_GEN) alternating, each a warm-up wave and a measured wave of B requests at concurrency B (servebench.aiperf).
  Every measured wave becomes a "long" or "short" record in driver.jsonl, the shape an L1 driver writes, with the step
  time estimated by servebench.cadence from the SSE timestamps, so the analysis reads L1 and L2 records alike. The
  client runs on its own cores (grid.cpusets).
- profiled (graph / node / metrics / hostload): the server runs under `nsys launch`; per batch B one untimed
  warm-up wave (SHORT_GEN tokens, outside the capture), then one wave of gen_len tokens from the threaded stream
  client inside one capture, `nsys start` before its requests and `nsys
  stop` after its last token; anatomy.tidy cuts the centred window from the trace's step anchors, counted back from
  the wave's last step (servebench.drivers.common.profile_window), so the latency of nsys start, which ran from 0.3 s
  natively to over 0.9 s through docker exec, never decides what the trace holds. Same mechanism for all four servers;
  no profiler endpoint needed. A hostload unit's waves run while spinners contend for the engine's cores, from the
  moment the server is ready, inside its container (servebench.containers.Contention).
A unit has a deadline (the grid's --timeout): every AIPerf and nsys call is bounded by what is left of it and ended
with its whole process group when it runs out (servebench.containers.run), and a batch that would start after it is
recorded as an error, so a hung client or server cannot stall an unattended session.
The server's own log is kept as container.log: start-up configuration and KV capacity are read from it. The server is
a container (Docker) or a process group (the native runtime, servebench.containers).
"""
import json
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from servebench import containers
from servebench.aiperf import RAW, SERVER_METRICS, aiperf_cmd, client_whole_ns, read_raw, server_metrics, server_step_ns
from servebench.cadence import differential, estimate, stream_span_ns
from servebench.cgroup import cpu_stat, since
from servebench.containers import CLEANUP_TIMEOUT_S, DOCKER, OUT, Contention, teardown, translate
from servebench.drivers.common import profile_window, prompts
from servebench.engines.base import Plan
from servebench.nsys.capture import nsys_shutdown_cmd, nsys_start_cmd, nsys_stop_cmd
from servebench.results import read_meta, update_meta
from servebench.stream_client import Wave

PLAN = Plan(model="", batches=())  # the waves' lengths, repeats, margin and window: the L1 drivers', one definition
REPEATS, SHORT_GEN = PLAN.repeats, PLAN.short_gen_len
NSYS_TIMEOUT_S = 300   # nsys start / stop / shutdown: seconds when they work (0.3-0.9 s in validation)
CONTENTION_WAIT_S = 30  # a hostload unit's busy loops must all be up within this, after the server is ready


@contextmanager
def pinned(cpuset):
    """This thread, and the threads it starts meanwhile (the profiled wave's streams), on cpuset's cores; the old
    affinity back after. The grid process is not pinned, so without this the client could share the engine's cores
    while the untraced step it is set against was measured by AIPerf on the client's own."""
    if not cpuset:
        yield
        return
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, containers.cores(cpuset))
    try:
        yield
    finally:
        os.sched_setaffinity(0, before)


def running(name, sh=containers.run):
    r = sh([*DOCKER, "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True, timeout=30)
    return r.returncode == 0 and r.stdout.strip() == "true"


def wait_ready(url, timeout, sleep=time.sleep, opener=urllib.request.urlopen, alive=lambda: True):
    """True once url answers 200; False on timeout, or at once when the server's container has exited (a model that
    fails to load must fail its unit in seconds, not after the timeout)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not alive():
            return False
        try:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return False
            with opener(url, timeout=min(5, remaining)) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        sleep(max(0, min(2, end - time.monotonic())))
    return False


def record(out, rec):
    with open(Path(out) / "driver.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def container_name(argv):
    return argv[argv.index("--name") + 1]


def port_free(port, wait_s=90, sleep=time.sleep):
    """True once the port binds on 0.0.0.0 without SO_REUSEADDR, waiting up to wait_s for an earlier server to let go.
    Binding, not connecting: a killed server's accepted connections stay in TIME_WAIT for 60 s, nothing listens, yet
    a server whose listener had no SO_REUSEADDR (TensorRT-LLM's) leaves them blocking the next server's bind (vLLM once
    failed three units in a row after TRT-LLM's, then started)."""
    import socket

    if not port:
        return True
    end = time.monotonic() + wait_s
    while time.monotonic() < end:
        with socket.socket() as s:
            try:
                s.bind(("0.0.0.0", int(port)))
                return True
            except OSError:
                pass
        sleep(max(0, min(1, end - time.monotonic())))
    return False


def checked(done, argv):
    """A command run in the server's environment (nsys start/stop) must succeed: an ignored failure left a profile
    record with no trace and a unit that counted as ok."""
    if getattr(done, "returncode", 0):
        raise RuntimeError(f"{' '.join(map(str, argv[:3]))} exit {done.returncode}: {(done.stderr or '')[-300:]}")
    return done


class DockerServer:
    """The server as a detached container: docker run -d, inspect, exec, logs, rm."""

    def __init__(self, cmds, out, sh):
        self.argv, self.name, self.sh = cmds["server"], container_name(cmds["server"]), sh
        self.port, self.log = cmds.get("port"), Path(out) / "container.log"

    def start(self, left=lambda cap: cap):
        """None once the container runs, else why not. Removing the last server and starting this one keep their own
        bounds (CLEANUP_TIMEOUT_S), never the unit's: a daemon that cannot answer is a broken box whatever time the
        unit has left, and raises (DockerHung), which the grid counts towards giving up; bounded by what was left, a
        llama.cpp server part that started with none raised the same and a timeout read as a broken box. Only the wait
        for the port is the unit's."""
        containers.remove(self.name, sh=self.sh)  # a stopped server left by a crashed run
        if not port_free(self.port, wait_s=left(90)):  # host networking: the same port as the last server's
            return f"server did not start: port {self.port} still in use by an earlier server"
        try:
            started = self.sh(self.argv, capture_output=True, text=True, timeout=CLEANUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"server did not start: docker run -d did not return in {CLEANUP_TIMEOUT_S} s"
        return None if started.returncode == 0 else f"server did not start: {started.stderr[-400:]}"

    def alive(self):
        return running(self.name, self.sh)

    def exec(self, argv, timeout=None):
        return checked(self.sh([*DOCKER, "exec", self.name, *argv], capture_output=True, text=True, timeout=timeout),
                       argv)

    def stop(self):
        """Its log saved, the container removed. A slow removal after a unit whose waves all ran is noted, not raised:
        the data stands, and a daemon that is really hung raises at the next start's removal (counted there)."""
        try:
            with open(self.log, "ab") as log:
                self.sh([*DOCKER, "logs", self.name], stdout=log, stderr=subprocess.STDOUT, timeout=CLEANUP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            pass
        try:
            containers.remove(self.name, sh=self.sh)
        except containers.DockerHung as e:
            with open(self.log, "a") as log:
                log.write(f"\n[anatomy.served] {e}\n")


class NativeServer:
    """The server as a process group on the host (servebench.containers.native_run): its log written as it runs, an
    nsys command run directly with the container paths translated, the whole group killed at the end."""

    def __init__(self, cmds, out, sh):
        self.argv, self.paths, self.sh = cmds["server"], cmds.get("paths") or [], sh
        self.log = Path(out) / "container.log"
        self.port, self.proc = cmds.get("port"), None
        self.env_dir, self.marker = cmds.get("env_dir"), cmds.get("marker")  # servebench.containers.teardown's
        self.nsys_session = cmds.get("nsys_session")  # run under `nsys launch` (anatomy.grid.commands says so)

    def start(self, left=lambda cap: cap):
        if not port_free(self.port, wait_s=left(90)):
            return f"server did not start: port {self.port} still in use by an earlier server"
        self.logf = open(self.log, "ab")
        try:
            self.proc = subprocess.Popen(self.argv, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            return f"server did not start: {e!r}"
        return None

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def exec(self, argv, timeout=None):
        return checked(self.sh([translate(a, self.paths) for a in argv], capture_output=True, text=True,
                               timeout=timeout), argv)

    def stop(self):
        """The nsys session first (the application of `nsys launch` belongs to it, not to our group), then the whole
        group and the venv's strays, always (servebench.containers.teardown)."""
        if self.proc is None:
            return
        if self.nsys_session:
            try:
                self.sh([translate(a, self.paths) for a in nsys_shutdown_cmd()], capture_output=True,
                        timeout=NSYS_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                pass  # the group and its strays are killed below either way
        teardown(self.proc, self.marker, self.env_dir, grace_s=60)
        self.logf.close()


def run_served(unit, cmds, out, *, sh=containers.run, aiperf="aiperf", client_cpuset=None, ready_timeout=1800,
               repeats=REPEATS, margin=PLAN.margin, profile_steps=PLAN.profile_steps, ready=wait_ready, wave_cls=Wave,
               metrics_set="gh100", timeout=None, clock=time.monotonic, probe=True):
    """Runs one L2 unit; returns None when every batch ran, else an error string (one failing batch does not stop the
    others: its error is recorded and the next batch runs). timeout: the unit's deadline in seconds from now; probe:
    whether this server ends the unit (a server per batch: the last), so the agreement probe runs on it."""
    out = Path(out)
    end = None if timeout is None else clock() + timeout

    def left(cap=None):
        """Seconds left before the deadline (cap at most), None without one."""
        if end is None:
            return cap
        rest = max(end - clock(), 0.0)
        return rest if cap is None else min(rest, cap)

    if left() == 0:  # a server per batch: an earlier server spent the unit's time, so this one never starts; its
        # batches are errors in the words anatomy.grid.timed_out reads (a timeout, never a broken box)
        for b in unit.batches:
            record(out, {"kind": "error", "batch": b, "error": "the unit's deadline passed before its server started"})
        return "; ".join(f"B={b}: the unit's deadline passed before its server started" for b in unit.batches)
    for fname, body in (cmds.get("files") or {}).items():
        (out / fname).write_text(body)
    base = cmds["ready"].rsplit("/", 1)[0]
    server = (NativeServer if cmds.get("runtime") == "native" else DockerServer)(cmds, out, sh)
    errors = []
    try:
        failed = server.start(left)  # a timed-out docker run may still have started a container
        if failed:
            return failed
        wait = left(ready_timeout)
        if not ready(cmds["ready"], wait, alive=server.alive):
            return f"server not ready (exited, or not up after {wait:.0f} s): see container.log"
        record(out, {"kind": "info", "level": "L2", "unit": unit.key, "server": cmds["server"],
                     "runtime": cmds.get("runtime", "docker"), "gen_len": unit.gen_len, "short_gen_len": SHORT_GEN,
                     "repeats": repeats, "margin": margin, "profile_steps": profile_steps})
        spin = Contention(cmds.get("contend"), None if cmds.get("runtime") == "native" else server.name)
        with spin:  # a hostload unit's spinners, from a ready server to its last wave
            # no batch before every loop runs: an uncontended first wave (llama.cpp at batch 1, the most host-bound
            # cell) would test nothing and pass
            unready = bool(cmds.get("contend")) and not spin.wait_ready(left(CONTENTION_WAIT_S))
            if unready:
                errors.append(spin.problem() or f"contention: the busy loops were not all up in {CONTENTION_WAIT_S} s")
            for b in [] if unready else unit.batches:
                if left() == 0:
                    errors.append(f"B={b}: the unit's deadline passed")
                    record(out, {"kind": "error", "batch": b, "error": "the unit's deadline passed"})
                    continue
                try:
                    if unit.capture == "served":
                        spans = {}
                        # long and short waves alternating, L S L S L S, as at L1 (a drift in the card's state falls
                        # on both alike): the differential (long - short) / (gen - short) is the L2 step's independent
                        # check for every server (SGLang and TRT-LLM expose no per-token metric by default)
                        for r in range(repeats):
                            for kind, gen, tag in (("long", unit.gen_len, "r"), ("short", SHORT_GEN, "s")):
                                art, cpu = out / f"aiperf-b{b}-{tag}{r}", cpu_stat()
                                with open(out / "client.log", "ab") as log:
                                    code = sh(aiperf_cmd(base, unit.model, unit.model, b, unit.prompt_len, gen, art,
                                                         seed=r, aiperf=aiperf, cpuset=client_cpuset,
                                                         extra_inputs=cmds.get("request")),
                                              stdout=log, stderr=subprocess.STDOUT, timeout=left()).returncode
                                if code:
                                    raise RuntimeError(f"aiperf exit {code}")
                                cpu = since(cpu)  # a CPU quota's throttling during the wave, where there is a quota
                                seqs, meta = read_raw(art / RAW)
                                spans.setdefault(kind, []).append(stream_span_ns(seqs))
                                wall = max((s[-1][0] for s in seqs if s), default=0) - min((s[0][0] for s in seqs if s),
                                                                                       default=0)
                                rec = {"kind": kind, "batch": b, "repeat": r, "gen_len": gen, "wall_ns": wall,
                                       "span_ns": spans[kind][-1],
                                       "tokens": [m["tokens"] for m in meta], "client": "aiperf",
                                       # the prompts as the server tokenised AIPerf's text: they must be prompt_len
                                       "prompt_tokens": [m.get("prompt_tokens") for m in meta],
                                       "prompt_len": unit.prompt_len,
                                       "cadence": estimate(seqs, margin),  # a short wave's own stall and gap checks
                                       **({"cpu": cpu} if cpu else {})}
                                if kind == "long":
                                    rec |= {"cadence_tail": estimate(seqs, margin, after_tokens=SHORT_GEN),
                                            "server_step_ns": server_step_ns(server_metrics(art / SERVER_METRICS)),
                                            "client_whole_ns": client_whole_ns(seqs)}
                                record(out, rec)
                        record(out, {"kind": "differential", "batch": b,
                                     "step_ns": [differential(lw, sw, unit.gen_len, SHORT_GEN) if lw and sw else None
                                                 for lw, sw in zip(spans["long"], spans["short"])]})
                    else:
                        # one wave inside one capture, as at L1 (servebench.drivers.common.profile_window): the
                        # capture opens before the wave and closes after it, and tidy cuts the centred window from the
                        # trace's own step anchors, counted back from the wave's last step, so no nsys latency matters
                        from_end, steps = profile_window(unit.gen_len, margin, profile_steps)
                        with pinned(client_cpuset):  # one untimed wave at this batch first, outside the capture: the
                            # kernels JIT-compiled at first use and the batch's CUDA graph captured are not traced
                            warm = wave_cls(f"{base}/v1/completions", prompts(b, unit.prompt_len, 0, "w"), SHORT_GEN,
                                            model=unit.model, extra=cmds.get("request")).start(timeout=left())
                            warm.join()
                            if warm.errors:
                                raise RuntimeError(f"warm-up: {len(warm.errors)} of {b} streams failed: "
                                                   f"{warm.errors[0][1][:200]}")
                        t0 = time.perf_counter()
                        server.exec(nsys_start_cmd(f"{OUT}/{unit.capture}-b{b}", unit.capture, metrics_set=metrics_set),
                                    timeout=left(NSYS_TIMEOUT_S))
                        t1, cpu = time.perf_counter(), cpu_stat()
                        with pinned(client_cpuset):  # the wave's threads on the client's cores, as AIPerf's are
                            wave = wave_cls(f"{base}/v1/completions", prompts(b, unit.prompt_len, 0, "p"),
                                            unit.gen_len, model=unit.model, extra=cmds.get("request"))
                            wave.start(timeout=left())
                            wave.join()
                        t2, cpu = time.perf_counter(), since(cpu)
                        server.exec(nsys_stop_cmd(), timeout=left(NSYS_TIMEOUT_S))
                        record(out, {"kind": "profile", "batch": b, "gen_len": unit.gen_len,
                                     "window_from_end": from_end, "profile_steps": steps, "errors": wave.errors,
                                     "nsys_start_s": t1 - t0, "nsys_stop_s": time.perf_counter() - t2,
                                     "cadence": estimate(wave.chunks, margin), "client": "stream_client",
                                     **({"cpu": cpu} if cpu else {})})
                        if wave.errors:  # a wave that never decoded is no window (TRT-LLM's 400s once read ok)
                            raise RuntimeError(f"{len(wave.errors)} of {b} streams failed: {wave.errors[0][1][:200]}")
                except Exception as e:  # noqa: BLE001  one batch's failure is recorded, the sweep goes on
                    # a call cut off because the unit's time ran out says so in the words anatomy.grid.timed_out
                    # counts, so a server hanging at its largest batch is skipped like any unit that timed out
                    late = isinstance(e, (subprocess.TimeoutExpired, TimeoutError)) and left() == 0
                    errors.append(f"B={b}: the unit's deadline passed ({e!r})" if late else f"B={b}: {e!r}")
                    record(out, {"kind": "error", "batch": b,
                                 "error": f"the unit's deadline passed ({e!r})" if late else repr(e)})
        if cmds.get("contend"):
            # per server, under its batches: a server per batch (llama.cpp) runs its own loops for each, and one
            # record for the unit would keep the last server's alone (servebench.results.contention_for reads them)
            key = ",".join(str(b) for b in unit.batches)
            known = (read_meta(out) or {}).get("contention") or {}
            update_meta(out, contention={**({} if "cpu_share" in known else known), key: spin.record()})
            if not unready and spin.problem():
                errors.append(spin.problem())
        if unit.capture == "served" and cmds.get("greedy_endpoint") and probe and not errors and left() != 0:
            # the agreement probe, after the unit's last wave so it can never change one, and only on a server that
            # ran every batch inside the unit's deadline (a hung server is stopped, not probed for minutes more)
            greedy(base + cmds["greedy_endpoint"], unit.model, cmds.get("request"), out,
                   deadline=None if end is None else end, clock=clock)
    finally:
        server.stop()
    return "; ".join(errors) or None


PROBE_TIMEOUT_S = 120


def greedy(url, model, extra, out, deadline=None, clock=time.monotonic, opener=urllib.request.urlopen):
    """A server's agreement probe, as an L1 driver's (servebench.drivers.common.probe): each prompt of
    configs/correctness/prompts.json alone, its greedy tokens' ids (llama-server's /completion with return_tokens, the
    release's own request fields, which make it greedy), one "greedy" record each; a failure is recorded, never the
    unit's. Each request waits PROBE_TIMEOUT_S at most, never past the unit's deadline: the prompts left then are not
    sent."""
    from servebench.drivers.common import probe_prompts

    ids, n = probe_prompts(model)
    for i, prompt in enumerate(ids or []):
        wait = PROBE_TIMEOUT_S if deadline is None else min(PROBE_TIMEOUT_S, deadline - clock())
        if wait <= 0:
            record(out, {"kind": "greedy", "prompt": i, "error": "the unit's deadline passed"})
            continue
        body = {**(extra or {}), "prompt": prompt, "n_predict": n, "return_tokens": True, "stream": False}
        try:
            req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with opener(req, timeout=wait) as r:
                record(out, {"kind": "greedy", "prompt": i, "ids": json.loads(r.read())["tokens"]})
        except Exception as e:  # noqa: BLE001  a check beside the measurement
            record(out, {"kind": "greedy", "prompt": i, "error": repr(e)})
