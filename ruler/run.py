"""The ruler on the GPU box, at the start of every study session (box.sh ruler for S1, scripts/session.sh for the
others): launch latency, achievable bandwidth, the DRAM counters against a known stream, and the synthetic busy
workload under the same nsys commands the engines get, traced and untraced, and a chain of PDL graph nodes
(ruler.pdl); then ruler.validate (Gate G1: the parser within 1 percentage point of the truth).

    SESSION=S1-20261007 [GRID=configs/grid/v1.yaml] python -m ruler.run   (scripts/ruler.sh)

Every command is built by the grid's own builders: servebench.containers.runner_for (the pinned vLLM image, or vLLM's
venv on a box without Docker, DA_RUNTIME=native) and servebench.nsys.capture.nsys_cmd for every traced run, the DRAM
counters' included, so the ruler and the cells are measured with the same commands. It holds the GPU's lock for its
whole run and waits for an idle GPU before each traced run (servebench.gpu; DA_IDLE_MODE=activity on a shared box).
While it runs it logs the GPU's processes five times a second and their SM activity once a second (nvidia-smi pmon), and
records the processes already present as residents: ruler.validate judges every window's contention from these.
The version set and the SM clock come from the grid (GRID, configs/grid/v1.yaml by default, read by anatomy.grid.load,
so a refresh grid gets its base grid's clock): the clock is locked at clock_mhz for the run, as the cells run locked,
so the launch costs it prices every cell's launch term with come from the same clock; clock.json says whether the
lock held. Like the grid, it unwinds on SIGTERM and SIGHUP (clocks unlocked, its container removed), its loggers die
with it, and every step is bounded in time (STEP_TIMEOUT_S): a step that hangs fails, its container removed, and an
unattended session goes on to its grid.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys

from anatomy.grid import cpusets, default_mounts, load, set_aside
from servebench import ROOT, cgroup, clocks, containers, env, gpu, results
from servebench.containers import OUT, WORK
from servebench.engines.base import Release, version_set
from servebench.nsys.capture import metric_set, nsys_cmd

DEVICE = "0"
NAME = "da-ruler"     # its container's name, natively its processes' DA_UNIT marker
STEP_TIMEOUT_S = 1800  # one ruler step: minutes when it works
LOGGERS = {  # file: shell loop (GPU 0 only: on a two-GPU box other work on GPU 1 is not contention)
    "gpu-processes.log": "while :; do printf '%s %s\\n' \"$(date +%s.%N)\" \"$(nvidia-smi -i 0 "
                         "--query-compute-apps=pid --format=csv,noheader | tr '\\n' ',')\"; sleep 0.2; done",
    "gpu-activity.log": "while :; do printf '%s %s\\n' \"$(date +%s.%N)\" \"$(nvidia-smi pmon -i 0 -c 1 -s u "
                        "2>/dev/null | awk '!/^#/ {printf \"%s:%s,\", $2, $4}')\"; done",
}


def release(versions):
    """vLLM at the version set the session measures (the grid's): its image, or its venv, hosts the ruler, as
    anatomy.grid hosts the cells."""
    return Release.load("vllm", version_set(versions)[1]["vllm"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = p.parse_args(argv)
    session, nsys = os.environ.get("SESSION"), os.environ.get("DA_NSYS")
    if not session or not nsys:
        raise SystemExit("set SESSION and DA_NSYS (the Nsight Systems install; provision*.sh writes it)")
    grid = load(ROOT / os.environ.get("GRID", "configs/grid/v1.yaml"))  # a refresh grid takes its base's clock
    versions, clock_mhz = grid["versions"], grid.get("clock_mhz")
    out = results.RAW_ROOT / session / "ruler"
    # the box's mounts, as a unit's (anatomy.grid.default_mounts): the HF cache and the engines' compile caches under
    # the home directory, never the session's ruler/ (Triton's and Inductor's cache files landed there once, and
    # their *.json went up with the session), Nsight from DA_NSYS
    mounts = default_mounts(out)
    run, _ = containers.runner_for(release(versions), mounts)
    cpuset, _ = cpusets(env.gpu_local_cpus(DEVICE), quota=cgroup.cpu_quota())  # the engines' own cores (anatomy.grid)
    argv_for = lambda cmd, perf_counters=False: run(cmd, name=NAME, mounts=mounts, gpu=DEVICE,  # noqa: E731
                                                    cpuset=cpuset, perf_counters=perf_counters)

    def script(name, *args):
        return ["python3", f"{WORK}/ruler/{name}", *args, "--out", OUT]

    steps = [("launch latency", argv_for(script("launch_latency.py")), True),
             ("bandwidth", argv_for(script("bandwidth.py")), True)]
    # an empty graph node's cost split, node by node, between its kernel's duration and the gap before the next
    # (ruler.validate.empty_node_split): what the engines' in-graph gaps should be read against
    steps.append(("empty nodes traced", argv_for([*nsys_cmd(f"{OUT}/empty-node", "node"),
                                                  *script("launch_latency.py", "--profile")]), False))
    mset = metric_set(env.gpu_query("name", device=DEVICE).get("name"))
    if mset:  # the DRAM counters against a stream of known bytes, under the metrics captures' own switches;
        # refused without root and CAP_SYS_ADMIN (a container-hosted GPU), and the run goes on without them
        steps.append(("DRAM counters", argv_for([*nsys_cmd(f"{OUT}/dram-calibration", "metrics", metrics_set=mset),
                                                 *script("bandwidth.py", "--profile")], perf_counters=True), False))
    for mode in ("graph", "node"):
        for variant in ("eager", "graph"):
            steps.append((f"busy {mode}/{variant}",
                          argv_for([*nsys_cmd(f"{OUT}/busy-{mode}-{variant}", mode),
                                    *script("synthetic_busy.py", "--mode", mode, "--variant", variant)]), True))
    for variant in ("eager", "graph"):  # the same windows untraced: what the cells' occupancy divides busy by
        steps.append((f"busy untraced/{variant}",
                      argv_for(script("synthetic_busy.py", "--mode", "none", "--variant", variant)), True))
    # a chain of PDL graph nodes of known work against the launch term's overlap model (ruler.pdl): traced node by
    # node, then untraced three times, so the truth's spread between runs is known (what a choice between the raw and
    # the corrected model must clear: anatomy.decompose.suggested_overlap_model); a card or Triton without PDL writes
    # a skip note, and a failure leaves the model unchecked
    steps.append(("pdl chain traced", argv_for([*nsys_cmd(f"{OUT}/pdl-node", "node"),
                                                *script("pdl.py", "--mode", "node")]), False))
    steps.append(("pdl chain untraced", argv_for(script("pdl.py", "--mode", "none", "--passes", "3")), False))
    if a.dry_run:
        for label, cmd, _ in steps:
            print(f"# {label}\n{shlex.join(map(str, cmd))}")
        return 0
    set_aside(out)
    out.mkdir(parents=True, exist_ok=True)
    # tmux kill-session, kill <pid>: unwind, so the finally blocks run (clocks unlocked, loggers and container ended)
    handlers = {sig: signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
                for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        locked = bool(clock_mhz) and gpu.lock_clocks(clock_mhz)  # the cells' clock, when root allows it
        (out / "clock.json").write_text(json.dumps({"sm_clock_lock_mhz": clock_mhz if locked else None}) + "\n")
        # the card it measured, beside its numbers: tidy lends one session's calibration to another only between
        # the same card type and Nsight version (ruler.validate.card)
        (out / "host.json").write_text(json.dumps(env.host(), indent=1) + "\n")
        print(f"ruler: SM clock {'locked at ' + str(clock_mhz) + ' MHz' if locked else 'not locked'}", flush=True)
        try:
            return measure(steps, out)
        finally:
            if locked:
                gpu.unlock_clocks()
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def step(cmd, timeout=STEP_TIMEOUT_S, sh=containers.run, remove=containers.remove_unit):
    """One step's exit code (124 when it ran out of time). Its container, or natively its marked processes, never
    outlive it: removed on a timeout and when the ruler unwinds (a signal) while it runs."""
    def leftovers():
        remove(NAME)

    try:
        return sh(cmd, cwd=ROOT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        leftovers()
        return 124
    except BaseException:
        try:
            leftovers()
        except containers.DockerHung:
            pass
        raise


def measure(steps, out):
    """The steps on an idle GPU, with the process loggers running; then ruler.validate."""
    gpu.wait_for_idle_gpu()
    residents = gpu.compute_pids(DEVICE) if gpu.IDLE_MODE == "activity" else []
    (out / "gpu-residents.txt").write_text("".join(f"{p}\n" for p in residents))
    loggers = [subprocess.Popen(["bash", "-c", loop], stdout=open(out / name, "w"), stderr=subprocess.DEVNULL,
                                start_new_session=True, preexec_fn=containers._die_with_parent)
               for name, loop in LOGGERS.items()]
    try:
        # the clock the card held through the run (clocks.csv, as a unit's): what every cell's clock is read against
        # (ruler.validate.card sm_mhz_held), since a card may hold less than the lock asked (the H100 PCIe VM held
        # 1620 MHz of 1755, idle and busy alike, no slowdown bit set) and the calibration is taken at what it held
        with clocks.ClockLog(out / "clocks.csv"):
            for label, cmd, required in steps:
                gpu.wait_for_idle_gpu()
                print(f"ruler: {label}", flush=True)
                code = step(cmd)
                if code and required:
                    raise SystemExit(f"ruler: {label} failed (exit {code})")
                if code:
                    print(f"ruler: {label} refused (exit {code}); the run goes on without it", file=sys.stderr)
    finally:
        for proc in loggers:
            containers.teardown(proc)
    from ruler.validate import main as validate
    return validate([str(out)])


if __name__ == "__main__":
    sys.exit(main())
