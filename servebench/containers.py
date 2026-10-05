"""The one docker run template every engine runs under, so containers differ only in image, command and environment.

    --gpus device=0 --ipc=host --network=host --shm-size=16g, the same --cpuset-cpus for every engine,
    memlock and stack ulimits (NGC's recommendation), seccomp unconfined, SYS_ADMIN only for GPU-metrics captures.
Why the privileges:
    - seccomp unconfined: Nsight Systems inside the container needs the syscalls Docker's default profile refuses
      (perf_event_open for its sampling, ptrace-style process attach). It is set for every unit, timed ones too, so a
      timed and a profiled run of a cell differ only by the profiler: the default profile's filter runs on every
      syscall, and host-bound steps are made of syscalls (futex, epoll, the CUDA driver's ioctls);
    - SYS_ADMIN: the GPU-metrics sampler reads the GPU's performance counters, which the driver reserves to
      administrators; only the DRAM-counter units (perf_counters) get it;
    - --network=host: the client and nsys's start/stop reach the server on loopback; every server binds 127.0.0.1
      (servebench.engines.adapters.HOST), so nothing listens on the host's interfaces.

Mounts (host -> container):
    repo   -> /work   ro   drivers and servebench.cadence
    out    -> /out    rw   the unit's records, logs, traces
    hf     -> /hf     rw   Hugging Face cache (HF_HOME; offline: models are fetched once by scripts/fetch_models.sh)
    cache  -> /cache  rw   per-engine compile and JIT caches (XDG_CACHE_HOME), kept across units so a cache warmed
                           by the timing run serves the profiled run
    models -> /models ro   GGUF files for llama.cpp
    nsys   -> /opt/nsys ro the pinned Nsight Systems install

The native runtime (DA_RUNTIME=native) is for a box that cannot run Docker: a container-hosted GPU (a cloud pod) has
Docker's default capabilities and a seccomp filter (no mount or user namespaces, no overlay mounts), so neither dockerd
nor rootless Docker nor Podman runs, and a ptrace sandbox would distort the host time being measured. Each engine runs
from its own venv at the pinned version (scripts/provision_native.sh: <DA_ENVS>/<engine>-<version>, frozen into the
session's host/), with the same command the container would run: the container paths above are translated to host
paths, the environment is set explicitly, and the same CPU set applies (taskset). llama.cpp's /app is its build's bin.
"""
import os
import shlex
import signal
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

# The docker command. DA_DOCKER="sudo -n docker" on a host whose user is not in the docker group (the shared harness
# validation box), rather than changing that host's groups.
DOCKER = shlex.split(os.environ.get("DA_DOCKER", "docker"))
RUNTIME = os.environ.get("DA_RUNTIME", "docker")  # docker | native
# where every container (and, natively, the translated command) finds the repo, the unit's output directory, the HF
# cache, the engines' compile caches, the models and Nsight: the one spelling of each path
WORK, OUT, HF, CACHE, MODELS, NSYS_ROOT = "/work", "/out", "/hf", "/cache", "/models", "/opt/nsys"
ENVS = Path(os.environ.get("DA_ENVS", "~/da-envs")).expanduser()
SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PASS_THROUGH = ("LD_LIBRARY_PATH", "CUDA_HOME", "LANG", "LC_ALL", "TZ", "USER", "LOGNAME")  # the box's, when set


@dataclass(frozen=True)
class Mounts:
    repo: Path
    out: Path
    hf: Path
    cache: Path
    models: Path
    nsys: Path | None = None

    def volumes(self):
        pairs = [(self.repo, WORK, "ro"), (self.out, OUT, "rw"), (self.hf, HF, "rw"),
                 (self.cache, CACHE, "rw"), (self.models, MODELS, "ro")]
        if self.nsys:
            pairs.append((self.nsys, NSYS_ROOT, "ro"))
        return [arg for host, ctr, mode in pairs for arg in ("-v", f"{host}:{ctr}:{mode}")]

    def paths(self, app=None):
        """[(container path, host path)] for the native runtime, longest first."""
        pairs = [(WORK, self.repo), (OUT, self.out), (HF, self.hf), (CACHE, self.cache),
                 (MODELS, self.models)] + ([(NSYS_ROOT, self.nsys)] if self.nsys else []) \
            + ([("/app", app)] if app else [])
        return sorted(((c, str(h)) for c, h in pairs), key=lambda p: -len(p[0]))


def translate(arg, paths):
    """A container path at the start of arg (or of the value in key=value) as the host path; anything else as is."""
    for ctr, host in paths:
        if arg == ctr or arg.startswith(ctr + "/"):
            return host + arg[len(ctr):]
    key, eq, value = arg.partition("=")
    if eq and value != arg:
        moved = translate(value, paths)
        if moved != value:
            return f"{key}={moved}"
    return arg


# the caches under the mounted /cache, so they outlive a container: XDG's (torch.compile, FlashInfer), the CUDA
# driver's JIT cache (the pinned llama.cpp image has no Hopper machine code, only compute_90 PTX, so every container
# would compile its kernels again at first use) and Triton's (keyed on HOME, not XDG)
BASE_ENV = {"HF_HOME": HF, "HF_HUB_OFFLINE": "1", "XDG_CACHE_HOME": CACHE, "PYTHONUNBUFFERED": "1",
            "CUDA_CACHE_PATH": "/cache/nv", "CUDA_CACHE_MAXSIZE": "4294967296", "TRITON_CACHE_DIR": "/cache/triton",
            # caches that ignore XDG and sit under $HOME or /tmp otherwise, so every container compiled again
            # (SGLang's first prefill-graph capture took 35-56 s in each container on the A100, 1 s once cached)
            "SGLANG_CACHE_DIR": "/cache/sglang", "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
            "FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer"}


def docker_run(image, cmd, *, name, mounts, env=None, cpuset=None, entrypoint=(), gpu="0", perf_counters=False,
               detach=False):
    """The docker run argv for one unit. entrypoint is the image's own wrapper when it needs one (NGC images set up
    their library paths in /opt/nvidia/nvidia_entrypoint.sh); cmd runs after it. gpu=None leaves the GPU out (the
    CPU rehearsal of the L2 path). A detached server is not --rm: a server that dies at start-up must leave its log
    for anatomy.served to copy into container.log before it removes the container."""
    argv = [*DOCKER, "run", *([] if detach else ["--rm"]), "--name", name,
            *(["--gpus", f"device={gpu}"] if gpu is not None else []),
            "--ipc=host", "--network=host", "--shm-size=16g", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
            "--security-opt", "seccomp=unconfined"]
    if detach:
        argv.append("-d")
    if cpuset:
        argv += ["--cpuset-cpus", cpuset]
    if perf_counters:
        argv += ["--cap-add", "SYS_ADMIN"]
    argv += mounts.volumes()
    for k, v in sorted({**BASE_ENV, **(env or {})}.items()):
        argv += ["-e", f"{k}={v}"]
    first, *rest = [*entrypoint, *cmd]
    return [*argv, "--entrypoint", first, image, *rest]


def native_dir(engine, version):
    return ENVS / f"{engine}-{version}"


def bundled_cuda(env_dir):
    """The CUDA toolkit an engine's own wheels bring (nvidia/cu13 in its site-packages), or None. Its kernels compile
    at first use with it, as they would in the engine's image: SGLang 0.5.20's DeepGEMM refuses an nvcc older than
    12.9, which a box's system toolkit can be (12.8 in validation)."""
    try:  # a dry run away from the box cannot see its envs: no bundled toolkit then
        hits = sorted(Path(env_dir).glob("lib/python3*/site-packages/nvidia/cu1*/bin/nvcc")) if env_dir else []
    except OSError:
        hits = []
    return hits[-1].parent.parent if hits else None


def native_run(image, cmd, *, name, mounts, env=None, cpuset=None, entrypoint=(), gpu="0", perf_counters=False,
               detach=False, env_dir=None, cuda="system"):
    """The native argv for what docker_run would run (see the module docstring): env with the container's variables
    on host paths, the engine's venv first on PATH, the GPU by CUDA_VISIBLE_DEVICES, then taskset and the command.
    image, name, entrypoint (NGC's wrapper sets up paths the venv already has) and detach (anatomy.served starts a
    server itself) do not apply; perf_counters needs nothing more, the process runs with the box's own rights."""
    env_dir = Path(env_dir) if env_dir else None
    paths = mounts.paths(app=env_dir / "bin" if env_dir else None)
    variables = {**BASE_ENV, **(env or {})}
    variables = {k: translate(str(v), paths) for k, v in variables.items()}
    variables |= {k: os.environ[k] for k in PASS_THROUGH if k in os.environ and k not in variables}
    # the toolkit kernels compiled at first use build with: the box's, or the one the engine's wheels bring where its
    # release says so (configs/engines: native_cuda). Chosen per engine, never guessed: SGLang 0.5.20's DeepGEMM
    # refuses a system nvcc 12.8, while vLLM 0.30.0's bundle mixes a 13.4 nvcc with older runtime headers
    # ("CUDA compiler and CUDA toolkit headers are incompatible") and builds with the box's 12.8.
    bundled = bundled_cuda(env_dir) if cuda == "bundled" else None
    if cuda == "bundled" and bundled is None and env_dir and env_dir.is_dir():  # on the box: never the system's quietly
        raise RuntimeError(f"{env_dir} has no bundled CUDA toolkit (its release says native_cuda: bundled)")
    variables["CUDA_HOME"] = str(bundled) if bundled else variables.get("CUDA_HOME", "/usr/local/cuda")
    path = [str(env_dir / "bin")] if env_dir else []
    variables |= {"PATH": ":".join([*path, f"{variables['CUDA_HOME']}/bin", SYSTEM_PATH]), "HOME": str(Path.home())}
    if env_dir:
        variables["VIRTUAL_ENV"] = str(env_dir)
    if gpu is not None:
        variables["CUDA_VISIBLE_DEVICES"] = str(gpu)
    variables["DA_UNIT"] = name  # inherited by whatever the run spawns: teardown finds its strays by it
    argv = ["env", "-i", *(f"{k}={v}" for k, v in sorted(variables.items()))]
    if cpuset:
        argv += ["taskset", "-c", cpuset]
    return [*argv, *(translate(a, paths) for a in cmd)]


def runner_for(release, mounts, runtime=None):
    """(run, extras) for one engine release on this box, chosen once for the grid and the ruler alike: run(cmd, **the
    docker_run keywords) -> argv, in the release's digest-pinned image or, natively, from its venv with the CUDA
    toolkit its arms.yaml names (native_cuda); extras, what a native run adds to a unit's commands (the runtime, the
    venv, the container paths' host equivalents)."""
    if (runtime or RUNTIME) == "native":
        env_dir = native_dir(release.engine, release.version)
        run = partial(native_run, release.digest_pinned_image, env_dir=env_dir,
                      cuda=release.extra.get("native_cuda", "system"))
        return run, {"runtime": "native", "env_dir": str(env_dir), "paths": mounts.paths(app=env_dir / "bin")}
    return partial(docker_run, release.digest_pinned_image), {}


def strays(marker, env_dir=None, proc=Path("/proc")):
    """Pids a native run left behind: every process whose environment carries DA_UNIT=marker (native_run sets it and
    children inherit it), and any process whose executable is the engine's own (env_dir/bin: the venv's python, the
    llama.cpp binaries) and that carries no DA_UNIT at all (a worker started with a cleaned environment). Never
    another run's: a process marked for a different unit is left alone, as is one that only mentions the environment
    in its arguments (a compiler building llama.cpp in it, a GGUF conversion in its own py/ env: a failed unit once
    killed a build running beside it), and one whose environment cannot be read."""
    tag, own = f"DA_UNIT={marker}".encode(), f"{env_dir}/bin/".encode() if env_dir else None
    return [pid for pid, environ, argv in processes(proc) if environ is not None
            and (tag in environ or (own and argv and argv[0].startswith(own)
                                    and not any(e.startswith(b"DA_UNIT=") for e in environ)))]


def teardown(proc, marker=None, env_dir=None, grace_s=0):
    """Ends a run started in its own session (start_new_session) whole: its process group (TERM and up to grace_s to
    exit, then KILL), then its strays outside the group (an engine's own spawned workers: TRT-LLM's MPI workers kept the
    GPU and the port). A worker left behind would hold the GPU, and on a shared box the activity gate would admit the
    next unit beside it."""
    def group(sig):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass

    if grace_s:
        group(signal.SIGTERM)
        try:
            proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            pass
    group(signal.SIGKILL)
    proc.wait()
    kill(strays(marker, env_dir) if marker else ())


def kill(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def run(argv, *, timeout=None, capture_output=False, **kw):
    """subprocess.run, in its own session, and on a timeout the whole process group ended before TimeoutExpired is
    raised: a timed-out command's children (AIPerf's worker processes, `hf upload` under sync_raw.sh) never outlive it
    beside the next unit."""
    if capture_output:
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.PIPE)
    proc = subprocess.Popen(argv, start_new_session=True, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        teardown(proc)
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


CLEANUP_TIMEOUT_S = 120


class DockerHung(RuntimeError):
    """The Docker daemon did not answer in time: the box is broken, not the unit, so the grid counts it towards giving
    the session up (anatomy.grid.GIVE_UP_AFTER) instead of waiting on it unit after unit."""



def remove_unit(name):
    """Ends whatever a unit left running, by its name: its container under Docker, natively the processes marked
    with it (DA_UNIT), as the grid and the ruler both do when a unit runs out of time or the session unwinds."""
    remove(name if RUNTIME == "docker" else None, marker=name)

def remove(name=None, marker=None, timeout=CLEANUP_TIMEOUT_S, sh=run):
    """The one way a unit's leftovers go: the processes carrying its marker (a native run's, and its workers), and its
    container (docker rm -f, bounded in time: a daemon that cannot remove a container within it raises DockerHung)."""
    kill(strays(marker) if marker else ())
    if name:
        try:
            sh([*DOCKER, "rm", "-f", name], capture_output=True, timeout=timeout)
        except OSError:  # no Docker on this box: nothing of ours is a container
            pass
        except subprocess.TimeoutExpired as e:
            raise DockerHung(f"docker rm -f {name} did not return in {timeout} s") from e


def clear(timeout=CLEANUP_TIMEOUT_S, proc=Path("/proc")):
    """Everything any unit of ours left: every process carrying a DA_UNIT marker, every da-* container. What the grid
    runs when the GPU will not go idle on a box that is the study's alone (the idle gate gave up: a container a killed
    grid left running is the likeliest holder). A daemon that does not answer raises DockerHung."""
    kill(marked(proc))
    try:
        ids = run([*DOCKER, "ps", "-aq", "--filter", "name=^da-"], capture_output=True, text=True,
                  timeout=timeout).stdout.split()
        if ids:
            run([*DOCKER, "rm", "-f", *ids], capture_output=True, timeout=timeout)
    except OSError:
        pass
    except subprocess.TimeoutExpired as e:
        raise DockerHung(f"docker did not answer in {timeout} s while clearing the GPU") from e


def marked(proc=Path("/proc")):
    """Pids of every process carrying a DA_UNIT marker, whichever unit's."""
    return [pid for pid, environ, _ in processes(proc) if any(e.startswith(b"DA_UNIT=") for e in environ or ())]


def processes(proc=Path("/proc")):
    """(pid, environ entries, argv) of every process, this one excepted; environ is None where this user may not read
    it (a container's root-owned processes), argv is readable for all."""
    for d in sorted(proc.iterdir(), key=lambda d: int(d.name) if d.name.isdigit() else -1):
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            argv = (d / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        except OSError:
            continue
        try:
            environ = (d / "environ").read_bytes().split(b"\0")
        except OSError:
            environ = None
        yield int(d.name), environ, argv


def cores(cpuset):
    """[0, 1, 2, 5] from "0-2,5"; the cores this process may use when cpuset is None (every core the host shows where
    the OS has no affinity call, macOS: planning and dry runs work there, collection does not)."""
    if not cpuset:
        return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    out = []
    for part in cpuset.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


# The hostload capture's contention: one busy loop per core of the engine's cpuset. The loop is a shell's, named
# SPIN_TAG ($0 of `sh -c`), so it is found in /proc whoever started it: natively the grid, under Docker the engine's
# container (docker exec), whose processes the host's /proc shows too. Native loops die with the grid
# (PR_SET_PDEATHSIG), a container's with the container: no sweep for leftovers is needed.
SPIN_TAG = "da-spin"
SPIN_LOOP = "while :; do :; done"


def spin_argv(core):
    return ["taskset", "-c", str(core), "sh", "-c", SPIN_LOOP, SPIN_TAG]


def spinners(proc=Path("/proc")):
    """Pids of the busy loops running now: their argv ends `-c <loop> da-spin`, whatever shell runs it (taskset has
    exec'd into sh; in the TensorRT-LLM image `sh` is a wrapper that execs `/bin/bash`, so the loops read
    `/bin/bash -c ... da-spin` and a match on the name `sh` saw none while they ran, and failed the unit)."""
    return [pid for pid, _, argv in processes(proc) if argv[-3:-1] == [b"-c", SPIN_LOOP.encode()]
            and argv[-1] == SPIN_TAG.encode()] if proc.is_dir() else []


def cpu_share(pid, proc=Path("/proc")):
    """The share of one core a process has had since it started: (utime + stime) over its wall time; None once gone."""
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        uptime = float((proc / "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    hz = os.sysconf("SC_CLK_TCK")
    cpu, started = (int(fields[11]) + int(fields[12])) / hz, int(fields[19]) / hz
    return cpu / (uptime - started) if uptime > started else None


def _die_with_parent():
    """preexec_fn: SIGKILL this child when the grid dies, however it dies (PR_SET_PDEATHSIG survives taskset's exec)."""
    import ctypes

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # 1: PR_SET_PDEATHSIG


class Contention:
    """with Contention(cpuset, container): one busy loop pinned to each core of cpuset while the block runs. The
    engine's host threads then share every core they run on, so host work slows while the GPU's work is unchanged (the
    hostload capture's test of occupancy's assumption). The loops must sit in the engine's own cgroup: under Docker
    they run inside its container (docker exec, once the container is up), since CFS gives loops in another cgroup only
    that cgroup's share of a core (in a test, a thread beside 14 such loops kept 94% of its core); natively they are
    the grid's children, killed with it if it dies. `ready` is set once every loop runs (wait_ready: anatomy.served
    waits for it before the first batch); `measured` holds each loop's share of its core, sampled while the loops run
    (a container's die with it, before the block ends): what the contention really took, for meta.json. A watcher
    that fails (a Docker call timing out) says so in `error`, and problem() fails the unit: a hostload unit whose loops
    never ran tested nothing."""

    def __init__(self, cpuset, container=None, sh=run, period_s=5.0):
        import threading

        self.cpus, self.container, self.sh, self.period_s = cores(cpuset) if cpuset else [], container, sh, period_s
        self.procs, self.measured, self.error = [], [], None
        self.ready, self.stop = threading.Event(), threading.Event()

    def __enter__(self):
        import threading

        if not self.cpus:
            return self
        if self.container is None:
            self.procs = [subprocess.Popen(spin_argv(c), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           start_new_session=True, preexec_fn=_die_with_parent) for c in self.cpus]
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        return self

    def _watch(self):
        try:
            if self.container is not None:
                self._launch_in_container()
            while not self.stop.is_set() and not self.ready.is_set():
                if len(spinners()) >= len(self.cpus):
                    self.ready.set()
                    self.sample()
                else:
                    self.stop.wait(0.5)
            while not self.stop.wait(self.period_s):
                self.sample()
        except Exception as e:  # noqa: BLE001  the watcher never dies silently: problem() reports it
            self.error = repr(e)

    def _launch_in_container(self):
        """The loops inside the engine's container, once it runs (a slow docker inspect is retried, not fatal)."""
        while not self.stop.is_set():
            try:
                r = self.sh([*DOCKER, "inspect", "-f", "{{.State.Running}}", self.container], capture_output=True,
                            text=True, timeout=30)
                if r.returncode == 0 and r.stdout.strip() == "true":
                    break
            except subprocess.TimeoutExpired:
                pass
            self.stop.wait(1.0)
        if self.stop.is_set():
            return
        loops = " ".join(f'(taskset -c {c} sh -c "{SPIN_LOOP}" {SPIN_TAG} 2>/dev/null || sh -c "{SPIN_LOOP}" '
                         f'{SPIN_TAG}) &' for c in self.cpus)
        r = self.sh([*DOCKER, "exec", "-d", self.container, "sh", "-c", f"{loops} wait"], capture_output=True,
                    text=True, timeout=60)
        if r.returncode:
            self.error = f"docker exec of the busy loops: exit {r.returncode}: {(r.stderr or '')[-200:]}"

    def wait_ready(self, timeout):
        """True once every loop runs, False after timeout seconds (or at once without loops to run)."""
        return bool(self.cpus) and self.ready.wait(timeout)

    def sample(self):
        shares = [x for pid in spinners() if (x := cpu_share(pid)) is not None]
        if shares:
            self.measured = sorted(round(x, 3) for x in shares)

    def __exit__(self, *exc):
        if not self.cpus:
            return
        self.sample()
        self.stop.set()
        self.thread.join(timeout=10)
        for proc in self.procs:
            teardown(proc)

    def problem(self):
        """Why the contention did not happen (no loop was ever measured), else None."""
        if self.cpus and not self.measured:
            return "contention: no busy loop ran on the engine's cores" + (f" ({self.error})" if self.error else "")
        return None

    def record(self):
        """What meta.json keeps: the loops asked for, each one's measured share of its core, and an error if any."""
        return {"spinners": len(self.cpus), "cpu_share": self.measured, **({"error": self.error} if self.error else {})}
