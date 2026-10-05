"""The grid: configs/grid/<name>.yaml expanded into units (servebench.results.Unit, one container run each, its batch
sweep inside), filtered by feasibility, ordered for a session, and run, or printed with --dry-run.

    python -m anatomy.grid configs/grid/v1.yaml --dry-run                  every unit's command, no GPU needed
    python -m anatomy.grid configs/grid/v1.yaml --session S2 --run         run S2's units (resumes: done units skip)
    python -m anatomy.grid configs/grid/v1.yaml --session S2 --list        unit keys and batch lists only

Expansion rules:
- level "headline" is L1 where the release has an offline API, else L2 (llama.cpp);
- arm "non-default" is every arm of the release except defaults, that runs at the level;
  named arms a release lacks are skipped (gpu-sampling exists for llama.cpp only);
- batches above the naive KV bound (servebench.physics Shape.max_batch at prompt + gen tokens, memory_fraction of the
  card) are dropped; "max" becomes that bound. The bound is an upper one: the driver records whatever the engine's
  own KV capacity admits, and a batch it cannot hold at once shows as an error or an all-active window that never
  opens, never as a silently preempted cell;
- captures: timing (the unprofiled sweep), graph / node / metrics / hostload (Nsight kinds,
  servebench.nsys.capture), served (L2 client), runner (L0); "all" means every feasible batch of the block, a list
  picks a subset. A hostload unit runs while spinner processes contend for the engine's cores
  (servebench.containers.Contention: inside its container under Docker);
- two blocks of one session asking for the same unit run it once, with the union of their batches.
Order: the session's sentinels first and last, its noise block (S1) just inside them; between, model by model (by
size), then engine, then block in the grid's order, then arm, and timing before captures. So everything one comparison
pairs runs back to back: a model's defaults cells, the same cells with one mitigation changed, at the server and at the
runner, so drift over a long session falls on both sides of a pair alike; and a model loads into a warm page cache and
a compile cache its timing unit warmed. The engines take turns in alphabetical order on every other model and in
reverse on the rest, so each pair of engines is measured both ways round and a drift over the session cannot side
with one engine at every model.

A refresh grid (configs/grid/refresh.yaml) names a base grid, one of its sessions and the engines whose release
changed: it is that session's blocks for those engines (every model and batch, so the page never shows a release
partly measured), with the base grid's sentinels for every engine, under the refresh's own version set.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import yaml

from anatomy.metrics import HEADLINE
from servebench import ROOT, cgroup, clocks, containers, env, gpu, results
from servebench.containers import HF, MODELS, OUT, Mounts
from servebench.engines.adapters import HOST, engine_for
from servebench.engines.base import Plan, Release, merge, render, version_set
from servebench.nsys.capture import KINDS, metric_set, nsys_cmd, nsys_launch_cmd
from servebench.physics import PEAKS, load_shape
from servebench.results import (
    ALWAYS_RUN,
    Unit,
    is_done,
    is_sentinel,
    model_name,
    read_jsonl,
    read_meta,
    read_status,
    unit_dir,
    update_meta,
    write_meta,
    write_status,
)

# A block names captures once for every engine; at L2 the unprofiled sweep is the served client, at L0 the runner.
LEVEL_CAPTURE = {"L2": {"timing": "served"}, "L0": {"timing": "runner"}}
CAPTURE_ORDER = ("timing", "runner", "served", *KINDS)  # the unprofiled sweep first, then the traces
DEFAULT_PORT = 8000  # a grid may set `port`
SYNC_TIMEOUT_S = 900   # a per-unit upload that hangs must not hold the GPU idle; session.sh syncs the whole session
SYNC_RETRY_EVERY = 5   # after a failed upload, the units still to sync are tried again this many units later
SYNC_EVERY = int(os.environ.get("DA_SYNC_EVERY", "1"))  # upload every this many units (the last always): one HF
                                                        # commit per unit idles the GPU for its duration
SYNC_GIVE_UP = 3       # failed upload rounds in a row: no more per-unit syncs (session.sh's final sync covers them)
SPIN = "hostload"      # the capture that runs under contention for the engine's cores
HUNG_AFTER = 2         # one engine's timeouts in a row on one model: its other units for that model are skipped
GIVE_UP_AFTER = 3      # units in a row that raise (the GPU never idles, nvidia-smi or Docker gone) or time out: the
                       # session ends there


def extended(path, seen=()):
    """A grid file with what it extends underneath: another grid, whole (itself resolved the same way), with this
    file's top-level settings on top, its overrides merged into the base's per engine (a validation run of the study's
    grid, a probe on a validation grid keeping the validation grid's overrides)."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"grids extend each other in a loop: {' -> '.join(map(str, (*seen, path)))}")
    grid = yaml.safe_load(path.read_text())
    if "extends" not in grid:
        return grid
    base = extended(ROOT / grid["extends"], (*seen, path))
    top = {k: v for k, v in grid.items() if k != "extends"}
    return {**base, **top, **({"overrides": merge(base["overrides"], top["overrides"])}
                              if "overrides" in base and "overrides" in top else {})}


def load(path):
    grid = extended(path)
    if "refresh" in grid:
        grid = refresh(grid)
    grid["versions_map"] = version_set(grid["versions"])[1]
    return grid


def refresh(spec):
    """A refresh grid (the module docstring) made whole from its base: the base session's blocks for the changed
    engines, renamed to the refresh's session, the base's sentinels and settings, the refresh's version set."""
    r = spec["refresh"]
    base = extended(ROOT / r["grid"])
    changed = set(r["engines"])
    blocks = {name: {**b, "session": r["as"], "engines": [e for e in b["engines"] if e in changed]}
              for name, b in base["blocks"].items() if b.get("session") == r["session"]
              and any(e in changed for e in b["engines"])}
    return {**{k: v for k, v in base.items() if k not in ("blocks", "noise")}, **spec, "blocks": blocks}


def levels_for(level, release):
    """The release's levels a block's level names: "headline" is the engine's headline level (anatomy.metrics.HEADLINE:
    the engine API, or llama.cpp's server), the one rule for the grid and the page alike."""
    if level == "headline":
        level = HEADLINE[release.engine]
    return [level] if level in release.levels else []


def arms_for(names, release, level):
    out = []
    for name in names:
        if name == "non-default":
            out += [a for a in release.arm_names if a != "defaults" and level in release.arm(a).levels]
        elif name in release.arms_raw and level in release.arm(name).levels:
            out.append(name)
    return list(dict.fromkeys(out))


def feasible(batches, model, prompt_len, gen_len, grid):
    """The block's batches a model can hold at once on the grid's card: the KV beside the weights in memory_fraction
    of the card, less the grid's kv_reserve_gib every engine also holds (activations, graph pools); "max" -> that
    bound, capped at the engines' max_num_seqs (grid max_batch), and added only when it is above every listed batch."""
    peak = PEAKS[grid["gpu"]]
    room = peak.memory * grid["memory_fraction"] - grid.get("kv_reserve_gib", 0) * 2 ** 30
    kv_bound = load_shape(model).max_batch(prompt_len + gen_len, room)
    bound = min(kv_bound, grid.get("max_batch", kv_bound))
    nums = sorted({b for b in batches if b != "max" and b <= bound})
    if "max" in batches and bound > 0 and (not nums or bound > nums[-1]):
        nums.append(bound)
    return tuple(nums)


def expand(grid, sessions=None, blocks=None, models=None):
    """The grid's units (filtered to sessions, blocks and models when given), in session order (module docstring)."""
    units = []
    for bname, block in grid["blocks"].items():
        if (sessions and block.get("session") not in sessions) or (blocks and bname not in blocks):
            continue
        ctx = {**grid["context"], **(block.get("context") or {})}
        for engine in block["engines"]:
            release = Release.load(engine, str(grid["versions_map"][engine]))
            for lvl in [lv for spec in block["levels"] for lv in levels_for(spec, release)]:
                for arm in arms_for(block["arms"], release, lvl):
                    cells = block.get("cells") or [[m, b] for m in block["models"] for b in block["batches"]]
                    by_model = {}
                    for m, b in cells:
                        by_model.setdefault(m, []).append(b)
                    for model, wanted in by_model.items():
                        if models and model not in models:
                            continue
                        ok = feasible(wanted, model, ctx["prompt_len"], ctx["gen_len"], grid)
                        for capture, which in block["captures"].items():
                            capture = LEVEL_CAPTURE.get(lvl, {}).get(capture, capture)
                            if lvl == "L0" and capture in KINDS:
                                raise ValueError(f"block {bname}: L0 runners are not profiled (no nsys wrapper for "
                                                 f"them); give L0 the timing capture only")
                            bs = ok if which == "all" else tuple(b for b in which if b in ok)
                            if bs:
                                units.append(Unit(engine=engine, version=release.version, arm=arm, level=lvl,
                                                  model=model, capture=capture, batches=bs, block=bname,
                                                  prompt_len=ctx["prompt_len"], gen_len=ctx["gen_len"]))
    order = {name: i for i, name in enumerate(grid["blocks"])}
    sizes = sorted({load_shape(u.model).params for u in units})
    engines = sorted({u.engine for u in units})

    def turn(u):  # engines alphabetically on every other model, in reverse on the rest (the module docstring)
        i, rank = sizes.index(load_shape(u.model).params), engines.index(u.engine)
        return rank if i % 2 == 0 else -rank

    # model by model, engine by engine, the blocks interleaved: a pair's two sides run back to back
    units.sort(key=lambda u: (load_shape(u.model).params, turn(u), order[u.block], u.arm,
                              CAPTURE_ORDER.index(u.capture)))
    return merge_same_key(units, {name: b.get("session") for name, b in grid["blocks"].items()})


def merge_same_key(units, session_of):
    """Units two blocks of one session both ask for (llama.cpp's gpu-sampling arm is a headline mitigation and a
    served-block arm) share a directory: one unit with the union of their batches, at the first block's place, so
    resume never skips the second block's batches as "done". Blocks of different sessions write to different session
    directories and are never merged."""
    merged = {}
    for u in units:
        k = (session_of.get(u.block), u.key)
        if k in merged:
            merged[k] = replace(merged[k], batches=tuple(sorted(set(merged[k].batches) | set(u.batches))))
        else:
            merged[k] = u
    return list(merged.values())


def sentinels(grid, tag):
    """The session's sentinel units for one position (start, end), and for "unlocked" the ones the grid marks to run
    again with the SM clock unlocked (one sentinel run unlocked shows what locking changes)."""
    out = []
    for s in grid.get("sentinels") or []:
        if tag == "unlocked" and not s.get("unlocked"):
            continue
        version = str(grid["versions_map"][s["engine"]])
        out.append(Unit(engine=s["engine"], version=version, arm=s["arm"], level=s["level"], model=s["model"],
                        capture=LEVEL_CAPTURE.get(s["level"], {}).get("timing", "timing"),
                        batches=tuple(s["batches"]), block=f"sentinel-{tag}", **grid["context"]))
    return out


def noise(grid, session, tag):
    """The session's noise units for one position (start, end): the grid's noise block (its `cells`, each a block of
    timing units), and the sentinels with it when it says so, each as `processes` separate processes (blocks
    noise-<tag>-p<n>), so anatomy.noise.noise_table reads the spread between processes: the tolerances' basis
    (configs/tolerances.yaml) and each comparison's uncertainty."""
    block = grid.get("noise")
    if not block or block.get("session") != session:
        return []
    cells = expand({**grid, "blocks": {name: {"captures": {"timing": "all"}, **b}
                                       for name, b in block["cells"].items()}})
    if block.get("with_sentinels"):
        cells += [replace(u, block="noise") for u in sentinels(grid, tag)]
    units = [replace(u, block=f"noise-{tag}-p{n}") for n in range(1, block.get("processes", 3) + 1) for u in cells]
    return merge_same_key(units, {u.block: session for u in units})  # a sentinel the block already has: one unit


def session_units(grid, session, models=None):
    """A session's units in run order; models (a lost host's recovery: the models not yet done) narrows its blocks,
    never its sentinels or its noise block."""
    return [*sentinels(grid, "start"), *noise(grid, session, "start"), *expand(grid, sessions={session}, models=models),
            *noise(grid, session, "end"), *sentinels(grid, "end"), *sentinels(grid, "unlocked")]


# commands ---------------------------------------------------------------------------------------------------

def gguf(model, version):
    """The GGUF llama.cpp loads, named for what made it: the llama.cpp release whose converter wrote it and the pinned
    snapshot it was converted from (configs/models/revisions.yaml), so a new pin or a new release is a new file, never
    a stale one reused (scripts/make_gguf.sh writes it, with its sha256 beside it). Always bf16, so it moves the same
    bytes as the other engines; the CPU rehearsal loads the same file."""
    return f"{MODELS}/gguf/llamacpp-{version}/{model_name(model)}-{(env.pinned(model) or 'unpinned')[:8]}-bf16.gguf"


def plan_for(unit):
    profiled = unit.capture in KINDS
    model = gguf(unit.model, unit.version) if unit.engine == "llamacpp" else unit.model
    if unit.engine == "trtllm" and (rev := env.model_revision(unit.model)):
        # the local snapshot, not the repo id: offline, TRT-LLM's server looked the id up online and came up with no
        # tokenizer, where the others read the same snapshot from the cache
        model = f"{HF}/hub/{env.hub_dir(unit.model).name}/snapshots/{rev}"
    return Plan(model=model, batches=unit.batches, prompt_len=unit.prompt_len, gen_len=unit.gen_len,
                repeats=0 if profiled else 3, profile="cuda" if profiled else "none")


def commands(unit, mounts, cpuset=None, port=DEFAULT_PORT, gpu="0", overrides=None, metrics_set="gh100",
             runtime=None):
    """{"run": argv} for an L0/L1 unit, {"server": argv, "ready": url, ...} for L2: what --dry-run prints and run
    executes. Profiled L1 units wrap the driver in nsys inside the container; profiled L2 units run the server under
    nsys launch, and anatomy.served opens the windows."""
    release = Release.load(unit.engine, unit.version)
    eng = engine_for(release)
    profiled = unit.capture in KINDS
    arm = release.arm(unit.arm, profile=profiled)
    over = (overrides or {}).get(unit.engine) or {}
    if over:  # a grid's per-engine args, env, L2 request fields and server_per_batch (a validation grid's, a
        # probe's): in the commands
        arm = replace(arm, args=merge(arm.args, over.get("args")), env={**arm.env, **(over.get("env") or {})})
    plan = plan_for(unit)
    entry = tuple(release.extra.get("entrypoint") or ())
    common = dict(name=unit.name, mounts=mounts, env=arm.env, cpuset=cpuset, entrypoint=entry, gpu=gpu,
                  perf_counters=unit.capture == "metrics")
    run, extras = containers.runner_for(release, mounts, runtime)  # the image, or (no Docker) the engine's venv
    if extras:
        extras["marker"] = unit.name  # native_run's DA_UNIT: teardown finds the run's strays by it
    if unit.capture == SPIN:  # the cores the spinners contend for: the engine's own (servebench.containers.Contention)
        extras["contend"] = cpuset or cpu_ranges(containers.cores(None))
    threads = len(containers.cores(cpuset)) if cpuset else None  # the engine's cores, for an engine with a setting
    if unit.level == "L0":
        return {"run": run(eng.runner_cmd(arm, plan, threads=threads), **common), **extras}
    if unit.level == "L1":
        cmd = eng.offline_cmd(arm, plan)
        if profiled:
            cmd = [*nsys_cmd(f"{OUT}/{unit.capture}", unit.capture, extra=arm.nsys, metrics_set=metrics_set), *cmd]
        return {"run": run(cmd, **common), **extras}
    def server(u):
        argv = eng.server_cmd(arm, plan_for(u), port, threads=threads)
        return run([*nsys_launch_cmd(u.capture, arm.nsys), *argv] if profiled else argv, detach=True, **common)

    # {its batches: its command} per server: one for the unit, or (a release that asks for it, llama.cpp) one per
    # batch, each sized for that batch alone: llama-server with idle slots ran a batch of 8 at 1.8x the step when its
    # sequences landed in slots that were not one run
    per_batch = over.get("server_per_batch", release.extra.get("server_per_batch"))  # a probe may turn it off
    parts = [(b,) for b in unit.batches] if per_batch else [unit.batches]
    servers = {",".join(map(str, bs)): server(replace(unit, batches=bs)) for bs in parts}
    return {"servers": servers, "files": eng.server_files(arm), "port": port, "ready": f"http://{HOST}:{port}/health",
            "request": {**(release.extra.get("request") or {}), **(over.get("request") or {})},
            "greedy_endpoint": release.extra.get("greedy_endpoint"), "nsys_session": profiled, **extras}


def cpusets(cpus, client_cores=4, engine_max=22, quota=None):
    """(engine cpuset, client cpuset) from the CPUs to use, in order (servebench.env.gpu_local_cpus): the engine the
    first up to engine_max, the L2 client the next client_cores, so they never share a core, every engine sees the
    same cores, and on a larger host the engine is held to 22 cores of the GPU's own node, as on the study's host.
    Under a CPU quota (servebench.cgroup.cpu_quota) only as many cores as it pays for are used, the client's share cut
    to a third of them: more cores than the quota would be throttled, not faster."""
    cpus = list(cpus)
    if quota and quota < len(cpus):
        cpus, client_cores = cpus[:quota], min(client_cores, max(1, quota // 3))
    if len(cpus) <= client_cores + 2:
        return None, None
    n = min(engine_max, len(cpus) - client_cores)
    return cpu_ranges(cpus[:n]), cpu_ranges(cpus[n:n + client_cores])


def cpu_ranges(cpus):
    """"0-21" from [0..21]; "0-3,8-11" from runs."""
    runs, out = [], []
    for c in cpus:
        if runs and c == runs[-1][1] + 1:
            runs[-1][1] = c
        else:
            runs.append([c, c])
    for lo, hi in runs:
        out.append(f"{lo}-{hi}" if hi > lo else str(lo))
    return ",".join(out)


# run --------------------------------------------------------------------------------------------------------

def execute(argv, log_path, timeout, marker=None, env_dir=None):
    """Runs argv with stdout and stderr to log_path; returns the exit code (124 on timeout, as coreutils). In its own
    process group, so a native run is ended whole (servebench.containers.teardown: the group, then its strays, found
    by its marker). A container that did not exit by itself is removed: on a timeout, and when the grid unwinds while
    it runs (SIGTERM or SIGHUP: the docker CLI is in its own session and gets no signal, so the engine would stay on
    the GPU). A removal the daemon does not answer raises servebench.containers.DockerHung."""
    name = argv[argv.index("--name") + 1] if "--name" in argv else None
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = 124
        except BaseException:  # the grid unwinding (a signal, Ctrl-C): the engine goes with it, then the unwinding
            try:
                containers.remove(name)
            except containers.DockerHung:
                pass
            containers.teardown(proc, marker, env_dir)
            raise
        try:
            if code == 124:
                containers.remove(name)
        finally:
            containers.teardown(proc, marker, env_dir)  # finished or not: an engine's workers outlive a driver
        return code


class ResumeMismatch(ValueError):
    """A completed unit belongs to another effective run specification; keep its evidence intact."""


def run_spec(unit, commands, revision, provenance):
    return {"unit": {k: v for k, v in unit.to_dict().items() if k != "block"}, "commands": commands,
            "model_revision": revision, "model_pin": env.pinned(unit.model),
            "execution": {k: provenance.get(k) for k in ("git", "versions", "runtime", "idle_mode", "validation",
                                                        "sm_clock_lock_mhz", "client_cpuset", "aiperf")}}


def resume_differences(meta, spec):
    """The fields in which a finished unit's recorded run (meta.json's run_spec) differs from the run asked for now:
    what the operator reads when a resumed session stops. `execution.git` is a new commit deployed mid-session; what
    is left then runs under a new session id."""
    old = meta.get("run_spec") or {}
    out = []
    for k, v in spec.items():
        was = old.get(k)
        if isinstance(v, dict) and isinstance(was, dict):
            out += [f"{k}.{f}" for f in sorted(set(v) | set(was)) if v.get(f) != was.get(f)]
        elif v != was:
            out.append(k)
    return out or ["a field the record does not carry"]


def run_unit(unit, session, mounts_for, cpuset=None, timeout=3600, run=execute, idle=gpu.wait_for_idle_gpu,
             provenance=None, served=None, gpu_index="0", port=DEFAULT_PORT, overrides=None, metrics_set="gh100",
             redo=False):
    """Runs one unit unless its status says done and its specification matches (or redo: a finished unit flagged, set
    aside like a failed one). mounts_for(out_dir) -> Mounts. served(unit, cmds, out) runs an L2 unit
    (anatomy.served.run_served with the client's settings bound). Returns (ok, seconds). The idle gate comes first,
    under the GPU's lock: a GPU that never idles raises (servebench.gpu.GPUBusy) before the unit writes anything, so
    a retry after clearing it leaves no attempt behind."""
    out = unit_dir(session, unit)
    cmds = commands(unit, mounts_for(out), cpuset, port=port, gpu=gpu_index, overrides=overrides,
                    metrics_set=metrics_set)
    rendered = {k: render(v) if isinstance(v, list) else
                {b: render(x) for b, x in v.items()} if k == "servers" else v for k, v in cmds.items()}
    revision, provenance = env.model_revision(unit.model), provenance or {}
    spec = run_spec(unit, rendered, revision, provenance)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()
    if is_done(out) and not redo:
        meta = read_meta(out) or {}
        if "run_fingerprint" not in meta:  # finished before its run was recorded: nothing says it was this run
            raise ResumeMismatch(f"{unit.key}: finished before its run was recorded (no run_spec in meta.json): run "
                                 "what is left under a new session id")
        if meta["run_fingerprint"] != fingerprint:
            raise ResumeMismatch(f"{unit.key}: finished, but its run differs from this one in "
                                 f"{', '.join(resume_differences(meta, spec))}: run what is left under a new session "
                                 "id, or --redo with this unit in reruns.csv to measure "
                                 "it again")
        return True, 0.0
    idle()
    set_aside(out)
    out.mkdir(parents=True, exist_ok=True)
    write_meta(out, unit, commands=rendered, model_revision=revision, run_fingerprint=fingerprint,
               run_spec=spec, **provenance)
    return _run_locked(unit, out, cmds, run, timeout, served)


def _run_locked(unit, out, cmds, run, timeout, served):
    t0, cpu = time.monotonic(), cgroup.cpu_stat()
    with clocks.ClockLog(out / "clocks.csv"):
        if "run" in cmds:
            # a hostload unit's spinners, for the whole run: inside its container once it runs, or the grid's own
            with containers.Contention(cmds.get("contend"), None if cmds.get("runtime") == "native" else unit.name) \
                    as spin:
                code = run(cmds["run"], out / "container.log", timeout, marker=cmds.get("marker"),
                           env_dir=cmds.get("env_dir"))
            error = None if code == 0 else f"exit {code}"
            if "contend" in cmds:  # loops that never ran tested nothing: the unit failed
                update_meta(out, contention=spin.record())
                error = error or spin.problem()
            error = error or batch_errors(out / "driver.jsonl") or missing_traces(unit, out)
        else:
            if served is None:
                raise RuntimeError("an L2 unit needs a served runner (anatomy.served)")
            # its spinners once the server is up, noted by it. A release that asks for it (llama.cpp:
            # server_per_batch) gets a fresh server for each batch, sized for it, inside the one unit and its deadline
            parts = [(replace(unit, batches=tuple(int(b) for b in key.split(","))), {**cmds, "server": argv})
                     for key, argv in cmds["servers"].items()]
            errors = [e for i, (part, part_cmds) in enumerate(parts)
                      if (e := served(part, part_cmds, out, probe=i == len(parts) - 1,
                                      timeout=None if timeout is None else
                                      max(timeout - (time.monotonic() - t0), 0.0)))]
            error = "; ".join(errors) or None
    # a runner loads and measures in one process with no waves to bracket: its whole run's quota periods, if any
    cpu = cgroup.since(cpu) if unit.level == "L0" else {}
    write_status(out, error is None, time.monotonic() - t0, error, **({"cpu": cpu} if cpu else {}))
    return error is None, time.monotonic() - t0


ATTEMPT = ".attempt-"


def set_aside(out):
    """A unit that is not done starts in an empty directory: a failed attempt's files move to <dir>.attempt-<n>, kept
    for provenance and skipped by anatomy.tidy. Rerunning into the old directory appended to driver.jsonl (its batch
    errors failed every resume) and numbered new traces after the old ones, so tidy paired batches with the wrong
    traces."""
    if not out.exists() or not any(out.iterdir()):
        return None
    n = 1 + sum(1 for p in out.parent.glob(f"{out.name}{ATTEMPT}*"))
    dest = out.with_name(f"{out.name}{ATTEMPT}{n}")
    out.rename(dest)
    return dest


def abandon(unit, session, error):
    """A unit that raised (the idle gate gave up, a command could not be built): its status says so, so the grid goes
    on to the next unit and a resumed session retries this one, and nothing it started stays on the GPU: its
    container, or natively the processes carrying its marker."""
    out = unit_dir(session, unit)
    out.mkdir(parents=True, exist_ok=True)
    write_status(out, False, 0.0, error)
    try:
        containers.remove_unit(unit.name)
    except containers.DockerHung:  # the unit is already counted as raised: the grid gives up after GIVE_UP_AFTER
        pass


def narrow_to_reruns(units, path, session, round_=1):
    """The units reruns.csv (anatomy.tidy) lists for `session`, with the session's sentinels: what --redo runs, every
    other unit left alone (on a fresh box nothing is "done", so the whole session would rerun). The sentinels run again
    as their own units (blocks sentinel-redo<round>-start/-end/-unlocked, the position last), so a redo in the same
    session measures them anew instead of finding the first run's done."""
    import csv

    with open(path) as f:
        rows = list(csv.DictReader(f))
    # noise and sentinel units are never rerun: they measure spread and drift, and running them again until they pass
    # would understate both
    listed = {r["unit"] for r in rows if r["session"] == session and not r["unit"].startswith(ALWAYS_RUN)}
    picked = [replace(u, block=u.block.replace("sentinel-", f"sentinel-redo{round_}-", 1))
              if is_sentinel(u.block) else u
              for u in units if u.key in listed or is_sentinel(u.block)]
    if not any(u.key in listed for u in picked):
        raise SystemExit(f"--redo {path}: no unit of session {session} is listed there (its sessions: "
                         f"{sorted({r['session'] for r in rows})})")
    return picked, listed


def redo_round(session):
    """1 + the redo rounds a session already ran (its sentinel-redo<n>-start directories)."""
    done = [p.name for p in (results.RAW_ROOT / session).glob("sentinel-redo*-start")]
    return 1 + max((int(n.removeprefix("sentinel-redo").split("-")[0]) for n in done), default=0)


def missing_traces(unit, out):
    """A profiled L1 unit must leave one export per profile window it opened (nsys numbers them in order, and tidy
    pairs them by that order alone), else the unit failed: it is rerun, never half-read."""
    if unit.capture not in KINDS or not (out / "driver.jsonl").exists():
        return None
    opened = sum(1 for r in read_jsonl(out / "driver.jsonl") if r.get("kind") == "profile")
    exported = len(list(out.glob(f"{unit.capture}.*.sqlite")))
    return None if opened == exported else f"{opened} profile windows but {exported} trace exports"


def batch_errors(path):
    """The driver isolates a failing batch and exits 0, so its unit's status must read the batch errors it recorded
    (in validation every TRT-LLM batch once failed while the unit reported ok). A batch skipped for KV capacity is not
    an error. None when every batch ran."""
    bad = [(r.get("batch"), str(r.get("error"))[:200]) for r in read_jsonl(path) if r.get("kind") == "error"]
    if not bad:
        return None
    return f"errors in batches {[b for b, _ in bad]}: {bad[0][1]}"


def default_mounts(out):
    """The box's mounts for a unit writing to out: the HF cache, the engines' compile caches and the models under the
    home directory, Nsight from DA_NSYS (provision*.sh writes it)."""
    home = Path("~").expanduser()
    return Mounts(repo=ROOT, out=out, hf=env.hf_home(), cache=home / "decode-anatomy-cache", models=home / "models",
                  nsys=Path(os.environ["DA_NSYS"]) if "DA_NSYS" in os.environ else None)


GETTABLE = ("models", "gpu", "clock_mhz", "versions", "validation", "images")


def get_value(grid, key, units, out=sys.stdout):
    """One grid value for the shell scripts (--get), so each has one definition: models (every model the selected units
    name, in the grid's order), gpu, clock_mhz, versions, validation (true/false), images (engine, version, image tag
    and the digest the runs pin, tab-separated, one line per engine of the version set)."""
    if key == "models":
        lines = list(dict.fromkeys(u.model for u in units))
    elif key == "images":
        lines = []
        for engine, version in grid["versions_map"].items():
            r = Release.load(engine, str(version))
            lines.append("\t".join((engine, r.version, r.image, r.digest_pinned_image)))
    elif key == "validation":
        lines = ["true" if grid.get("validation") else "false"]
    else:
        value = grid.get(key)
        lines = ["" if value is None else str(value)]
    print("\n".join(lines), file=out)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("grid", type=Path)
    p.add_argument("--session", help="S1..S3: that session's blocks, with sentinels first and last")
    p.add_argument("--block", action="append", help="only these blocks (no sentinels)")
    p.add_argument("--models", nargs="+", help="only these models' units, the session's sentinels kept: a lost host's "
                                               "unfinished models, rerun under a new session id")
    p.add_argument("--engines", nargs="+", help="only these engines' units (a validation, or one engine's rerun)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--run", action="store_true")
    p.add_argument("--session-id", help="the results directory name (default: <session>-<date>)")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--no-gpu", action="store_true",
                   help="CPU rehearsal: no --gpus, no card gate (llama.cpp L2 only; see configs/grid/rehearsal.yaml)")
    p.add_argument("--aiperf", default=str(ROOT / ".venv-aiperf" / "bin" / "aiperf"))
    p.add_argument("--redo", type=Path, help="reruns.csv (anatomy.tidy): this session's units listed there run again "
                                             "though finished, their earlier attempt set aside")
    p.add_argument("--redo-session", help="the session whose reruns.csv rows to run (default: --session-id): reruns on "
                                          "another host go under a new session id")
    g.add_argument("--progress", action="store_true",
                   help="per model, how many of the session's units (--session-id's directory) are done, and the "
                        "--models argument that would finish it on a new host")
    g.add_argument("--get", choices=sorted(GETTABLE),
                   help="print one grid value for the shell scripts (one definition of each): models (every model the "
                        "selected units name), gpu, clock_mhz, versions, validation, or images (engine, version, "
                        "image, pinned digest per line)")
    a = p.parse_args(argv)
    grid = load(a.grid)
    if a.session and not a.block:
        units = session_units(grid, a.session, models=a.models)
    else:
        units = expand(grid, blocks=set(a.block) if a.block else None, models=a.models)
    if a.engines:
        unknown = set(a.engines) - {u.engine for u in units}
        if unknown:
            raise SystemExit(f"--engines {' '.join(sorted(unknown))}: no unit of the grid runs on it")
        units = [u for u in units if u.engine in a.engines]
    if a.get:
        return get_value(grid, a.get, units)
    missing = [m for m in a.models or () if m not in {u.model for u in units}]
    if missing:  # typed by hand after a lost host: a misspelt model would silently never run
        raise SystemExit(f"--models: no unit of this grid has {missing}")
    session = a.session_id or f"{a.session or 'adhoc'}-{time.strftime('%Y%m%d')}"
    redo = set()
    if a.redo:  # before anything touches the GPU: a reruns.csv for another session stops here
        units, redo = narrow_to_reruns(units, a.redo, a.redo_session or session, redo_round(session))
        redo -= redone(units, session, a.redo.stat().st_mtime)
    if a.progress:
        return progress(units, session)
    if a.list:
        for u in units:
            print(f"{u.block:14} {u.key}  B={list(u.batches)}")
        print(f"{len(units)} units, {sum(len(u.batches) for u in units)} cells", file=sys.stderr)
        return 0
    engine_cpus, client_cpus = cpusets(env.gpu_local_cpus(), quota=cgroup.cpu_quota())
    gpu_index = None if a.no_gpu else "0"
    if a.dry_run:
        for u in units:
            cmds = commands(u, default_mounts(Path("/OUT") / u.key), engine_cpus, port=grid.get("port", DEFAULT_PORT),
                            gpu=gpu_index, overrides=grid.get("overrides"),
                            metrics_set=metric_set(grid["gpu"]) or "gh100")
            print(f"# {u.block} {u.key}  B={list(u.batches)}")
            if "run" in cmds:
                print(f"run: {render(cmds['run'])}")
            for batches, argv in (cmds.get("servers") or {}).items():
                print(f"server B={batches}: {render(argv)}")
        return 0
    if any(u.capture in KINDS for u in units) and not os.environ.get("DA_NSYS"):
        # every profiled command calls /opt/nsys/bin/nsys, which only a mounted (or, natively, translated) install has
        raise SystemExit("profiled units need DA_NSYS (the Nsight Systems install; provision*.sh writes it to "
                         "~/.decode-anatomy.env)")
    if off_pin := env.pin_problems(sorted({u.model for u in units})):
        # every engine loads what refs/main names: a cell on another snapshot is rejected (revision_problem) after the
        # rental has paid for it, so a study grid stops here, before the first unit
        if not grid.get("validation"):
            raise SystemExit("models off their pins:\n  " + "\n  ".join(off_pin))
        print("warning: models off their pins (validation grid, going on):\n  " + "\n  ".join(off_pin), flush=True)
    if gpu.IDLE_MODE != "empty" and not grid.get("validation"):
        # the activity gate starts units beside resident processes: validation on a shared box only, never the study
        raise SystemExit(f"DA_IDLE_MODE={gpu.IDLE_MODE} is for validation on a shared GPU: {a.grid} does not set "
                         "validation: true")
    clock_mhz = None if a.no_gpu else grid.get("clock_mhz")
    if not a.no_gpu:
        gpu.check_card(env.gpu_query("name").get("name"), grid["gpu"])
    locked = bool(clock_mhz) and gpu.lock_clocks(clock_mhz)
    if clock_mhz and not locked and not grid.get("validation") and os.environ.get("DA_ALLOW_UNLOCKED") != "1":
        # S1's ruler and every other session ran locked: one session unlocked would differ in what is measured
        raise SystemExit(f"the SM clock could not be locked at {clock_mhz} MHz (sudo -n nvidia-smi -lgc refused): the "
                         "study's sessions run locked. Choose another VM, or set DA_ALLOW_UNLOCKED=1 for every session "
                         "alike")
    # tmux kill-session, kill <pid>: unwind, so the finally blocks run (clocks unlocked, spinners and servers ended)
    handlers = {sig: signal.signal(sig, lambda signum, _frame: sys.exit(128 + signum))
                for sig in (signal.SIGTERM, signal.SIGHUP)}
    host = env.host()
    prov = {"host": host, "git": env.git_sha(ROOT), "grid": str(a.grid), "versions": grid["versions"],
            "idle_mode": gpu.IDLE_MODE,
            "runtime": containers.RUNTIME, "validation": bool(grid.get("validation")),
            "client_cpuset": client_cpus, "aiperf": a.aiperf,
            "sm_clock_lock_mhz": clock_mhz if locked else None}
    print(f"SM clock {'locked at ' + str(clock_mhz) + ' MHz' if locked else 'not locked'}", flush=True)
    from anatomy.served import run_served
    mset = metric_set(grid["gpu"]) or "gh100"
    served = partial(run_served, aiperf=a.aiperf, client_cpuset=client_cpus, metrics_set=mset)
    idle = (lambda: None) if a.no_gpu else gpu.wait_for_idle_gpu
    syncing = os.environ.get("DA_SYNC") == "1" and bool(os.environ.get("DATASET"))
    unsynced, sync_failures = [], 0
    failed, raised, stalled, stalled_units, ended = [], 0, set(), 0, None
    hung, skip = {}, set()  # an engine timing out on a model, unit after unit: (engine, model) -> timeouts in a row
    try:
        for i, u in enumerate(units, 1):
            print(f"[{i}/{len(units)}] {u.key} B={list(u.batches)}", flush=True)
            grid_unit = not u.block.startswith(ALWAYS_RUN)  # those always run: drift and noise
            if grid_unit and (u.engine, u.model) in skip and not is_done(unit_dir(session, u)):
                # failed, not done: a resumed session runs it again
                out = unit_dir(session, u)
                out.mkdir(parents=True, exist_ok=True)
                write_status(out, False, 0.0, f"skipped: {u.engine} ran out of time {HUNG_AFTER} units in a row on "
                                              f"{model_name(u.model)}")
                failed.append(u.key)
                print("    skipped (its engine hangs on this model)", flush=True)
                continue
            unlocked = is_sentinel(u.block) and u.block.endswith("-unlocked") and locked
            if unlocked:
                gpu.unlock_clocks()
            run = partial(run_unit, u, session, default_mounts, engine_cpus, a.timeout,
                          provenance={**prov, "sm_clock_lock_mhz": None if unlocked else prov["sm_clock_lock_mhz"]},
                          served=served, gpu_index=gpu_index, port=grid.get("port", DEFAULT_PORT),
                          overrides=grid.get("overrides"), metrics_set=mset, idle=idle,
                          redo=u.key in redo)
            try:
                try:
                    ok, secs = run()
                except gpu.GPUBusy as e:  # whatever holds the GPU: ours, left by a killed run; cleared, tried once more
                    print(f"    {e}; clearing what our units left and trying once more", flush=True)
                    if gpu.IDLE_MODE == "empty":  # a box that is the study's alone: nothing on it is anyone else's
                        containers.clear()
                    ok, secs = run()
                raised = 0
            except ResumeMismatch as e:
                print(str(e), file=sys.stderr)
                raise SystemExit(2) from e  # preserve the attempt; session.sh must not run QC on a partial grid
            except Exception as e:  # noqa: BLE001  one unit's failure never ends an unattended session
                abandon(u, session, repr(e))
                ok, secs, raised, last = False, 0.0, raised + 1, repr(e)
                print(f"    raised {last}", flush=True)
            if unlocked:
                gpu.lock_clocks(clock_mhz)
            print(f"    {'ok' if ok else 'FAILED'} in {secs:.0f} s", flush=True)
            if not ok:
                failed.append(u.key)
            # a run of timeouts ends the session only when it spans engines: one engine hanging on one model times
            # out unit after unit (its captures run back to back), and the other engines and closing sentinels must
            # still run; a broken box shows in `raised` (GPUBusy, DockerHung) as well
            if timed_out(unit_dir(session, u)):
                stalled.add(u.engine)
                stalled_units += 1
            else:
                stalled, stalled_units = set(), 0
            if grid_unit:  # HUNG_AFTER timeouts of one engine on one model: an hour a unit is not spent on the rest
                hung[u.engine, u.model] = hung.get((u.engine, u.model), 0) + 1 if timed_out(unit_dir(session, u)) else 0
                if hung[u.engine, u.model] >= HUNG_AFTER:
                    skip.add((u.engine, u.model))
            if syncing and secs > 0:  # raw data off the rental, per unit that ran; a failed upload is tried again
                unsynced.append(u.key)
            due = i % SYNC_EVERY == 0 or i == len(units)  # i counts from 1: every SYNC_EVERY-th unit, and the last
            if syncing and unsynced and due and (sync_failures == 0 or i % SYNC_RETRY_EVERY == 0):
                while unsynced and sync(unsynced[0], session, grid["versions"]):
                    unsynced.pop(0)
                sync_failures = sync_failures + 1 if unsynced else 0
                if sync_failures >= SYNC_GIVE_UP:
                    syncing = False
                    print(f"    {sync_failures} upload rounds in a row failed: no more per-unit syncs this session "
                          "(session.sh syncs it all at the end)", flush=True)
            if raised >= GIVE_UP_AFTER or (stalled_units >= GIVE_UP_AFTER and len(stalled) > 1):  # the box is broken
                ended = (f"{raised} units in a row raised (the last: {last})" if raised >= GIVE_UP_AFTER else
                         f"{stalled_units} units of {len(stalled)} engines in a row ran out of time") + \
                    "; the session ends here"
                print(f"grid: {ended}", flush=True)
                break
    finally:
        if locked:
            gpu.unlock_clocks()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps({"session": session, "units": len(units), "failed": failed, "ended_early": ended}, indent=1))
    return 2 if ended else 1 if failed else 0  # 2: the session ended early (scripts/session.sh writes no QC list)


def progress(units, session):
    """Per model, the session's grid units done in its directory (sentinels and noise apart: a new host runs its own),
    and the --models argument that would finish the session on a new host."""
    by = {}
    for u in units:
        if not u.block.startswith(ALWAYS_RUN):
            by.setdefault(u.model, []).append(is_done(unit_dir(session, u)))
    for model, done in by.items():
        print(f"{model:24} {sum(done)}/{len(done)} units done")
    left = [m for m, done in by.items() if not all(done)]
    print(f"--models {' '.join(left)}" if left else "every model done")
    return 0


def redone(units, session, since):
    """The keys of units a redo already ran again after reruns.csv was written (a finished unit whose meta.json is
    newer than it): a resumed --redo leaves them done instead of paying for them twice."""
    out = set()
    for u in units:
        d = unit_dir(session, u)
        if is_done(d) and (read_meta(d) or {}).get("written", 0) > since:
            out.add(u.key)
    return out


def timed_out(out):
    """Whether a unit's status says it ran out of time (exit 124, or an L2 unit's deadline)."""
    error = (read_status(out) or {}).get("error") or ""
    return error.startswith("exit 124") or "deadline passed" in error


def sync(key, session, versions):
    """One unit's raw data to the HF dataset (scripts/sync_raw.sh), bounded in time and ended whole on a timeout
    (`hf upload` with it). True when it worked; the grid keeps a failed unit's key and tries it again later
    (SYNC_RETRY_EVERY), and gives per-unit syncing up after SYNC_GIVE_UP failed rounds in a row (a slow or
    rate-limiting Hub must not hold the GPU idle after every unit); session.sh uploads the whole session at the end."""
    try:
        code = containers.run(["scripts/sync_raw.sh", key], cwd=ROOT, timeout=SYNC_TIMEOUT_S,
                              env={**os.environ, "SESSION": session, "VERSIONS": versions}).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    if code:
        print(f"    sync of {key} failed ({code}); it is tried again later", flush=True)
    return not code


if __name__ == "__main__":
    sys.exit(main())
