"""The shell that runs the paid sessions, driven from pytest with stubs: scripts/session.sh (the grid, the box's own
reruns, the final upload and its exit codes), scripts/sync_raw.sh (nothing secret-looking ever uploaded) and
scripts/box.sh (the job it writes on the box, its exit code, and `wait`). Each runs in a scratch git repo or home,
its tools (.venv/bin/python, hf, ruler.sh, ssh, tmux) replaced by scripts that log what they were asked to do."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import gnu_timeout

ROOT = Path(__file__).resolve().parents[1]
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}


def stub(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(0o755)


def git_repo(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


PYTHON = """echo "python $*" >> "$CALLS"
case "$1 $2" in
  "-m anatomy.grid")
    case " $* " in
      *" --get versions "*) echo 2026-09 ;;
      *" --get validation "*) echo "${VALIDATION:-false}" ;;
      *) exit "${GRID_EXIT:-0}" ;;
    esac ;;
  "-m anatomy.tidy")
    [ "${TIDY_EXIT:-0}" = 0 ] || exit "$TIDY_EXIT"
    out=""; prev=""; for a in "$@"; do [ "$prev" = "--out" ] && out=$a; prev=$a; done
    mkdir -p "$out"; printf 'session,unit,batches,ranges\\n%s' "${RERUNS:-}" > "$out/reruns.csv" ;;
  "-m servebench.nsys.scrub") exit "${SCRUB_EXIT:-0}" ;;
esac
"""


@pytest.fixture
def repo(tmp_path):
    """A scratch repo holding the real session.sh and sync_raw.sh, its python, ruler and hf replaced."""
    r = git_repo(tmp_path / "repo")
    (r / "scripts").mkdir()
    for name in ("session.sh", "sync_raw.sh"):
        shutil.copy(ROOT / "scripts" / name, r / "scripts" / name)
    stub(r / ".venv" / "bin" / "python", PYTHON)
    stub(r / ".venv" / "bin" / "hf", 'echo "hf $*" >> "$CALLS"\n')
    # the ruler stub writes the fingerprint the real one does first (unless RULER_WRITES=0)
    stub(r / "scripts" / "ruler.sh", 'echo "ruler $*" >> "$CALLS"\n'
                                     '[ "${RULER_WRITES:-1}" = 1 ] || exit 1\n'
                                     'mkdir -p "$DA_RAW_ROOT/$SESSION/ruler"\n'
                                     'echo "{}" > "$DA_RAW_ROOT/$SESSION/ruler/bandwidth.json"\n'
                                     'echo "{}" > "$DA_RAW_ROOT/$SESSION/ruler/launch_latency.json"\n')
    (r / "results").mkdir()
    (r / "results" / "README.md").write_text("card\n")
    return r


def session(repo, tmp_path, *args, ruler_done=False, **env):
    raw = tmp_path / "runs"
    if ruler_done:
        (raw / "S2-x" / "ruler").mkdir(parents=True, exist_ok=True)
        (raw / "S2-x" / "ruler" / "ruler_report.json").write_text("{}")
    calls = tmp_path / "calls"
    calls.write_text("")
    full = {**os.environ, "SESSION": "S2-x", "DA_RAW_ROOT": str(raw), "CALLS": str(calls), "DATASET": "u/d", **env}
    full = {k: v for k, v in full.items() if v is not None}
    code = subprocess.run(["bash", str(repo / "scripts" / "session.sh"), "S2", *args], cwd=repo, env=full,
                          capture_output=True, text=True).returncode
    return code, calls.read_text().splitlines()


def grid_calls(calls):  # the grid runs, not the script's --get queries of the grid file
    return [c for c in calls if c.startswith("python -m anatomy.grid") and "--get" not in c]


@gnu_timeout
def test_a_session_runs_its_grid_and_uploads_it(repo, tmp_path):
    code, calls = session(repo, tmp_path, "--timeout", "60")
    assert code == 0 and len(grid_calls(calls)) == 1 and grid_calls(calls)[0].endswith("--run --timeout 60")
    assert "ruler " in calls  # no ruler report yet: the session's own full ruler first, never another card's
    uploads = [c for c in calls if c.startswith("hf upload u/d") and "--include" in c]  # the raw data's uploads
    assert uploads and all("--exclude *.nsys-rep" in c and "--include *.sqlite" in c for c in uploads)  # known types
    for c in uploads:  # one --include per pattern: the pinned CLI refuses a second pattern as an extra argument
        words = c.split()
        assert all(words[words.index(w) - 1] in ("--include", "--exclude") for w in words if "*" in w), c


@gnu_timeout
def test_a_session_with_its_ruler_done_runs_none(repo, tmp_path):
    _, calls = session(repo, tmp_path, ruler_done=True)
    assert not [c for c in calls if c.startswith("ruler")]


@gnu_timeout
def test_a_failed_final_upload_exits_3(repo, tmp_path):
    stub(repo / ".venv" / "bin" / "hf", 'echo "hf $*" >> "$CALLS"; exit 1\n')
    code, calls = session(repo, tmp_path)
    assert code == 3 and len([c for c in calls if c.startswith("hf upload") and "(whole session)" in c]) == 2


@gnu_timeout
def test_flagged_units_get_one_redo_round_on_the_same_box_with_the_same_arguments(repo, tmp_path):
    code, calls = session(repo, tmp_path, "--timeout", "60",
                          RERUNS="S2-x,vllm-0.30.0/defaults/L1/Qwen3-8B/timing,64,B=64: repeats 9%\n")
    grid = grid_calls(calls)
    assert code == 0 and len(grid) == 2 and "--redo" in grid[1] and grid[1].endswith("--timeout 60")


@gnu_timeout
def test_a_session_that_ended_early_writes_no_reruns_list(repo, tmp_path):
    """The grid exits 2 when the box broke under it: a list built then would cover half a session and a resume would
    reuse it, so QC waits for the resumed session to reach its end (its own run builds the list over every unit)."""
    code, calls = session(repo, tmp_path, GRID_EXIT="2", RERUNS="S2-x,vllm/x/timing,64,x\n")
    assert code == 2 and len(grid_calls(calls)) == 1 and not [c for c in calls if "anatomy.tidy" in c]


@gnu_timeout
def test_a_failed_reruns_check_leaves_the_session_as_it_is(repo, tmp_path):
    code, calls = session(repo, tmp_path, TIDY_EXIT="2", RERUNS="S2-x,vllm/x/timing,64,x\n")
    assert code == 0 and len(grid_calls(calls)) == 1


@gnu_timeout
def test_a_validation_grid_gets_no_redo_round(repo, tmp_path):
    """The smoke grid and the slot probes are validation data: their flagged units are never rerun, and no reruns
    list is built for them (an S1 probe would otherwise --redo S1's own flagged units through the probe grid)."""
    code, calls = session(repo, tmp_path, VALIDATION="true", RERUNS="S2-x,vllm/x/timing,64,x\n")
    assert code == 0 and len(grid_calls(calls)) == 1 and not [c for c in calls if "anatomy.tidy" in c]


@gnu_timeout
def test_a_resumed_session_reuses_its_reruns_list(repo, tmp_path):
    """One redo round per session: a session resumed after an interruption runs the round's list as written, never a
    regenerated one in which a unit still above range_ok gets another chance to pass."""
    qc = tmp_path / "runs" / "S2-x" / "qc"
    qc.mkdir(parents=True)
    (qc / "reruns.csv").write_text("session,unit,batches,ranges\nS2-x,vllm-0.30.0/defaults/L1/Qwen3-8B/timing,64,x\n")
    code, calls = session(repo, tmp_path, TIDY_EXIT="2")  # tidy would fail if it were asked for a new list
    grid = grid_calls(calls)
    assert code == 0 and len(grid) == 2 and "--redo" in grid[1] and not [c for c in calls if "anatomy.tidy" in c]


def test_without_a_dataset_nothing_is_uploaded(repo, tmp_path):
    code, calls = session(repo, tmp_path, DATASET=None)
    assert code == 0 and not [c for c in calls if c.startswith("hf")]


def test_sync_refuses_to_upload_while_anything_secret_looking_remains(repo, tmp_path):
    calls = tmp_path / "calls"
    calls.write_text("")
    env = {**os.environ, "SESSION": "S2-x", "DATASET": "u/d", "VERSIONS": "2026-09", "DA_RAW_ROOT": str(tmp_path),
           "CALLS": str(calls), "SCRUB_EXIT": "1"}
    code = subprocess.run(["bash", str(repo / "scripts" / "sync_raw.sh")], cwd=repo, env=env,
                          capture_output=True, text=True).returncode
    assert code == 4 and not [c for c in calls.read_text().splitlines() if c.startswith("hf")]


FAKE_SSH = """while [ "$1" = "-o" ]; do shift 2; done
shift
cd "$FAKE_HOME" && HOME=$FAKE_HOME exec bash -c "$*"
"""
FAKE_TMUX = 'exec bash -c "${@: -1}"\n'  # the job runs now, start to end, instead of in a detached session


def test_box_runs_a_job_records_its_exit_code_and_wait_returns_it(tmp_path):
    origin = git_repo(tmp_path / "origin")
    stub(origin / "scripts" / "ruler.sh", 'echo "ruler on the box: SESSION=$SESSION GRID=$GRID"; exit 7\n')
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-qm", "box"], check=True, env={**os.environ, **GIT_ENV})
    home = tmp_path / "home"
    home.mkdir()
    subprocess.run(["git", "clone", "-q", str(origin), str(home / "decode-anatomy")], check=True)
    stub(tmp_path / "bin" / "ssh", FAKE_SSH)
    stub(tmp_path / "bin" / "tmux", FAKE_TMUX)
    env = {**os.environ, "BOX": "box", "FAKE_HOME": str(home), "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
           "DA_WAIT_S": "0", **GIT_ENV}
    run = subprocess.run(["bash", str(ROOT / "scripts" / "box.sh"), "ruler", "S1-x"], env=env, capture_output=True,
                         text=True)
    assert run.returncode == 0 and "ruler started in tmux da-ruler" in run.stdout
    job = (home / "decode-anatomy-job.sh").read_text()
    assert "SESSION=S1-x" in job and "scripts/ruler.sh" in job and "exit.code" in job
    box = home / "decode-anatomy"
    assert (box / "exit.code").read_text().strip() == "7" and not (box / "busy").exists()
    assert "SESSION=S1-x GRID=configs/grid/v1.yaml" in (box / "job.log").read_text()
    waited = subprocess.run(["bash", str(ROOT / "scripts" / "box.sh"), "wait"], env=env, capture_output=True,
                            text=True)
    assert waited.returncode == 7                                       # wait ends with the job's own exit code


def test_models_are_fetched_at_their_pin_and_refs_main_points_at_it(tmp_path):
    """A download by commit writes no refs/main, and every engine resolves main offline: the script writes it, so a
    box with an older download loads the pin, never what it had."""
    import sys
    r = git_repo(tmp_path / "repo")
    (r / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "fetch_models.sh", r / "scripts" / "fetch_models.sh")
    (r / ".venv" / "bin").mkdir(parents=True)
    (r / ".venv" / "bin" / "python").symlink_to(sys.executable)
    hub = tmp_path / "hf" / "hub"
    # the stub hf: the snapshot of the commit asked for, as huggingface_hub lays it out (no ref written)
    stub(r / ".venv" / "bin" / "hf", 'echo "hf $*" >> "$CALLS"; repo=$2; rev=$4\n'
                                     'd="$HF_HOME/hub/models--${repo//\\//--}"; mkdir -p "$d/snapshots/$rev"\n')
    calls = tmp_path / "calls"
    calls.write_text("")
    old = hub / "models--Qwen--Qwen3-0.6B" / "refs"
    old.mkdir(parents=True)
    (old / "main").write_text("0" * 40)  # an older download's ref
    env = {**os.environ, "HF_HOME": str(tmp_path / "hf"), "DA_RAW_ROOT": str(tmp_path / "runs"),
           "PYTHONPATH": str(ROOT), "CALLS": str(calls)}
    done = subprocess.run(["bash", str(r / "scripts" / "fetch_models.sh"), "Qwen/Qwen3-0.6B"], cwd=r, env=env,
                          capture_output=True, text=True)
    from servebench.env import pinned
    assert done.returncode == 0, done.stderr
    # every pattern behind its own --exclude: the CLI takes one value per flag, and a bare second pattern is read as
    # a file name to download, which left a snapshot without its .gitattributes, LICENSE and README on this laptop
    # and AIPerf's offline tokenizer loader refusing it as incomplete
    (call,) = [c for c in calls.read_text().splitlines() if c.startswith("hf download")]
    words = call.split()
    patterns = [w for w in words if "*" in w]
    assert patterns and all(words[words.index(w) - 1] == "--exclude" for w in patterns), call
    assert (old / "main").read_text() == pinned("Qwen/Qwen3-0.6B")
    unpinned = subprocess.run(["bash", str(r / "scripts" / "fetch_models.sh"), "org/unknown"], cwd=r, env=env,
                              capture_output=True, text=True)
    assert unpinned.returncode == 2 and "no pin" in unpinned.stderr


@gnu_timeout
def test_a_session_whose_ruler_left_nothing_does_not_run_its_grid(repo, tmp_path):
    """Two ruler attempts and no bandwidth or launch latency for this card: no cell could be read against it."""
    code, calls = session(repo, tmp_path, RULER_WRITES="0")
    assert code == 2 and [c for c in calls if c.startswith("ruler")] == ["ruler ", "ruler "] and not grid_calls(calls)


def provision(tmp_path, card, driver="580.126.09", clock_ok=True):
    """scripts/provision.sh in a scratch home with every tool stubbed: nvidia-smi answers with the card and driver
    given, sudo passes its command through, docker and uv do nothing, the pinned Nsight is already in place, and the
    model and GGUF scripts only log. Returns (exit code, the calls logged, the session's host directory)."""
    r = git_repo(tmp_path / "repo")
    (r / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "provision.sh", r / "scripts" / "provision.sh")
    (r / "configs" / "grid").mkdir(parents=True)
    shutil.copy(ROOT / "configs" / "grid" / "v1.yaml", r / "configs" / "grid" / "v1.yaml")
    for e, v in (("llamacpp", "v0.5.0"), ("vllm", "0.30.0")):
        (r / "configs" / "engines" / e / v).mkdir(parents=True)
        (r / "configs" / "engines" / e / v / "image.txt").write_text(f"{e}:tag\n{e}@sha256:0\n")
    calls = tmp_path / "calls"
    calls.write_text("")
    for name in ("fetch_models.sh", "make_gguf.sh", "capture_help.sh"):
        stub(r / "scripts" / name, f'echo "{name} $*" >> "$CALLS"\n')
    home = tmp_path / "home"
    nsys = home / "opt" / "nsight-systems" / "2026.5.1" / "bin"
    stub(nsys / "nsys", 'echo "nsys 2026.5.1"\n')
    bin_ = tmp_path / "bin"
    stub(bin_ / "nvidia-smi", f'echo "nvidia-smi $*" >> "$CALLS"\n'
                              f'case " $* " in\n'
                              f'  *"--query-gpu=name"*) echo "{card}" ;;\n'
                              f'  *"--query-gpu=driver_version"*) echo "{driver}" ;;\n'
                              f'  *" -lgc "*) exit {0 if clock_ok else 1} ;;\n'
                              f'esac\n')
    stub(bin_ / "sudo", '[ "$1" = -n ] && shift; echo "sudo $*" >> "$CALLS"; "$@"\n')
    stub(bin_ / "docker", 'echo "docker $*" >> "$CALLS"\n[ "$1" = version ] && echo "docker 29"\n'
                          'for arg in "$@"; do case "$arg" in *:/out) touch "${arg%:/out}/check.sqlite" ;; esac; done\n'
                          'exit 0\n')
    stub(bin_ / "uv", 'echo "uv $*" >> "$CALLS"\n')
    stub(r / ".venv" / "bin" / "python", 'echo "python $*" >> "$CALLS"\n')
    env = {**os.environ, "HOME": str(home), "SESSION": "S1-x", "DA_RAW_ROOT": str(tmp_path / "runs"),
           "CALLS": str(calls), "PATH": f"{bin_}:{os.environ['PATH']}"}
    env.pop("DATASET", None)
    done = subprocess.run(["bash", str(r / "scripts" / "provision.sh")], cwd=r, env=env, capture_output=True,
                          text=True)
    return done, calls.read_text().splitlines(), tmp_path / "runs" / "S1-x" / "host"


def test_provision_gates_on_the_grids_card_before_downloading_anything(tmp_path):
    done, calls, _ = provision(tmp_path, card="NVIDIA H100 PCIe")
    assert done.returncode == 1 and "NVIDIA H100 80GB HBM3" in done.stderr   # the card the grid names, from v1.yaml
    assert not [c for c in calls if c.startswith(("docker pull", "fetch_models", "make_gguf"))]


def test_provision_records_the_host_and_locks_the_grids_clock(tmp_path):
    done, calls, host = provision(tmp_path, card="NVIDIA H100 80GB HBM3")
    assert done.returncode == 0, done.stderr
    assert (host / "clock-lock.txt").read_text().strip() == "locked"
    assert any(c.startswith("nvidia-smi -i 0 -lgc 1830,1830") for c in calls)   # the grid's clock_mhz, not a literal
    assert (host / "nsys-version.txt").exists() and (host / "images.txt").exists()
    assert any(c.startswith("python -m servebench.nsys.scrub") for c in calls)
    assert not list(host.glob(".scrub-check.*"))
    assert [c for c in calls if c.startswith("fetch_models.sh")] and [c for c in calls if c.startswith("make_gguf.sh")]
    env_file = (tmp_path / "home" / ".decode-anatomy.env").read_text()
    assert "DA_NSYS=" in env_file and "nsight-systems/2026.5.1" in env_file


def test_provision_records_a_refused_clock_lock_and_goes_on(tmp_path):
    done, _, host = provision(tmp_path, card="NVIDIA H100 80GB HBM3", clock_ok=False)
    assert done.returncode == 0 and (host / "clock-lock.txt").read_text().strip() == "refused"
