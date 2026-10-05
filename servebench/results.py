"""Where a unit's files live, and what makes a unit done.

A unit is one container run: one (engine, version, arm, level, model, capture) with its batch sweep inside. Its cells
are the (unit, batch) pairs. Raw files are written outside the clone (a run must never change the tree it runs from)
under RAW_ROOT (and, with DATASET set, synced to a dataset):

    <RAW_ROOT>/<session>/<engine>-<version>/<arm>/<level>/<model>/<capture>/

results/README.md (the dataset card) lists every file a unit writes. status.json is written last: a unit with an ok
status is done and is skipped when the grid resumes; anything else is rerun.
"""
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = "decode-anatomy/unit/1"
RAW_ROOT = Path(os.environ.get("DA_RAW_ROOT", "~/decode-anatomy-runs")).expanduser()


def model_name(model):
    return str(model).rstrip("/").rsplit("/", 1)[-1].removesuffix(".gguf")


# blocks whose units are keyed by their block: sentinels and noise processes run more than once a session, and a
# probe-<name> block measures a unit the study's own blocks also measure, under other settings (llamacpp-slots)
REPEATED = ("sentinel-", "noise-", "probe-")
ALWAYS_RUN = ("sentinel-", "noise-")  # never narrowed away, never rerun, never a cell: they measure drift and spread


def keyed_by_block(block):
    """Whether a block's units carry the block in their key (REPEATED): never cells of the study's grid."""
    return (block or "").startswith(REPEATED)


def is_sentinel(block):
    return (block or "").startswith("sentinel-")


def is_noise(block):
    return (block or "").startswith("noise-")


def context(prompt_len, gen_len):
    """The context's part of a unit's key and a cell's id: nothing at the study's 896 + 288, else -p<prompt>g<gen>."""
    return "" if (prompt_len, gen_len) == (896, 288) else f"-p{prompt_len}g{gen_len}"


def cell_id(engine, version, arm, level, model, batch, prompt_len=896, gen_len=288):
    """A cell's id: its units' key without the capture, with its batch (vllm-0.30.0/defaults/L1/Qwen3-8B/b1). Unique
    across releases, arms, levels and contexts; with its session, the dataset path of every capture that measured it
    (<version set>/<session>/<id minus the batch>/<capture>/)."""
    return f"{engine}-{version}/{arm}/{level}/{model_name(model)}{context(prompt_len, gen_len)}/b{batch}"


def row_cell_id(r):
    """A cell row's id (cell_id from its fields): what every pattern, principle, strip and page element names a cell
    by."""
    return cell_id(r["engine"], r.get("version"), r["arm"], r["level"], r["model"], r["batch"],
                   r.get("prompt_len", 896), r.get("gen_len", 288))


def resolve(name, roots):
    """The path a row's source (<session>/<path under it>) names, under the session root it names; None when no root
    is that session."""
    session, _, rest = (name or "").partition("/")
    root = next((Path(r) for r in roots if Path(r).name == session), None)
    return root / rest if root and rest else None


@dataclass(frozen=True)
class Unit:
    engine: str
    version: str
    arm: str
    level: str
    model: str
    capture: str
    batches: tuple
    prompt_len: int = 896
    gen_len: int = 288
    block: str = ""

    @property
    def key(self):
        """The unit's directory under its session. A sentinel or a noise process runs more than once per session (start,
        end, unlocked clocks; process 1, 2, 3), so its block is part of its key: otherwise the second run would find
        the first's status and skip."""
        prefix = f"{self.block}/" if self.block.startswith(REPEATED) else ""
        return (f"{prefix}{self.engine}-{self.version}/{self.arm}/{self.level}/{model_name(self.model)}"
                f"{context(self.prompt_len, self.gen_len)}/{self.capture}")

    @property
    def name(self):
        """A docker container name: unique per unit, legal characters only."""
        return "da-" + "".join(c if c.isalnum() or c in "-_." else "-" for c in self.key)

    def to_dict(self):
        return {**asdict(self), "batches": list(self.batches), "key": self.key}


def unit_dir(session, unit, root=None):
    return Path(root or RAW_ROOT) / session / unit.key


def write_json(path, obj):
    """A record file written whole or not at all: to <name>.tmp beside it, then renamed over it, so a process killed
    mid-write (a unit's deadline, a session unwound by a signal) never leaves a torn meta.json or status.json for
    the grid's resume or tidy to trip on."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path):
    """The record a unit's file holds, or None when the file does not exist. A file that is there but not JSON is an
    error with its path, never a default: the writes are atomic, so a torn record means something else is wrong."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not a JSON record ({e})") from None


def read_meta(unit_dir):
    return read_json(Path(unit_dir) / "meta.json")


def read_status(unit_dir):
    return read_json(Path(unit_dir) / "status.json")


def is_done(path):
    status = read_status(path)
    return status is not None and status.get("ok") is True


def write_meta(path, unit, **provenance):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "meta.json", {"schema": SCHEMA, "unit": unit.to_dict(), "written": time.time(), **provenance})


def update_meta(path, **fields):
    """meta.json with fields added after the run (what a hostload unit's contention took)."""
    write_json(Path(path) / "meta.json", {**(read_meta(path) or {}), **fields})


def write_status(path, ok, seconds, error=None, **extra):
    write_json(Path(path) / "status.json", {"ok": ok, "seconds": round(seconds, 1), "error": error, **extra})


def contention_for(contention, batch):
    """A hostload unit's contention record for one batch: a server per batch (llama.cpp) records one per server under
    its batches ("64", "64,256"); one record for the unit (every other engine, and the validation sessions) is
    every batch's. None without one."""
    if not contention:
        return None
    if "cpu_share" in contention:
        return contention
    for key, rec in contention.items():
        if str(batch) in key.split(","):
            return rec
    return None


def json_lines(text):
    """Every line of text that holds a JSON object; anything else (log lines around a runner's output, the torn last
    line of a process killed mid-write) is skipped."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_jsonl(path):
    p = Path(path)
    return json_lines(p.read_text(errors="replace")) if p.exists() else []
