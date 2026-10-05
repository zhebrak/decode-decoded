"""One engine release as the grid sees it: configs/engines/<engine>/<version>/arms.yaml, merged into the arguments and
environment a cell runs with, and rendered into the commands its container runs.

An arm is the release's defaults plus one change (graphs off, overlap off, ...): common args, then the arm's args, then
the profile args when the cell is an Nsight capture. A None value removes a key (TRT-LLM's graphs-off arm sets
cuda_graph_config to None, which the LLM API reads as "no graphs", so None is passed on there rather than removed: the
adapter decides). Every merged result is rendered to golden files in tests, so a flag never changes unnoticed.

Levels (README.md): L0 the forward loop alone, L1 the offline engine API with its scheduler, L2 the OpenAI server.
"""
import copy
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from servebench import ROOT
from servebench.containers import OUT, WORK

CONFIGS = ROOT / "configs" / "engines"
VERSION_SETS = CONFIGS.parent / "versions"


def merge(base, over):
    """Deep merge of mappings: over wins; nested mappings merge; lists and scalars replace."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class Arm:
    name: str
    args: dict
    env: dict
    levels: tuple
    nsys: tuple = ()
    anchor: str = "graph"  # what marks a decode step in a trace: graph launches, or a once-a-step kernel's name


@dataclass
class Release:
    engine: str
    version: str
    image: str
    levels: tuple
    common: dict
    arms_raw: dict
    profile: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, engine, version, root=CONFIGS):
        path = Path(root) / engine / version / "arms.yaml"
        raw = yaml.safe_load(path.read_text())
        known = {"engine", "version", "image", "levels", "common", "arms", "profile"}
        return cls(engine=raw["engine"], version=str(raw["version"]), image=raw["image"],
                   levels=tuple(raw["levels"]), common=raw.get("common") or {}, arms_raw=raw["arms"],
                   profile=raw.get("profile") or {}, extra={k: v for k, v in raw.items() if k not in known},
                   path=path)

    @property
    def digest_pinned_image(self):
        """repo@sha256:... from image.txt (scripts/capture_help.sh), else the tag: runs pin the digest."""
        txt = self.path.parent / "image.txt" if self.path else None
        if txt and txt.exists():
            lines = txt.read_text().split()
            if len(lines) > 1 and "@sha256:" in lines[1]:
                return lines[1]
        return self.image

    @property
    def arm_names(self):
        return list(self.arms_raw)

    def arm(self, name, profile=False):
        a = self.arms_raw[name] or {}
        args = merge(self.common.get("args") or {}, a.get("args"))
        env = merge(self.common.get("env") or {}, a.get("env"))
        if profile:
            args, env = merge(args, self.profile.get("args")), merge(env, self.profile.get("env"))
        return Arm(name=name, args=args, env={k: str(v) for k, v in env.items()},
                   levels=tuple(a.get("levels") or self.levels),
                   nsys=tuple(self.profile.get("nsys") or ()) if profile else (),
                   anchor=a.get("anchor") or self.common.get("anchor") or "graph")


@dataclass(frozen=True)
class Plan:
    """What one container run measures: the batch sweep of one (engine, arm, level, model)."""
    model: str
    batches: tuple
    prompt_len: int = 896
    gen_len: int = 288
    short_gen_len: int = 96
    repeats: int = 3
    warmup: int = 2
    margin: int = 16
    profile: str = "none"  # none | cuda: bracket profile_steps window steps per batch with cudaProfilerStart/Stop
    profile_steps: int = 64
    seed: int = 0

    def driver_args(self):
        return ["--model", self.model, "--batches", ",".join(map(str, self.batches)),
                "--prompt-len", str(self.prompt_len), "--gen-len", str(self.gen_len),
                "--short-gen-len", str(self.short_gen_len), "--repeats", str(self.repeats),
                "--warmup", str(self.warmup), "--margin", str(self.margin), "--profile", self.profile,
                "--profile-steps", str(self.profile_steps), "--seed", str(self.seed), "--out", f"{OUT}/driver.jsonl"]


class Engine:
    """Per-engine command rendering. Subclasses fill in the CLI conventions."""
    name = ""
    driver = None  # servebench/drivers/<name>_offline.py, run with the image's python

    def __init__(self, release):
        self.release = release

    def offline_cmd(self, arm, plan):
        """L1: the offline driver inside the image, the merged args as JSON."""
        if self.driver is None or "L1" not in arm.levels:
            raise ValueError(f"{self.name} {arm.name} has no L1 driver")
        return ["python3", f"{WORK}/servebench/drivers/{self.driver}", "--engine-args",
                json.dumps(arm.args, sort_keys=True, separators=(",", ":")), *plan.driver_args()]

    def server_cmd(self, arm, plan, port, threads=None):
        raise NotImplementedError

    def runner_cmd(self, arm, plan, threads=None):
        raise ValueError(f"{self.name} has no L0 runner")

    def server_files(self, arm):
        """Files a server reads from its /out, written before it starts (TRT-LLM's options YAML); none by default."""
        return {}


def version_set(name):
    """(name, {engine: version}) of configs/versions/<name>.yaml: the releases a session measures (the grid names its
    set)."""
    if not name:
        raise ValueError("a version set is named (configs/versions/<name>.yaml), never assumed")
    return name, {e: str(v) for e, v in yaml.safe_load((VERSION_SETS / f"{name}.yaml").read_text()).items()}


def version_sets():
    """[(name, {engine: version})] of every configs/versions file, oldest first (the files are named by date)."""
    return [version_set(f.stem) for f in sorted(VERSION_SETS.glob("*.yaml"))]


def kebab(key):
    return "--" + key.replace("_", "-")


def render(cmd):
    """A command as one shell-quoted line: golden files show exactly this."""
    return shlex.join(str(c) for c in cmd)
