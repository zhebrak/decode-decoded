"""Provenance for every cell record: the code, the host, the GPU, the images and the model revisions a number came from.
The host is a variable in this study (the same H100 behind a slower CPU gives a different answer), so it is recorded,
never assumed. Everything here is best effort: a field that cannot be read is None, never a guess.

Beyond the basics (shell, git sha, GPU name) it records the driver, CUDA, nsys, CPU and governor, the image digests
and the Hugging Face snapshot each model resolved to.
"""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

from servebench import ROOT
from servebench.containers import DOCKER, cores


def hf_home():
    """The Hugging Face cache root, as huggingface_hub resolves it (HF_HOME, else ~/.cache/huggingface): the engines'
    /hf mount and the revision lookup read the same one."""
    return Path(os.environ.get("HF_HOME") or "~/.cache/huggingface").expanduser()


def hub_dir(repo):
    """A model's folder in the hub cache (models--org--name), the layout huggingface_hub writes."""
    return hf_home() / "hub" / f"models--{repo.replace('/', '--')}"


def shell(*cmd, timeout=30):
    """stdout of a command, stripped, or None when it is missing, fails or times out."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def git_sha(cwd=None):
    """git describe of the code that ran, -dirty when a tracked file outside results/ differs from HEAD (the repo
    tracks its tidy results, and a run that rewrote them still ran the committed code)."""
    git = ("git", "-C", str(cwd)) if cwd else ("git",)
    sha = shell(*git, "describe", "--always", "--tags")
    if sha and shell(*git, "status", "--porcelain", "--untracked-files=no", "--", ":/", ":(top,exclude)results"):
        sha += "-dirty"
    return sha


def gpu_query(*fields, device="0"):
    """{field: value} from nvidia-smi --query-gpu for one device, or {} without nvidia-smi."""
    out = shell("nvidia-smi", "-i", device, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits")
    return dict(zip(fields, (v.strip() for v in out.split(",")))) if out else {}


def gpu_local_cpus(device="0", sys_root=Path("/sys")):
    """The CPUs this process may use on the GPU's own NUMA node (its PCI device's local_cpulist, within the affinity
    mask), in order; every allowed CPU where the node is unknown. On a two-socket host, threads that drift to the
    other socket pay a cross-node hop a single-node VM never does."""
    allowed = cores(None)
    bus = (gpu_query("pci.bus_id", device=device).get("pci.bus_id") or "").lower()
    local = sys_root / "bus" / "pci" / "devices" / bus[-12:] / "local_cpulist"  # 00000000:18:00.0 -> 0000:18:00.0
    near = set(cores(local.read_text().strip())) if bus and local.exists() else set()
    return [c for c in allowed if c in near] or allowed


def cpu():
    """The host CPU as lscpu reports it, the cores this process may use, and the frequency governor."""
    info = {}
    for line in (shell("lscpu") or "").splitlines():
        k, _, v = line.partition(":")
        info[k.strip()] = v.strip()
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    return {"model": info.get("Model name"), "sockets": info.get("Socket(s)"),
            "cores_per_socket": info.get("Core(s) per socket"), "threads_per_core": info.get("Thread(s) per core"),
            "cpus": info.get("CPU(s)"), "usable_cpus": len(cores(None)),
            "max_mhz": info.get("CPU max MHz"), "governor": gov.read_text().strip() if gov.exists() else None}


def gpu():
    q = gpu_query("name", "pci.bus_id", "driver_version", "vbios_version", "memory.total", "clocks.max.sm",
                  "clocks.max.mem", "power.limit", "persistence_mode")
    smi = shell("nvidia-smi") or ""
    cuda = next((p.split(":")[1].strip().split()[0] for p in smi.split("|") if "CUDA Version" in p), None)
    return {**q, "cuda_driver_api": cuda}


def tools():
    """The tools' versions; nsys is the pinned install every capture mounts (DA_NSYS), else whatever is on the PATH."""
    nsys = Path(os.environ["DA_NSYS"]) / "bin" / "nsys" if os.environ.get("DA_NSYS") else "nsys"
    return {"nsys": shell(str(nsys), "--version"),
            "docker": shell(*DOCKER, "version", "--format", "{{.Server.Version}}"),
            "kernel": platform.release(), "python": platform.python_version()}


def host():
    """The host fingerprint: its id is a hash of what identifies the machine type, so sessions on the same instance
    type share it and a different CPU or driver shows up as a different id."""
    rec = {"hostname": platform.node(), "cpu": cpu(), "gpu": gpu(), "tools": tools()}
    ident = {"cpu": rec["cpu"]["model"], "cpus": rec["cpu"]["cpus"], "gpu": rec["gpu"].get("name"),
             "driver": rec["gpu"].get("driver_version"), "nsys": rec["tools"].get("nsys")}
    rec["id"] = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
    return rec


def model_revision(repo):
    """The commit a Hugging Face model id resolves to in the local cache (refs/main), or None: never downloads."""
    ref = hub_dir(repo) / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


PINS = ROOT / "configs" / "models" / "revisions.yaml"


def pinned(repo):
    """The commit configs/models/revisions.yaml pins a model at (what scripts/fetch_models.sh downloads and points
    refs/main at), or None for a model it does not list."""
    import yaml

    return (yaml.safe_load(PINS.read_text()) or {}).get(repo) if PINS.exists() else None


def pin_problems(models):
    """Each model whose local refs/main is not its pin, with what it points at: what a session checks before its
    grid, since every engine loads whatever refs/main names (scripts/fetch_models.sh writes it)."""
    out = []
    for m in models:
        want, got = pinned(m), model_revision(m)
        if want is None:
            out.append(f"{m}: no pin in configs/models/revisions.yaml")
        elif got != want:
            out.append(f"{m}: refs/main is {got or 'missing'}, the pin is {want} (scripts/fetch_models.sh {m})")
    return out
