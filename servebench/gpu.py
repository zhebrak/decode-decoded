"""The GPU gates every session passes before a cell runs: the right card, nothing else on it, and clocks locked when the
provider allows it.

- SXM gate: nvidia-smi must name the card "NVIDIA H100 80GB HBM3" (H100 SXM5). PCIe (2.0 TB/s) and NVL (3.9 TB/s)
  cards have other peaks and are refused, never mixed into the same chart.
- Idle gate: gpu_state and wait_for_idle_gpu, with injectable state, sleep and clock so it is tested without a GPU.
- Clock lock: nvidia-smi -lgc needs root; on a rental that refuses it the session records "unlocked" and goes on
  (clocks are logged every second either way, servebench.clocks).
"""
import os
import time

from servebench.env import shell

EXPECTED = "NVIDIA H100 80GB HBM3"
IDLE_MIB, IDLE_CHECKS, IDLE_EVERY_S = 1024, 3, 2.0
IDLE_TIMEOUT_S = float(os.environ.get("DA_IDLE_TIMEOUT", 180))  # longer on a shared box, where others' jobs finish


class WrongGPU(RuntimeError):
    pass


class GPUBusy(RuntimeError):
    """The idle gate gave up: the GPU stayed busy, or nvidia-smi gave no answer (a GPU fallen off the bus)."""


def check_card(name, expected=EXPECTED):
    """Raises unless the card is the one the grid names (configs/grid/*.yaml `gpu`): the study grid names the H100
    SXM5, a validation grid the card it validates the harness on, and results never mix across them."""
    if name != expected:
        raise WrongGPU(f"this box has {name!r}; this grid runs on {expected!r} only: other cards have other peaks "
                       f"and must not share a chart")


IDLE_MODE = os.environ.get("DA_IDLE_MODE", "empty")  # empty: no memory, no process; activity: no utilisation


def gpu_state(device="0"):
    """(MiB used, compute processes) on this GPU, or None without nvidia-smi. With DA_IDLE_MODE=activity (a shared
    validation box where another user's idle server holds memory), memory and processes are reported as zero whenever
    the GPU's utilisation is 0: only work, not residence, keeps it busy. The study grid keeps the strict default."""
    used = shell("nvidia-smi", "-i", device, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits")
    if used is None:
        return None
    mib, util = (int(x.strip()) for x in used.splitlines()[0].split(","))
    if IDLE_MODE == "activity":
        return (0, 0) if util == 0 else (mib, 1)
    return mib, len(compute_pids(device))


def compute_pids(device="0"):
    """The pids of the compute processes on this GPU, [] without nvidia-smi."""
    apps = shell("nvidia-smi", "-i", device, "--query-compute-apps=pid", "--format=csv,noheader") or ""
    return [a.strip() for a in apps.splitlines() if a.strip()]


def wait_for_idle_gpu(state=gpu_state, sleep=time.sleep, clock=time.monotonic):
    """Returns once the GPU is idle on IDLE_CHECKS checks in a row, IDLE_EVERY_S apart; raises GPUBusy after
    IDLE_TIMEOUT_S, and at once when nvidia-smi gives no answer: a GPU that cannot be queried is not an idle one (a
    unit started on it fails fast, and so would every unit after it). A run with no GPU passes no gate at all."""
    t0, idle = clock(), 0
    while True:
        s = state()
        if s is None:
            raise GPUBusy("nvidia-smi gave no answer: the GPU cannot be queried")
        idle = idle + 1 if s[0] < IDLE_MIB and s[1] == 0 else 0
        if idle >= IDLE_CHECKS:
            return
        if clock() - t0 > IDLE_TIMEOUT_S:
            raise GPUBusy(f"GPU still busy after {IDLE_TIMEOUT_S:.0f} s ({s[0]} MiB used, {s[1]} compute processes); "
                          "the previous cell's container may not have exited")
        sleep(IDLE_EVERY_S)


def lock_clocks(mhz, device="0", run=shell):
    """Locks the SM clock at mhz (both ends of the range); True when nvidia-smi accepted it."""
    return run("sudo", "-n", "nvidia-smi", "-i", device, "-lgc", f"{mhz},{mhz}") is not None


def unlock_clocks(device="0", run=shell):
    return run("sudo", "-n", "nvidia-smi", "-i", device, "-rgc") is not None
