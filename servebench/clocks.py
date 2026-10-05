"""A 1 Hz nvidia-smi log beside every cell: SM and memory clocks, why the clocks were held down, temperature, power.
A cell timed under a thermal or hardware slowdown is rejected, not silently averaged in; power capping is reported.

nvidia-smi's utilization.gpu is deliberately not logged as a metric: it reports whether any kernel ran in each sample
period, so one tiny kernel per period reads 100%. Occupancy comes from Nsight traces only.

The throttle field was renamed (clocks_throttle_reasons.active -> clocks_event_reasons.active); the logger asks
nvidia-smi which one it knows.
"""
import csv
import math
import subprocess
from pathlib import Path
from statistics import median

from servebench.env import shell

BASE = ("timestamp", "clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")
REASONS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")
# Bits of the active-reasons mask that mean the clock was held below what was asked (NVML clocksEventReasons):
# software power cap, hardware slowdown, software and hardware thermal slowdown, power brake.
SLOWDOWN_BITS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "power_brake"}
IDLE_BIT = 0x1
# Slowdowns that mean something went wrong (heat, a hardware brake), as against the card working at its configured
# power limit (sw_power_cap: normal under load, and part of the card's performance; its share is reported).
FAULTS = ("hw_slowdown", "sw_thermal", "hw_thermal", "power_brake")


def reasons_field(helptext=None):
    helptext = helptext if helptext is not None else (shell("nvidia-smi", "--help-query-gpu") or "")
    return next((f for f in REASONS if f in helptext), REASONS[0])


APPS = ("timestamp", "pid", "process_name", "used_memory")


class ClockLog:
    """with ClockLog(path): ... writes path (CSV, one row a second) for the duration of the block, and beside it
    gpu-processes.csv: every compute process on the GPU, each second (the unit's own engine and anything else, so a
    unit that shared the GPU says so); no-op without nvidia-smi."""

    def __init__(self, path, device="0", period_ms=1000):
        self.path, self.device, self.period_ms = Path(path), device, period_ms
        self.proc = self.apps = None

    def __enter__(self):
        fields = (*BASE, reasons_field())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(",".join(fields) + "\n")
        try:
            self.out = self.path.open("a")
            self.proc = subprocess.Popen(["nvidia-smi", "-i", self.device, f"--query-gpu={','.join(fields)}",
                                          "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
                                         stdout=self.out, stderr=subprocess.DEVNULL)
            apps = self.path.with_name("gpu-processes.csv")
            apps.write_text(",".join(APPS) + "\n")
            self.apps_out = apps.open("a")
            self.apps = subprocess.Popen(["nvidia-smi", "-i", self.device, f"--query-compute-apps={','.join(APPS)}",
                                          "--format=csv,noheader,nounits", "-lms", str(self.period_ms)],
                                         stdout=self.apps_out, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            self.proc = self.apps = None
        return self

    def __exit__(self, *exc):
        for proc, out in ((self.proc, "out"), (self.apps, "apps_out")):
            if proc:
                proc.terminate()
                proc.wait(timeout=10)
                getattr(self, out).close()


def summarise(path):
    """{sm_mhz: min/median/max; sm_mhz_median_unthrottled: the median over the busy samples no slowdown touched, what
    a clock lock held (None when every busy sample was slowed: a power cap pulls the clock under the lock by design,
    0.9 of the busy samples at 14B on the pod, and the median of every sample read 1455 MHz against a 1980 lock);
    mem_mhz median; max temperature and power; slowdowns seen}; None for an empty log."""
    rows = list(csv.reader(Path(path).open()))
    if not rows or not set(BASE).issubset(rows[0]) or not any(f.endswith("reasons.active") for f in rows[0]):
        return None
    header, rows = rows[0], [r for r in rows[1:] if len(r) == len(rows[0])]
    if not rows:
        return None
    col = {h: i for i, h in enumerate(header)}
    def num(r, f):
        try:
            v = float(r[col[f]])
            return v if math.isfinite(v) else None
        except ValueError:
            return None
    reasons_col = next(f for f in header if f.endswith("reasons.active"))
    masks = [int(r[col[reasons_col]].strip(), 16) if r[col[reasons_col]].strip().startswith("0x") else 0 for r in rows]
    sm = [(v, m) for r, m in zip(rows, masks) if (v := num(r, "clocks.sm")) is not None]
    if not sm:
        return None
    held = [v for v, m in sm if not m & IDLE_BIT and not any(m & bit for bit in SLOWDOWN_BITS)]
    busy = [m for m in masks if not m & IDLE_BIT] or [0]  # shares over the samples the GPU was working
    shares = {name: sum(1 for m in busy if m & bit) / len(busy) for bit, name in SLOWDOWN_BITS.items()}
    return {"samples": len(rows), "sm_mhz_min": min(v for v, _ in sm), "sm_mhz_median": median(v for v, _ in sm),
            "sm_mhz_max": max(v for v, _ in sm), "sm_mhz_median_unthrottled": median(held) if held else None,
            "mem_mhz_median": median([v for r in rows if (v := num(r, "clocks.mem")) is not None] or [None]),
            "temperature_max": max((v for r in rows if (v := num(r, "temperature.gpu")) is not None), default=None),
            "power_max_w": max((v for r in rows if (v := num(r, "power.draw")) is not None), default=None),
            "slowdowns": {k: round(v, 3) for k, v in shares.items() if v}}


def processes(path):
    """The most compute processes seen on the GPU at once in a unit's gpu-processes.csv (its own engine is one), or
    None without a log."""
    path = Path(path)
    if not path.exists():
        return None
    at = {}
    for r in list(csv.reader(path.open()))[1:]:
        if len(r) >= 2:
            at.setdefault(r[0], set()).add(r[1].strip())
    return max((len(p) for p in at.values()), default=0)
