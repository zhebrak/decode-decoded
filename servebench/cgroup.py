"""The CPU quota a process runs under (cgroup v2), in the standard library alone so the drivers inside an engine's
image can read it too. A container can see 64 CPUs and be allowed 6.8, and a throttled quota stretches host loops
without leaving a trace: anatomy.grid uses as many cores as the quota pays for, and each measured wave records the
quota periods it spanned and how many of them throttled it (anatomy.tidy rejects a cell throttled beyond tolerance).
Inside a Docker container the files are the container's own: no quota unless one was set."""
from pathlib import Path

ROOT = Path("/sys/fs/cgroup")


def cpu_quota(cgroup=ROOT):
    """Whole CPUs the quota allows ("680000 100000" -> 6), None without one."""
    try:
        quota, period = (cgroup / "cpu.max").read_text().split()
        return None if quota == "max" else max(1, int(quota) // int(period))
    except (OSError, ValueError):
        return None


def cpu_stat(cgroup=ROOT):
    """{"periods", "throttled"}: the quota's enforcement periods so far and how many throttled the cgroup; {} without
    a quota or outside cgroup v2."""
    if cpu_quota(cgroup) is None:
        return {}
    try:
        stat = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines() if line.strip())
        return {"periods": int(stat["nr_periods"]), "throttled": int(stat["nr_throttled"])}
    except (OSError, ValueError, KeyError):
        return {}


def since(before, cgroup=ROOT):
    """The periods and throttled periods since `before` (a cpu_stat), {} without a quota: what a wave record keeps."""
    after = cpu_stat(cgroup)
    return {k: after[k] - before[k] for k in after} if before and after else {}


def throttled_share(records):
    """The share of the quota periods the records' waves spanned in which the quota throttled them, None where none
    was under a quota."""
    cpu = [r["cpu"] for r in records if r and r.get("cpu")]
    periods = sum(c["periods"] for c in cpu)
    return sum(c["throttled"] for c in cpu) / periods if periods else None
