"""llama-server's log read for what its slots did: whether each measured wave ran its batch in one consecutive run of
slots with the graph reused every step, or in slots that are not one run (a step up to 1.8x as long in validation). What
anatomy.tidy checks on every llama.cpp server unit (llama_slots), per server of the unit's log."""
import re
import statistics
from pathlib import Path

LAUNCH = re.compile(r"launch_slot_: id +(\d+) \| task (\d+) \| processing task")
REUSED = re.compile(r"print_timing: id +\d+ \| task (\d+) \| +graphs reused = +(\d+)")
EVAL = re.compile(r"print_timing: id +\d+ \| task (\d+) \| +eval time = [^/]*/ +(\d+) tokens")
PROMPT = re.compile(r"print_timing: id +\d+ \| task (\d+) \| +prompt eval time = +([\d.]+) ms")
REUSE_MIN = 0.9  # llama.cpp's graph reuses a measured wave must reach, over its steps (a whole wave: about 285 of 288)
STALL_X = 10  # a measured wave whose prefill took this many times the server's median prefill stalled (a fresh
# server compiling its kernels: seconds for 896 tokens against about 100 ms; the span differential then spans the
# stall where the median cadence does not)


def llama_slots(d, recs):
    """{batch: why llama-server's measured waves there are not one batch} from its log. container.log holds every
    server of the unit in turn (a server per batch appends its own), each starting at its load_model line with its
    slot count and its task ids again from 0, so the log is cut per server first, and each server takes its share of
    the unit's batches in order (one batch each, or every batch for one shared server). Within a server, its requests
    in launch order are cut into the waves its batches recorded: a warm-up and a measured wave of B each, or, in logs
    from before the warm-up wave, the measured wave alone; a count that fits neither is a problem stated, never a pass.
    A batch whose wave failed (an error record) ends what can be cut on its server. Each measured wave is judged: its
    sequences in one consecutive run of slots, and llama.cpp's graph-reuse count rising by at least REUSE_MIN of the
    wave's steps; a server sized for one batch must have one slot per sequence.
    A shared server picks a repeated prompt's old slot back by similarity, so a later batch lands in slots that are
    not one run and its graph is reused a few times in 288 steps instead of about 285, at up to 1.8x the step of a
    server started fresh for that batch. {} for a log without the lines."""
    return judged(d, recs)[0]


def llama_prefill(d, recs):
    """{batch: why a measured wave's prefill stalled} from llama-server's log: a request whose prompt evaluation took
    STALL_X times the server's median. The decode cadence (a median) ignores such a pause; the span differential and a
    noise or sentinel reading do not (anatomy.noise.reading_problems excludes the reading; a cell records it as
    prefill_problem, its range rule sending the unit to the redo round)."""
    return judged(d, recs)[1]


def judged(d, recs):
    """(slots problems, prefill stalls), each {batch: message}, from the unit's log cut per server (llama_slots)."""
    log = Path(d) / "container.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if not LAUNCH.search(text):
        return {}, {}
    starts = [m.start() for m in SERVER.finditer(text)] or [0]
    parts = [text[x:y] for x, y in zip([0, *starts[1:]], [*starts[1:], len(text)])]
    waves = {}  # batch -> its recorded waves, in order; a batch's error ends it
    for r in recs:
        if r.get("kind") in ("long", "short", "profile", "error") and "batch" in r:
            waves.setdefault(r["batch"], []).append(r["kind"])
    batches = list(waves)
    if len(parts) == len(batches):
        share = [[b] for b in batches]
    elif len(parts) == 1:
        share = [batches]
    else:
        return {b: f"could not cut the log: {len(parts)} servers for {len(batches)} batches" for b in batches}, {}
    out, stalls = {}, {}
    # the agreement probe's requests (anatomy.served.greedy: one per prompt, after the unit's last wave) land on the
    # last server: launches the waves do not account for, never a cut that fails for them
    probes = sum(1 for r in recs if r.get("kind") == "greedy")
    for i, (part, bs) in enumerate(zip(parts, share)):
        slots, stalled = server_waves(part, bs, waves, extra=probes if i == len(parts) - 1 else 0)
        out |= slots
        stalls |= stalled
    return out, stalls


SERVER = re.compile(r"load_model: initializing, n_slots = (\d+)")


def server_waves(text, batches, waves, extra=0):
    """(slots problems, prefill stalls) for one server's share of the log, each {batch: problem}. extra: requests after
    the measured waves that the waves do not account for (the greedy probe's), allowed at the end of the launches and
    never judged."""
    launches = [(int(t), int(slot)) for slot, t in LAUNCH.findall(text)]
    reused = {int(t): int(n) for t, n in REUSED.findall(text)}
    steps = {int(t): int(n) for t, n in EVAL.findall(text)}
    prompt = {int(t): float(ms) for t, ms in PROMPT.findall(text)}
    typical = statistics.median(prompt.values()) if prompt else None  # this server's prefill, warm-ups included
    m = SERVER.search(text)
    out, stalls = {}, {}
    if m and len(batches) == 1 and int(m.group(1)) != batches[0]:
        out[batches[0]] = f"a server of {m.group(1)} slots ran batch {batches[0]}"
    ok = []  # the waves that ran, up to the first error
    for b in batches:
        n = 0
        for kind in waves[b]:
            if kind == "error":
                break
            n += 1
        ok.append((b, n))
        if n < len(waves[b]):
            break
    measured = sum(b * n for b, n in ok)
    if len(launches) not in (measured, 2 * measured, measured + extra, 2 * measured + extra):
        return {**out, **{b: f"could not cut the log: {len(launches)} launches where its waves need {measured} or "
                             f"{2 * measured}" + (f" (and {extra} probe requests after them)" if extra else "")
                          for b in batches}}, {}
    per = 2 if len(launches) in (2 * measured, 2 * measured + extra) else 1  # a warm-up wave before each, or none
    i, last = 0, 0
    for b, n in ok:
        for _ in range(n):
            for measured_wave in [False] * (per - 1) + [True]:
                wave, i = launches[i:i + b], i + b
                done = max((reused[t] for t, _ in wave if t in reused), default=None)
                if measured_wave:
                    slots = sorted(s for _, s in wave)
                    k = max((steps.get(t, 0) for t, _ in wave), default=0)
                    if slots != list(range(slots[0], slots[0] + b)):
                        out.setdefault(b, f"llama-server ran batch {b} in slots that are not one run ({slots[0]}-"
                                          f"{slots[-1]}, {len(set(slots))} distinct)")
                    elif done is not None and k and done - last < REUSE_MIN * k:
                        out.setdefault(b, f"llama-server reused its graph {done - last} times in a {k}-step wave")
                    worst = max((prompt.get(t, 0.0) for t, _ in wave), default=0.0)
                    if typical and worst > STALL_X * typical:
                        stalls.setdefault(b, f"llama-server's prefill stalled: {worst:.0f} ms for a request of a "
                                             f"measured wave against the server's median {typical:.0f}")
                last = done if done is not None else last
    return out, stalls
