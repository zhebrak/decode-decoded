"""Decode step time from when output tokens arrive, computed the same way for every engine and every level (an
offline driver's streaming API at L1, the AIPerf client's SSE timestamps at L2). Standard library only: the drivers
import it inside the engine containers.

Input: one list per sequence of (arrival ns, tokens so far) per output chunk, all on one clock. In a fixed batch every
running sequence gains one token per engine step, so while all B sequences are decoding, a sequence's time per token
is the step time.

Window: from the arrival of the last sequence's first token (every prefill done; no prefill or mixed step after it)
to the arrival of the first sequence's last token (no sequence finished yet), then `margin` steps trimmed from each
end, so ramp-up and tail steps never count.

Two estimates inside the window, which agree unless chunks coalesce several tokens:
- interval: the median gap between consecutive single-token chunks;
- rate: per sequence, (last chunk time - first chunk time) / tokens between them, then the median over sequences.
  A coalesced chunk (two tokens arriving together) bends the interval estimate towards zero but not the rate one,
  so rate is the primary.

Stalls: a sequence that made at least STALL_TOKENS fewer tokens inside the window than the batch's median sequence
stopped decoding while the others went on: preempted and recomputed, or queued behind a prefill. A fixed-batch cell
must have none; the count is reported so a silently preempted cell (a batch above the engine's real KV capacity) is
flagged, not averaged in. A pause that holds every sequence at once (a host hiccup: a collection, a log flush) is not
a stall: it is part of the engine's step, stays in the mean, and shows as long_gaps (gaps over LONG_GAP steps) and
max_gap_steps. At batch 1 nothing can be preempted. (A single gap over 3 steps is not a stall: among thousands of
gaps per cell such a rule flags arrival jitter, 3.02x on SGLang's batch 1 in validation, not lost progress.)

after_tokens restricts the estimate to chunks past that many tokens: the differential estimator spans tokens
short_gen_len..gen_len, so its agreement check compares against the cadence over the same tokens (the same context:
KV bytes per step grow with it).
"""
from statistics import median

STALL_TOKENS = 2  # tokens behind the batch's median sequence over the window
LONG_GAP = 3.0    # steps: a gap reported as a hiccup


def all_active(seqs):
    """(start, end) ns of the span where every sequence has started and none has finished, or None."""
    if not seqs or any(not s for s in seqs):
        return None
    lo, hi = max(s[0][0] for s in seqs), min(s[-1][0] for s in seqs)
    return (lo, hi) if hi > lo else None


def _gaps(seqs, lo, hi):
    return [b[0] - a[0] for s in seqs for a, b in zip(s, s[1:]) if a[0] >= lo and b[0] <= hi and b[1] - a[1] == 1]


def _per_token(seqs, lo, hi):
    """Each chunk's gap over the tokens it carried: a coalesced chunk of k tokens gives its gap / k."""
    return [(b[0] - a[0]) / (b[1] - a[1]) for s in seqs for a, b in zip(s, s[1:])
            if a[0] >= lo and b[0] <= hi and b[1] > a[1]]


def _rates(seqs, lo, hi):
    out = []
    for s in seqs:
        inside = [c for c in s if lo <= c[0] <= hi]
        if len(inside) >= 2 and inside[-1][1] > inside[0][1]:
            out.append((inside[-1][0] - inside[0][0]) / (inside[-1][1] - inside[0][1]))
    return out


def _made(seqs, lo, hi):
    """Tokens each sequence made inside [lo, hi]: its count at its last chunk by hi less its count at its last by lo."""
    def at(s, t):
        return max((n for c, n in s if c <= t), default=0)
    return [at(s, hi) - at(s, lo) for s in seqs]


def _quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def estimate(seqs, margin=16, after_tokens=0):
    """Step-time estimates (ns) for one fixed-batch generation. Returns None fields, and says why in `unidentified`,
    when no window can be read: no span with every sequence decoding, or fewer than two steps after trimming."""
    if after_tokens:
        seqs = [[c for c in s if c[1] > after_tokens] for s in seqs]
    span = all_active(seqs)
    out = {"batch": len(seqs), "window_ns": None, "steps": None, "step_ns": None, "interval_ns": None,
           "interval_p10_ns": None, "interval_p90_ns": None, "coalesced_chunks": 0, "stalls": 0, "long_gaps": 0,
           "max_gap_steps": None, "unidentified": None}
    if span is None:
        return out | {"unidentified": "no span where every sequence is decoding"}
    coalesced = lambda lo, hi: sum(1 for s in seqs for a, b in zip(s, s[1:])  # noqa: E731
                                   if a[0] >= lo and b[0] <= hi and b[1] - a[1] > 1)
    # the margin's step from single-token gaps, or, where every chunk carried several tokens (a server streaming
    # every k tokens), from each chunk's gap over its tokens: the rate estimate below never needed single tokens
    rough = _gaps(seqs, *span) or _per_token(seqs, *span)
    if not rough:
        return out | {"coalesced_chunks": coalesced(*span), "unidentified": "no chunk inside the span"}
    step = median(rough)
    lo, hi = span[0] + margin * step, span[1] - margin * step
    if hi - lo < 2 * step:
        return out | {"coalesced_chunks": coalesced(*span), "unidentified": "fewer than two steps after the margins"}
    gaps, rates = _gaps(seqs, lo, hi), _rates(seqs, lo, hi)
    rate = median(rates) if rates else None
    per_token = _per_token(seqs, lo, hi)
    worst = max(per_token) / rate if per_token and rate else None
    made = _made(seqs, lo, hi)
    return {"batch": len(seqs), "window_ns": [lo, hi], "steps": round((hi - lo) / step),
            "step_ns": rate, "interval_ns": median(gaps) if gaps else None,
            "interval_p10_ns": _quantile(gaps, 0.1), "interval_p90_ns": _quantile(gaps, 0.9),
            "coalesced_chunks": coalesced(lo, hi), "stalls": sum(1 for m in made if median(made) - m >= STALL_TOKENS),
            "long_gaps": sum(1 for g in per_token if rate and g > LONG_GAP * rate), "max_gap_steps": worst,
            "unidentified": None if rate else "no sequence with two chunks inside the window"}


def stream_span_ns(seqs):
    """A wave's wall for the differential: each stream's first token to its last, the mean over streams. No prefill,
    and none of the scatter in when the streams started: across the wave (first token of any to last of any) it moved
    vLLM's batch-8 L2 differential 4% between repeats in validation; per stream, 0.7%. The mean, not the median: where
    prefill overlaps decode, streams' spans come in steps, one per prefill chunk, and the median stream falls on a
    different one in each wave (vLLM 14B at 211 sequences: differentials 10% apart by the median, 0.6% by the mean;
    over 46 validation cells, a 90th-percentile range of 7.8% against 10.2%)."""
    spans = [s[-1][0] - s[0][0] for s in seqs if s]
    return sum(spans) / len(spans) if spans else None


def differential(wall_long_ns, wall_short_ns, gen_long, gen_short):
    """Step time from two generations that differ only in output length: prefill, ramp and fixed overheads cancel,
    leaving (gen_long - gen_short) decode steps at the full batch: the step time's independent check."""
    return (wall_long_ns - wall_short_ns) / (gen_long - gen_short)
