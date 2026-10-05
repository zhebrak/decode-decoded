import random

import pytest

from servebench.cadence import all_active, differential, estimate


def staggered(batch=8, gen=200, step=1_000_000, stagger=3, jitter=0, seed=0, coalesce_every=0):
    """A fixed batch whose prefills finish `stagger` steps apart (chunked prefill), then one token per step each."""
    rng = random.Random(seed)
    seqs = []
    for i in range(batch):
        start = i * stagger
        chunks, n = [], 0
        for k in range(gen):
            n += 1
            t = (start + k) * step + rng.randint(-jitter, jitter)
            if coalesce_every and k % coalesce_every == 1 and chunks:
                chunks[-1] = (t, n)  # this token arrived with the previous one: one chunk of two
                continue
            chunks.append((t, n))
        seqs.append(chunks)
    return seqs


def test_window_starts_after_every_prefill_and_ends_before_any_finish():
    seqs = staggered(batch=4, gen=50, step=10, stagger=5)
    assert all_active(seqs) == (15 * 10, 49 * 10)


def test_exact_cadence():
    e = estimate(staggered())
    assert e["step_ns"] == 1_000_000 and e["interval_ns"] == 1_000_000
    assert e["steps"] == pytest.approx(200 - 7 * 3 - 32, abs=1)


def test_jitter_is_median_robust():
    e = estimate(staggered(jitter=50_000, seed=3))
    assert e["step_ns"] == pytest.approx(1_000_000, rel=0.005)
    assert e["interval_p10_ns"] < 1_000_000 < e["interval_p90_ns"]


def test_coalesced_chunks_bias_intervals_not_rate():
    e = estimate(staggered(coalesce_every=5))
    assert e["coalesced_chunks"] > 0
    assert e["step_ns"] == pytest.approx(1_000_000, rel=0.002)


def test_too_short_a_window_gives_none():
    assert estimate(staggered(batch=4, gen=20, stagger=5))["step_ns"] is None
    assert estimate([])["step_ns"] is None


def test_differential():
    assert differential(500_000_000, 308_000_000, 288, 96) == 1_000_000


def test_a_preempted_sequence_shows_as_stalls():
    seqs = staggered(batch=4, gen=200)
    s = seqs[2]
    k = next(i for i, c in enumerate(s) if c[1] == 120)
    seqs[2] = s[:k] + [(t + 15 * 1_000_000, n) for t, n in s[k:]]  # stopped for 15 steps, then recomputed
    e = estimate(seqs)
    assert e["stalls"] >= 1 and e["max_gap_steps"] > 10
    assert estimate(staggered())["stalls"] == 0


def test_after_tokens_restricts_to_the_tail():
    seqs = staggered(batch=4, gen=200)
    tail = estimate(seqs, after_tokens=96)
    assert tail["step_ns"] == 1_000_000 and tail["window_ns"][0] > estimate(seqs)["window_ns"][0]


def test_a_pause_of_every_sequence_is_a_hiccup_not_a_stall():
    seqs = staggered(batch=4, gen=200)
    k = next(i for i, c in enumerate(seqs[0]) if c[1] == 120)
    t_pause = seqs[0][k][0]
    seqs = [[(t + 5_000_000 if t >= t_pause else t, n) for t, n in s] for s in seqs]   # all held 5 steps at once
    e = estimate(seqs)
    assert e["stalls"] == 0 and e["long_gaps"] >= 4 and e["max_gap_steps"] > 3
    one = staggered(batch=1, gen=200)
    k = next(i for i, c in enumerate(one[0]) if c[1] == 120)
    one[0] = one[0][:k] + [(t + 5_000_000, n) for t, n in one[0][k:]]
    assert estimate(one)["stalls"] == 0                                              # batch 1: nothing to preempt


def test_stream_span_ignores_when_streams_started():
    """Four streams of the same 10 steps, started 0-3 ms apart: each stream's own span is the wave's wall for the L2
    differential, never the span across the wave (first token of any to last token of any)."""
    from servebench.cadence import stream_span_ns
    seqs = [[(start + k * 1_000_000, k + 1) for k in range(10)] for start in (0, 1_000_000, 2_000_000, 3_000_000)]
    assert stream_span_ns(seqs) == 9_000_000 and stream_span_ns([[], []]) is None
