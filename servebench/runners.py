"""L0 runner outputs (the forward loop alone, no scheduler) into the same per-batch timing records the L1 drivers write,
so the stack chart reads every level alike. Formats read from the pinned images' source:

- SGLang `python -m sglang.benchmark.one_batch --result-filename F`: appends one JSON line per run with batch_size,
  input_len, output_len and median_decode_latency (seconds per decode step, each step synchronised; v0.5.20
  one_batch.py, `measurement_results`).
- llama.cpp `llama-batched-bench --output-format jsonl`: prints one JSON line per configuration to stdout with pl (the
  batch), tg (decode steps) and t_tg (seconds for all of them; v0.5.0 batched-bench). The step is t_tg / tg, over a
  context growing from the prompt to prompt + tg, the same span the L1 window averages over.
Each batch runs `repeats` times (the runner commands list it repeatedly), so an L0 cell has a median and a range too.
"""
from servebench.results import json_lines


def one_batch_steps(text):
    """{batch: [step ms per run]} from SGLang one_batch's result file."""
    out = {}
    for r in json_lines(text):
        if r.get("median_decode_latency") is not None and r.get("batch_size") is not None:
            out.setdefault(int(r["batch_size"]), []).append(r["median_decode_latency"] * 1e3)
    return out


def batched_bench_steps(text):
    """{batch: [step ms per run]} from llama-batched-bench's jsonl lines (mixed into the container log). Its first run
    is a warm-up the adapter prepends (a first batch-1 run in validation was cold: 23.6 ms a token against 2.87) and is
    dropped here."""
    out = {}
    runs = [r for r in json_lines(text) if {"pl", "tg", "t_tg"} <= r.keys() and r["tg"]]
    for r in runs[1:]:
        out.setdefault(int(r["pl"]), []).append(r["t_tg"] / r["tg"] * 1e3)
    return out
