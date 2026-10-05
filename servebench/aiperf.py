"""NVIDIA AIPerf (v0.13.0) as the L2 timing client: its multi-process workers avoid the single-process asyncio queueing
that inflates latency under load, and it is a tool maintainers already trust. Each invocation sends one
warm-up wave of B requests and one measured wave of B, all at concurrency B, so the measured wave is a fixed batch
through the full serving path. --export-level raw keeps every SSE message with its perf_ns (CLOCK_MONOTONIC, so the
worker processes share a clock), which is turned into the same (ns, tokens so far) series the L1 drivers record and
estimated by servebench.cadence the same way (flags checked against aiperf 0.13.0 --help).

Records of the measured phase carry metadata.benchmark_phase == "profiling"; warm-up records are dropped.
"""
import csv
import json
from pathlib import Path

from servebench.stream_client import advance

RAW = "profile_export_raw.jsonl"


def aiperf_cmd(url, model, tokenizer, batch, prompt_len, gen_len, artifact_dir, seed=0, aiperf="aiperf",
               cpuset=None, extra_inputs=None):
    # no min_tokens: on vLLM and SGLang it turns on a per-step logits penalty pass (see the L1 drivers)
    extras = {"ignore_eos": True, "temperature": 0, **(extra_inputs or {})}
    cmd = [aiperf, "profile", "--model", model, "--tokenizer", tokenizer, "--url", url,
           "--endpoint-type", "completions", "--streaming", "--concurrency", str(batch), "--request-count", str(batch),
           "--warmup-request-count", str(batch), "--isl", str(prompt_len), "--isl-stddev", "0", "--osl", str(gen_len),
           "--osl-stddev", "0", "--export-level", "raw", "--random-seed", str(seed), "--ui-type", "none",
           "--artifact-dir", str(artifact_dir)]
    # one JSON object, never key:value pairs: AIPerf splits those at commas, so a nested value (TRT-LLM's
    # stream_options) arrives in pieces and the server answers 400; JSON also keeps booleans typed
    cmd += ["--extra-inputs", json.dumps(extras, separators=(",", ":"))]
    # offline, as the engines are: online, AIPerf would resolve the tokenizer's `main` on the Hub, and huggingface_hub
    # rewrites refs/main to any newer commit there, so one README edit upstream would repoint the pinned snapshot
    # (configs/models/revisions.yaml) at one holding only tokenizer files, and every later engine would load that
    cmd = ["env", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1", *cmd]
    return ["taskset", "-c", cpuset, *cmd] if cpuset else cmd


def read_raw(path, phase="profiling"):
    """[[(perf_ns, tokens so far), ...] per request] from a raw export, measured phase only, requests in start order;
    plus per-request metadata (status, cancelled, tokens, and the prompt's tokens as the server counted them, from the
    stream's usage) for the record."""
    from servebench.results import json_lines

    seqs, meta = [], []
    recs = json_lines(Path(path).read_text())  # a torn last line (the client killed mid-write) is skipped, as in
    recs = [r for r in recs if r["metadata"].get("benchmark_phase") == phase]  # driver.jsonl
    recs.sort(key=lambda r: r["metadata"]["request_start_ns"])
    for r in recs:
        n, chunks, prompt = 0, [], None
        for resp in r.get("responses") or []:
            for pkt in resp.get("packets") or []:
                if pkt.get("name") != "data" or pkt.get("value", "").strip() == "[DONE]":
                    continue
                body = json.loads(pkt["value"])
                prompt = (body.get("usage") or {}).get("prompt_tokens") or prompt  # the last chunk's usage
                m = advance(n, body)
                if m > n:
                    n = m
                    chunks.append((resp["perf_ns"], n))
        seqs.append(chunks)
        meta.append({"status": r.get("status"), "cancelled": r["metadata"].get("was_cancelled"),
                     "start_ns": r["metadata"].get("request_start_ns"), "tokens": n, "prompt_tokens": prompt})
    return seqs, meta


SERVER_METRICS = "server_metrics_export.csv"


def server_metrics(path):
    """{metric: {column: value}} from AIPerf's server-metrics CSV (one section per metric type, each with its own
    header; counters carry the profiling phase's delta as `total`, gauges and histograms their statistics). {} when
    the file is missing."""
    path = Path(path)
    if not path.exists():
        return {}
    out, header = {}, None
    for row in csv.reader(line for line in path.read_text().splitlines() if not line.startswith("#")):
        if not row:
            header = None
            continue
        if header is None:
            header = row
            continue
        rec = dict(zip(header, row))
        vals = {}
        for k, v in rec.items():
            try:
                vals[k] = float(v)
            except ValueError:
                pass
        out[rec.get("Metric", "")] = vals
    return out


def client_whole_ns(seqs):
    """The client's time per token over whole requests, first token to last, every stream pooled: the same population
    the server's own figure covers (server_step_ns), prefills of other streams interleaved with decode included."""
    spans = [(s[-1][0] - s[0][0], s[-1][1] - s[0][1]) for s in seqs if len(s) > 1 and s[-1][1] > s[0][1]]
    return sum(t for t, _ in spans) / sum(n for _, n in spans) if spans else None


def server_step_ns(metrics):
    """The server's own time per generated token per sequence, in ns, from its metrics over the profiling phase: the
    client's timestamps checked against the server's clock. It covers whole requests (prefills of other streams
    inside a generation included), so it is compared with client_whole_ns, never with the steady-window cadence.
    None where the server exposes nothing usable.
    - llama-server: tokens_predicted_seconds / tokens_predicted (each slot's generation time summed over slots, over
      the tokens they made). In validation (A100, B8): 6.00-6.11 ms against the client's 6.03-6.14 over the same
      requests, whose prompts came from the slot cache, so no prefill fell inside a generation.
    - a server with an inter-token latency histogram (vLLM's vllm:inter_token_latency_seconds; SGLang's by the same
      suffix, not yet seen in a real export): sum / count over the phase, not the CSV's avg, which is rounded to
      0.1 ms. vLLM in validation (A100): 2.446 ms at B1 and 3.480 at B8, against the client's 2.461 and 3.501."""
    t, n = metrics.get("llamacpp:tokens_predicted_seconds", {}), metrics.get("llamacpp:tokens_predicted", {})
    if t.get("total") and n.get("total"):
        return t["total"] / n["total"] * 1e9
    for name, m in metrics.items():
        if name.endswith("inter_token_latency_seconds"):
            if m.get("count") and m.get("sum"):
                return m["sum"] / m["count"] * 1e9
            if m.get("avg"):
                return m["avg"] * 1e9
    return None
