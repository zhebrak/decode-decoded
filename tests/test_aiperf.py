"""Against a real AIPerf 0.13.0 raw export: llama-server v0.5.0 on CPU (Qwen3-0.6B Q8_0, 4 streams of 48 tokens), the
CPU rehearsal of the L2 path. Warm-up record first, then the measured wave."""
from pathlib import Path

import pytest

from servebench.aiperf import aiperf_cmd, read_raw
from servebench.cadence import estimate
from servebench.stream_client import chunk_tokens

RAW = Path(__file__).parent / "fixtures" / "aiperf_raw_llamacpp_cpu.jsonl"


def test_reads_the_measured_wave_only():
    seqs, meta = read_raw(RAW)
    assert len(seqs) == 4 and all(m["status"] == 200 and not m["cancelled"] for m in meta)
    assert [s[-1][1] for s in seqs] == [48] * 4  # the closing empty chunk is not a token
    assert all(b[0] >= a[0] and b[1] == a[1] + 1 for s in seqs for a, b in zip(s, s[1:]))
    assert len(read_raw(RAW, phase="warmup")[0]) == 1


def test_cadence_on_the_rehearsal():
    e = estimate(read_raw(RAW)[0], margin=4)
    assert e["step_ns"] == pytest.approx(18e6, rel=0.15)  # ~18 ms a step, as AIPerf's own inter_chunk_latency says


@pytest.mark.parametrize("msg,n", [
    ({"choices": [{"text": " man", "finish_reason": None}]}, 1),
    ({"choices": [{"text": "", "finish_reason": None}]}, 1),          # a partial UTF-8 character
    ({"choices": [{"text": "", "finish_reason": "length"}]}, 0),      # llama.cpp's closing chunk
    ({"choices": [{"text": "end", "finish_reason": "length"}]}, 1),   # vLLM's last token
    ({"choices": [{"token_ids": [1, 2], "text": "ab"}]}, 2),
    ({"choices": [], "usage": {"completion_tokens": 48}}, 0),         # the usage-only chunk
])
def test_chunk_tokens(msg, n):
    assert chunk_tokens(msg) == n


def test_command():
    cmd = aiperf_cmd("http://127.0.0.1:8000", "m", "Qwen/Qwen3-8B", 64, 896, 288, "/o", cpuset="22-25")
    s = " ".join(cmd)
    assert cmd[:6] == ["taskset", "-c", "22-25", "env", "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]  # never online
    assert "--concurrency 64 --request-count 64 --warmup-request-count 64" in s
    assert '--extra-inputs {"ignore_eos":true,"temperature":0}' in s and "min_tokens" not in s
    assert "--export-level raw" in s
