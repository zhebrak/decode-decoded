"""The pattern pipeline under a null: tables where no engine differs from another, their noise as the study's is (a
process's offset shared by every batch it measured, heavy tails), run through derive.tables and patterns like real
cells. The differences it calls real must stay near what chance() expects; and the A/A check, the comparisons' rule
on processes of one cell, must find the rate its K implies."""
import math
import random

from anatomy.report import derive
from anatomy.report.patterns import patterns

ENGINES = {"vllm": "L1", "sglang": "L1", "trtllm": "L1", "llamacpp": "L2"}
MODELS = {"Qwen/Qwen3-0.6B": 6e8, "Qwen/Qwen3-1.7B": 1.7e9, "Qwen/Qwen3-8B": 8e9}
BATCHES = (1, 8, 64)
SD = 0.004  # 0.4% between processes, inside what validation measured


def t4(rng):
    """Student's t with 4 degrees of freedom, scaled to unit variance: heavier tails than the Gaussian chance()
    assumes."""
    z = rng.gauss(0, 1)
    chi = sum(rng.gauss(0, 1) ** 2 for _ in range(4))
    return z / math.sqrt(chi / 4) / math.sqrt(2)


def null_tables(seed):
    rng = random.Random(seed)
    rows, noise_rows = [], []
    for engine, level in ENGINES.items():
        for model, params in MODELS.items():
            offset = SD * rng.gauss(0, 1) / math.sqrt(2)  # the unit's process, shared by its batches
            for b in BATCHES:
                step = (0.5 + params / 1e9 * 0.3) * (1 + 0.01 * math.log2(b))  # the same truth for every engine
                rows.append({"engine": engine, "version": "x", "arm": "defaults", "level": level, "model": model,
                             "params": params, "batch": b, "step_ms": step * (1 + offset + SD / math.sqrt(2) * t4(rng)),
                             "occupied": 0.95, "gpu": "NVIDIA H100 80GB HBM3", "session": "S2"})
                # the noise S1 would measure: six processes of the same noise (offset and tails), their sample SD,
                # never the true SD (a null given the truth tests the rule, not the noise it will be read with)
                procs = [1 + SD * rng.gauss(0, 1) / math.sqrt(2) + SD / math.sqrt(2) * t4(rng) for _ in range(6)]
                mean = sum(procs) / 6
                sd = math.sqrt(sum((x - mean) ** 2 for x in procs) / 5) / mean
                noise_rows.append({"session": "S1", "engine": engine, "arm": "defaults", "level": level,
                                   "model": model, "batch": b, "processes": 6, "sd": sd})
    return derive.tables(rows, derive.Noise(noise_rows))


def test_the_pipeline_calls_no_more_differences_under_a_null_than_chance_expects():
    found, expected = [], []
    for seed in range(40):
        got, stats = patterns(null_tables(seed))
        found.append(sum(1 for p in got if p["kind"] == "engines"))
        expected.append(stats["expected_by_chance"])
    mean_found, mean_expected = sum(found) / len(found), sum(expected) / len(expected)
    # heavier tails than chance() assumes, and a process's offset shared across its batches, cross the line somewhat
    # more often: the margin is what a t4 adds at K = 3 (about 2x), never an order of magnitude
    assert mean_found <= 2.5 * mean_expected + 0.25, (mean_found, mean_expected)


def test_the_a_a_check_finds_the_rate_its_k_implies():
    rng = random.Random(7)
    rows = []
    for b in (1, 8, 64):
        steps = [10.0 * (1 + SD * rng.gauss(0, 1)) for _ in range(6)]
        rows.append({"session": "S1", "engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B",
                     "batch": b, "processes": 6, "sd": SD, "process_steps": " ".join(f"{x:.6f}" for x in steps)})
    aa = derive.Noise(rows).aa()["processes"]
    assert aa["pairs"] == 45 and aa["rate"] < 0.05
    rows[0]["process_steps"] = "10 10 10 10.01 9.99 11"  # one process 10% off: its five pairs cross
    assert derive.Noise(rows).aa()["processes"]["beyond"] >= 5
    assert derive.Noise([{**r, "process_steps": ""} for r in rows]).aa() is None


def test_the_a_a_check_is_not_judged_by_the_noise_of_the_pairs_it_judges():
    """Judged by the SD of the same processes, two of six can never sit K x sqrt 2 x s apart (at most s x sqrt 10):
    the old check read 0 by construction. Left two out, a heavy-tailed process crosses; and each session's sentinel
    end reading is judged against its start at the first session's noise, never the noise check() scaled by them."""
    tails = [10.0, 10.02, 9.98, 10.01, 9.99, 10.4]  # one process 4% off five that agree within 0.2%
    row = {"session": "S1", "engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B",
           "batch": 1, "processes": 6, "sd": 0.016, "process_steps": " ".join(map(str, tails))}
    assert derive.Noise([row]).aa()["processes"]["beyond"] == 5
    sentinel = {"engine": "vllm", "level": "L1", "arm": "defaults", "model": "Qwen/Qwen3-0.6B",
                "batch": 1, "valid": "True"}
    readings = [{**sentinel, "session": "S2", "position": "start", "step_ms": 10.0},
                {**sentinel, "session": "S2", "position": "end", "step_ms": 10.5},  # 5% apart at 0.2% noise
                {**sentinel, "session": "S3", "position": "start", "step_ms": 10.0},
                {**sentinel, "session": "S3", "position": "end", "step_ms": 10.01}]
    noise = derive.Noise([{**row, "sd": 0.002, "process_steps": ""}], readings)
    noise.check([])  # scales S2's noise by its own spread: the A/A rate must not read that
    assert noise.aa()["sentinels"] == {"pairs": 2, "beyond": 1, "rate": 0.5}
