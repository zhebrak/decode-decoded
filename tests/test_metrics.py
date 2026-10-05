import pytest

from anatomy.decompose import decompose
from anatomy.metrics import physics, spread, timing_cells


def longs(batch, steps_ns, intervals=None):
    intervals = intervals or steps_ns
    return [{"kind": "long", "batch": batch, "repeat": i, "gen_len": 288, "tokens": [288] * batch,
             "cadence": {"step_ns": s, "interval_ns": intervals[i]}} for i, s in enumerate(steps_ns)]


def test_timing_cells_agree_and_range():
    recs = [*longs(8, [10.0e6, 10.1e6, 10.2e6]), {"kind": "differential", "batch": 8, "step_ns": [10.05e6, 10.1e6]}]
    c = timing_cells(recs)[8]
    assert c["step_ms"] == pytest.approx(10.1) and c["repeats"] == 3
    assert c["estimators_agree"] and not c["rerun"] and c["step_range"] == pytest.approx(0.2 / 10.1)


def test_timing_cells_flag_disagreement_rerun_and_errors():
    recs = [*longs(1, [10e6, 11e6]), {"kind": "differential", "batch": 1, "step_ns": [12e6]},
            {"kind": "error", "batch": 256, "error": "OOM"}]
    c = timing_cells(recs)
    assert c[1]["estimators_agree"] is False and c[1]["rerun"]
    assert c[256]["step_ms"] is None and c[256]["error"] == "OOM"


def test_differential_is_judged_against_the_tail_cadence():
    """The differential covers tokens 96-288 (more context, more KV per step): compared with the whole-window cadence
    it would disagree by construction on KV-heavy cells; compared with the tail it must agree."""
    recs = [{"kind": "long", "batch": 64, "repeat": i, "gen_len": 288, "tokens": [288] * 64,
             "cadence": {"step_ns": 10.0e6, "interval_ns": 10.0e6, "stalls": 0},
             "cadence_tail": {"step_ns": 10.5e6}} for i in range(3)]
    recs.append({"kind": "differential", "batch": 64, "step_ns": [10.5e6]})
    c = timing_cells(recs)[64]
    assert c["estimators_agree"] and c["tail_ms"] == pytest.approx(10.5) and c["valid"]


def test_stalls_and_short_sequences_invalidate_the_cell():
    recs = [{"kind": "long", "batch": 8, "repeat": 0, "gen_len": 288, "tokens": [288] * 7 + [200],
             "cadence": {"step_ns": 5e6, "stalls": 2}}]
    c = timing_cells(recs)[8]
    assert not c["valid"] and len(c["problems"]) == 2 and c["stalls"] == 2


def test_skipped_batches_are_explained():
    c = timing_cells([{"kind": "skipped", "batch": 256, "reason": "needs 303104 KV tokens"}])[256]
    assert c["step_ms"] is None and c["error"].startswith("skipped") and not c["valid"]


def test_spread():
    assert spread([1, 1, 1]) == 0 and spread([None, 2]) == 0 and spread([9, 10, 11]) == pytest.approx(0.2)


def test_physics_8b_batch_64():
    row = physics("Qwen/Qwen3-8B", 64, 896, 288, step_ms=10.0, bw_achievable=3.1e12)
    kv = 64 * (1040 + 1) * 147_456
    assert row["bytes_per_step"] == pytest.approx(15.14e9 + kv, rel=1e-3)
    assert row["mbu_datasheet"] == pytest.approx(row["bytes_per_step"] / 0.01 / 3.35e12)
    assert row["mbu"] > row["mbu_datasheet"] and row["sol"] == pytest.approx(row["floor_ms"] / 10)
    assert not row["compute_leaning"]


def test_physics_without_measured_bandwidth_or_step():
    row = physics("Qwen/Qwen3-0.6B", 1, 896, 288, step_ms=None)
    assert row["floor_ms"] is None and "mbu" not in row and row["floor_ms_datasheet"] > 0.35


def test_decompose_sums_to_the_step():
    d = decompose(step_ms=2.0, occupied=0.75, kernel_ms=1.2, floor_ms=0.4)
    assert d["floor_ms"] + d["kernel_exc_ms"] + d["launch_ms"] + d["host_stall_ms"] == pytest.approx(2.0)
    assert d["host_stall_ms"] == pytest.approx(0.5) and d["bubbles_ms"] == pytest.approx(0.3) and d["consistent"]
    assert d["launch_ms"] == pytest.approx(0.3) and not d["launch_model"]


def test_calibrated_launch_term_on_the_a100_vllm_cell():
    """vLLM 0.30.0, Qwen3-0.6B, batch 1 on an A100 in validation (val-vllm-20261001): 326 kernels a step, the ruler's
    1.64 us graph node floor and 0.261 us node-trace inflation. The launch term is the kernels' launch floor plus the
    small gap."""
    d = decompose(step_ms=2.449, occupied=0.979, kernel_ms=2.427, floor_ms=0.917, kernels_per_step=326.06,
                  node_floor_us=1.64, inflation_ns=261)
    assert d["kernel_ms_corrected"] == pytest.approx(2.427 - 0.0851, abs=1e-3)
    assert d["launch_floor_ms"] == pytest.approx(0.5347, abs=1e-3) and d["bubbles_ms"] > 0 and d["launch_model"]
    assert d["launch_share"] == pytest.approx(0.24, abs=0.01) and d["consistent"]
    assert sum(d[f"{t}_ms"] for t in ("floor", "kernel_exc", "launch", "host_stall")) == pytest.approx(2.449)


def test_decompose_flags_traces_that_disagree_with_timing():
    assert decompose(step_ms=1.0, occupied=0.5, kernel_ms=0.8, floor_ms=0.3)["consistent"] is False  # bubbles < 0
    assert decompose(1.0, None, 0.5, 0.3) is None


def test_l2_is_checked_against_the_servers_own_step():
    recs = [{"kind": "long", "batch": 8, "repeat": i, "gen_len": 288, "tokens": [288] * 8,
             "cadence": {"step_ns": 6.1e6, "interval_ns": 5.8e6},
             "server_step_ns": 6.07e6, "client_whole_ns": 6.6e6} for i in range(3)]  # prefills inside: 8% above
    c = timing_cells(recs)[8]
    assert c["server_step_ms"] == pytest.approx(6.07) and c["step_ms"] == pytest.approx(6.1)
    assert c["irregularity"] == pytest.approx(6.1 / 5.8 - 1)          # reported, not a reason to reject
    assert c["estimators_agree"] is False  # the server's 6.07 against the client's 6.6 over the same requests
    for r in recs:
        r["server_step_ns"] = 6.55e6
    assert timing_cells(recs)[8]["estimators_agree"]  # same population, same clockwork: agrees; the step stays 6.1


def test_conditions_reject_faults_and_undeclared_sharing_but_not_power_capping():
    from anatomy.metrics import accepted, rejection
    base = {"step_ms": 2.0, "valid": True}
    assert accepted(base | {"slowdowns": {"sw_power_cap": 0.25}})            # the card at its limit: reported only
    hot = base | {"slowdowns": {"sw_thermal": 0.1}}
    assert not accepted(hot) and "sw_thermal in 10% of samples" in rejection(hot)
    assert not accepted(base | {"gpu_shared": True, "shared_box": False})   # a study box must be exclusive
    assert accepted(base | {"gpu_shared": True, "shared_box": True})         # a declared shared box: recorded


def test_clock_log_shares_are_over_busy_samples(tmp_path):
    from servebench.clocks import processes, summarise
    log = tmp_path / "clocks.csv"
    log.write_text("timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active\n"
                   + "t, 345, 2619, 35, 62.0, 0x0000000000000001\n" * 2      # idle: not counted
                   + "t, 1785, 2619, 50, 300.0, 0x0000000000000004\n"        # power capped
                   + "t, 1785, 2619, 50, 290.0, 0x0000000000000000\n" * 3)
    assert summarise(log)["slowdowns"] == {"sw_power_cap": 0.25}
    apps = tmp_path / "gpu-processes.csv"
    apps.write_text("timestamp,pid,process_name,used_memory\nt1, 1, x, 5\nt2, 1, x, 5\nt2, 9, y, 5\n")
    assert processes(apps) == 2 and processes(tmp_path / "missing.csv") is None


def test_a_launch_hidden_under_its_predecessor_is_charged_only_what_it_leaves_exposed():
    """A programmatic dependent launch runs its launch under the kernel before it (vLLM's 28 a step at batch 1 in
    validation, 0.75 us each): of a 1 us node cost, 0.25 us shows. Each kernel is clamped on its own: the external
    review's three kernels under a 1 us node, two overlapping by 0.1 and 1.9 us, leave 1.9 us exposed (1.0 by the
    clamped mean)."""
    from anatomy.decompose import exposed_launch_ms
    assert exposed_launch_ms(300, 1.0) == pytest.approx(0.300)
    assert exposed_launch_ms(300, 1.0, [750] * 28) == pytest.approx(0.272 + 0.007)
    assert exposed_launch_ms(10, 1.0, [2000] * 10) == 0.0   # wholly hidden: never negative
    assert exposed_launch_ms(3, 1.0, [100, 1900]) == pytest.approx(0.0019)
    assert exposed_launch_ms(3, 1.0, [100, 1900] * 4, steps=4) == pytest.approx(0.0019)  # a window of four steps
    assert exposed_launch_ms(3, 1.0, [1000, 1000]) == pytest.approx(0.001)  # full overlap: the third kernel's node
    # node tracing lengthens the predecessor, so a traced overlap reads long: 105 ns off each first
    assert exposed_launch_ms(1, 1.0, [1000], inflation_ns=105) == pytest.approx(0.000105)


def shorts(batch, n, gen=96):
    return [{"kind": "short", "batch": batch, "repeat": i, "gen_len": gen, "tokens": [gen] * batch,
             "cadence": {"stalls": 0}} for i in range(n)]


def test_a_batch_cut_short_is_invalid_with_its_reason():
    """A crash after two of three repeats, or a unit killed mid-batch, keeps the earlier waves: never an accepted cell
    with fewer repeats and no independent check."""
    info = {"kind": "info", "repeats": 3}
    full = [info, *longs(8, [5e6] * 3), *shorts(8, 3), {"kind": "differential", "batch": 8, "step_ns": [5e6] * 3}]
    assert timing_cells(full)[8]["valid"]
    crashed = [info, *longs(8, [5e6] * 2), {"kind": "error", "batch": 8, "error": "CUDA error: illegal access"}]
    c = timing_cells(crashed)[8]
    assert not c["valid"] and c["step_ms"] == pytest.approx(5)
    assert any("cut short" in p for p in c["problems"]) and any("2 of 3 long waves" in p for p in c["problems"])
    killed = [info, *longs(8, [5e6] * 3), *shorts(8, 2)]  # the last short wave never ran: no differential record
    assert timing_cells(killed)[8]["problems"] == ["2 of 3 short waves recorded", "0 of 3 differentials usable"]
    assert timing_cells(longs(8, [5e6] * 2))[8]["valid"]  # without the plan (an older record) nothing to hold it to


def test_the_views_read_one_family_at_each_engine_s_newest_release(monkeypatch):
    from anatomy import metrics
    monkeypatch.setattr(metrics, "release_ranks", lambda: {("vllm", "0.29.0"): 0, ("vllm", "0.30.0"): 1,
                                                           ("sglang", "0.5.20"): 1})
    row = {"engine": "vllm", "arm": "defaults", "level": "L1", "model": "Qwen/Qwen3-0.6B", "batch": 1,
           "step_ms": 1.0, "version": "0.30.0"}
    llama = row | {"model": "unsloth/Llama-3.2-1B-Instruct"}             # the second family: read on its own
    old = row | {"version": "0.29.0"}                                     # a release a refresh replaced
    sglang = row | {"engine": "sglang", "version": "0.5.20"}              # another engine: its own newest
    arm = row | {"arm": "graphs-off"}
    assert metrics.current_versions([old, row, sglang]) == {"vllm": "0.30.0", "sglang": "0.5.20"}
    assert metrics.current_versions([old]) == {"vllm": "0.29.0"}         # newest present, not newest listed
    assert metrics.headline_rows([row, llama, old, sglang, arm]) == [row, sglang]
    assert metrics.headline_rows([row, llama, old, sglang, arm], arms=None) == [row, sglang, arm]  # every arm




def test_a_cpu_quota_throttling_the_measured_waves_rejects_the_cell():
    """The share is of the measured waves' own quota periods (servebench.cgroup), long and short: a unit's model load
    and compile, throttled or not, are no part of it."""
    from anatomy.metrics import rejection
    recs = [*longs(8, [10.0e6, 10.0e6]), {"kind": "differential", "batch": 8, "step_ns": [10.0e6]},
            {"kind": "short", "batch": 8, "repeat": 0, "cadence": {}, "cpu": {"periods": 40, "throttled": 3}}]
    recs[0]["cpu"] = {"periods": 60, "throttled": 0}
    c = timing_cells(recs)[8]
    assert c["cpu_throttled"] == pytest.approx(0.03)
    assert "quota periods" in rejection({**c, "batch": 8})
    assert timing_cells(longs(8, [10.0e6]))[8]["cpu_throttled"] is None  # no quota: nothing to judge


def test_a_batch_never_all_decoding_says_so():
    """SGLang at max_running_requests 256 ran 255 and then the 256th alone: no all-active window in any wave."""
    from anatomy.metrics import rejection
    recs = [{"kind": "long", "batch": 256, "repeat": i, "gen_len": 288, "tokens": [288] * 256,
             "cadence": {"step_ns": None, "stalls": 0}} for i in range(3)]
    c = timing_cells(recs)[256]
    assert c["step_ms"] is None and "never had every sequence decoding at once" in rejection({**c, "batch": 256})


def test_missing_or_broken_evidence_never_makes_an_accepted_cell():
    """The external review's cases: three long waves of which one gave a cadence, with null differentials; a short
    wave that produced 1 token of its 96; a client that recorded 255 streams of a 256 batch; a repeat recorded twice; a
    NaN, an infinity, a zero or a negative step; an empty token vector. Each was accepted before; each is rejected."""
    from anatomy.metrics import accepted
    info = {"kind": "info", "repeats": 3}
    diff = {"kind": "differential", "batch": 8, "step_ns": [5e6] * 3}
    one = [info, *longs(8, [5e6, None, None]), *shorts(8, 3), {"kind": "differential", "batch": 8,
                                                              "step_ns": [None] * 3}]
    c = timing_cells(one)[8]
    assert not c["valid"] and "1 of 3 long waves gave a cadence" in c["problems"]
    assert "0 of 3 differentials usable" in c["problems"] and not accepted({**c, "model": "m"})
    lacking = shorts(8, 3)
    lacking[0]["tokens"] = [1] + [96] * 7
    c = timing_cells([info, *longs(8, [5e6] * 3), *lacking, diff])[8]
    assert c["problems"] == ["short repeats [0] did not produce gen_len tokens in every sequence"]
    dropped = longs(256, [5e6] * 3)
    dropped[1]["tokens"] = [288] * 255
    c = timing_cells([info, *dropped, *shorts(256, 3), {**diff, "batch": 256}])[256]
    assert c["problems"] == ["long repeats [1] recorded a stream count other than the batch"]
    twice = longs(8, [5e6] * 3)
    twice[2]["repeat"] = 1
    assert timing_cells([info, *twice, *shorts(8, 3), diff])[8]["problems"] == ["long repeats [1] recorded twice"]
    for bad in (float("nan"), float("inf"), 0.0, -5e6):
        c = timing_cells([info, *longs(8, [5e6, 5e6, bad]), *shorts(8, 3), diff])[8]
        assert not c["valid"] and c["step_ms"] == pytest.approx(5) and c["repeats"] == 2
    empty = longs(8, [5e6] * 3)
    empty[0]["tokens"] = []
    assert not timing_cells([info, *empty, *shorts(8, 3), diff])[8]["valid"]


def test_a_stream_of_coalesced_chunks_still_has_a_cadence():
    """A server that streams two tokens a chunk: no single-token gap to set the margin from, but the rate estimate
    never needed one; where no window exists the estimate says why."""
    from servebench.cadence import estimate
    seq = [(i * 2_000_000, i * 2) for i in range(1, 145)]   # 2 tokens every 2 ms: a 1 ms step
    e = estimate([seq, [(t + 1000, n) for t, n in seq]])
    assert e["step_ns"] == pytest.approx(1_000_000) and e["coalesced_chunks"] > 100 and e["unidentified"] is None
    assert estimate([seq[:3], seq[:3]])["unidentified"] == "fewer than two steps after the margins"
    assert estimate([[], seq])["unidentified"] == "no span where every sequence is decoding"


def test_copies_outside_any_kernel_are_their_own_part_not_launch_cost():
    """llama.cpp's logits copy to the host (3.16 ms of a 202.5 ms step at 256 slots on the pod) is busy time that is
    neither a kernel nor a launch: taken out of the gaps, the five parts still sum to the step."""
    d = decompose(step_ms=10.0, occupied=0.95, kernel_ms=8.0, floor_ms=5.0, copy_ms=1.0)
    assert d["copy_ms"] == 1.0 and d["bubbles_ms"] == pytest.approx(0.5) and d["launch_ms"] == pytest.approx(0.5)
    assert sum(d[f"{t}_ms"] for t in ("floor", "copy", "kernel_exc", "launch", "host_stall")) == pytest.approx(10.0)


def test_a_wave_whose_prompts_were_another_length_is_rejected():
    """AIPerf sends text, the server tokenises it: its own count must be the cell's prompt length."""
    info = {"kind": "info", "repeats": 1}
    long = {**longs(8, [5e6])[0], "prompt_len": 896, "prompt_tokens": [896] * 8}
    short = {**shorts(8, 1)[0], "prompt_len": 896, "prompt_tokens": [896] * 8}
    diff = {"kind": "differential", "batch": 8, "step_ns": [5e6]}
    assert timing_cells([info, long, short, diff])[8]["valid"]
    off = {**long, "prompt_tokens": [896] * 7 + [897]}
    assert timing_cells([info, off, short, diff])[8]["problems"] == [
        "long repeats [0] sent prompts of another length than prompt_len"]


def test_a_cell_read_with_a_calibration_it_could_not_check_is_not_study_data():
    from anatomy.metrics import dataset_status, row_status
    r = {"gpu": "NVIDIA H100 80GB HBM3", "ruler_from": "S2"}
    assert row_status(r) == "study"
    assert row_status({**r, "ruler_from": "S1 (unchecked)"}) == "validation"
    assert dataset_status([r, {**r, "ruler_from": "S1 (unchecked)"}]) == "validation"


def test_a_failed_unit_without_metadata_decides_nothing_about_the_tables_status():
    """S2-20261003: SGLang's runner ran out of memory at 14B before its metadata was written, and its three rows,
    with no card recorded, had made the whole study's table validation data (the page a preview, its study-only
    check skipped). Missing evidence is not evidence of another card."""
    from anatomy.metrics import STUDY_CARD, dataset_status, not_study, row_status
    study = {"gpu": STUDY_CARD, "ruler_from": "S1", "accepted": True}
    failed = {"gpu": "", "accepted": False, "rejection": "no step time"}
    assert not_study(failed) == [] and row_status(failed) is None
    assert dataset_status([study, failed]) == "study" and dataset_status([failed]) == "validation"
    assert dataset_status([study, {**failed, "gpu": "NVIDIA H100 NVL"}]) == "validation"  # a card recorded is evidence


def test_a_validation_row_says_why_it_is_not_the_studys():
    from anatomy.metrics import STUDY_CARD, not_study
    assert not_study({"gpu": STUDY_CARD, "ruler_from": "S1"}) == []
    assert not_study({"gpu": STUDY_CARD, "validation": True, "ruler_from": "S1 (unchecked)"}) == ["validation",
                                                                                                    "unchecked"]
    assert not_study({"gpu": "NVIDIA H100 NVL"}) == ["card"]
