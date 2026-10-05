"""The PDL chain's host side (ruler/pdl.py): its reader against synthetic exports and truth, the skip paths, and the
GPU module importing where there is no CUDA or Triton (the GPU half runs in the ruler's image)."""
import json

import pytest

from ruler import pdl
from tests.nsysfake import fake_trace

FLOOR_US, WORK_NS, NODES = 1.0, 10_000, 20


def chain_export(path, overlap_ns, nodes=NODES, replays=3):
    """Replays of a chain whose every node but the first starts overlap_ns before its predecessor ends."""
    kernels, t = [], 0
    for r in range(replays):
        for j in range(nodes):
            start = t - overlap_ns if j else t
            kernels.append((start, start + WORK_NS, "_node", {"graphNodeId": j, "graphId": 1, "correlationId": r}))
            t = start + WORK_NS
        t += 50_000  # the graph's launch gap: no overlap across replays
    return fake_trace(path, kernels=kernels)


def write_run(folder, tails=(500, 2000), overlaps=None, pdl_device=10_300.0, nopdl_device=10_850.0, work=10_050.0):
    traced = {"mode": "node", "windows": [{"variant": "nopdl", "tail_ns": 0, "chain_ok": True}]}
    untraced = {"mode": "none", "windows": [{"variant": "nopdl", "tail_ns": 0, "chain_ok": True,
                                              "node_device_ns": nopdl_device, "work_ns": work}]}
    chain_export(folder / "pdl-node.1.sqlite", 0)
    for i, tail in enumerate(tails, 2):
        traced["windows"].append({"variant": "pdl", "tail_ns": tail, "chain_ok": True})
        untraced["windows"].append({"variant": "pdl", "tail_ns": tail, "chain_ok": True,
                                    "node_device_ns": pdl_device, "work_ns": work})
        chain_export(folder / f"pdl-node.{i}.sqlite", (overlaps or {}).get(tail, tail))
    (folder / "pdl-node.json").write_text(json.dumps(traced))
    (folder / "pdl-none.json").write_text(json.dumps(untraced))
    (folder / "launch_latency.json").write_text(json.dumps({"graph_node_us": FLOOR_US}))


def test_the_model_is_priced_against_the_untraced_truth(tmp_path):
    write_run(tmp_path, tails=(300, 500, 2000))
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert c["pdl_exposed_ns_per_node"] == pytest.approx(250)  # 10,300 - 10,050
    assert c["nopdl_exposed_ns_per_node"] == pytest.approx(800)
    assert c["pdl_tail_ns"] == 500 and c["pdl_overlap_ns_traced"] == 500  # the median overlap's window
    assert c["pdl_floor_ns"] == 1000 and c["pdl_model_ns"] == 500  # 1000 - 500
    assert c["pdl_model_error_ns"] == pytest.approx(250)
    assert c["pdl_inflation_ns"] == 100 and c["pdl_model_corrected_ns"] == 600  # 1000 - (500 - 100)
    assert c["pdl_model_corrected_error_ns"] == pytest.approx(350)
    assert c["pdl_overlapped_share"] == pytest.approx((NODES - 1) / NODES)
    assert [w["tail_ns"] for w in c["pdl_windows"]] == [300, 500, 2000] and c["pdl_chain_ok"]
    assert c["pdl_windows"][2]["pdl_model_ns"] == 0  # the bare model charges an overlap past the floor nothing
    # the release: the smallest exposed time over the windows (every window reads 250 here), what the floored model
    # charges at least: on the plateau it fits by construction, below it the bare charge stands where it is larger
    assert c["pdl_release_ns"] == pytest.approx(250) and c["pdl_release_tail_ns"] == 300
    assert [w["pdl_model_floored_ns"] for w in c["pdl_windows"]] == [700, 500, 250]
    assert [w["pdl_model_floored_error_ns"] for w in c["pdl_windows"]] == pytest.approx([450, 250, 0])
    assert c["pdl_windows"][2]["pdl_model_floored_corrected_ns"] == 250  # 1000 - (2000 - 100) < 0: the release
    assert c["pdl_model_floored_error_ns"] == pytest.approx(250)  # the reference window's (tail 500)


def test_the_truth_is_the_untraced_passes_median_and_their_spread_is_recorded(tmp_path):
    write_run(tmp_path, tails=(500, 2000))
    one = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert one["pdl_truth_passes"] == 1 and "pdl_truth_spread_ns" not in one
    truth = json.loads((tmp_path / "pdl-none.json").read_text())
    shifted = lambda by: [{**w, "node_device_ns": w["node_device_ns"] + by} for w in truth["windows"]]  # noqa: E731
    truth["repeats"] = [shifted(40), shifted(-20)]  # two more passes: exposed 290 and 230 beside the first's 250
    (tmp_path / "pdl-none.json").write_text(json.dumps(truth))
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert c["pdl_truth_passes"] == 3 and c["pdl_truth_spread_ns"] == pytest.approx(60)
    assert c["pdl_exposed_ns_per_node"] == pytest.approx(250) and c["nopdl_exposed_ns_per_node"] == pytest.approx(800)
    assert all(w["pdl_exposed_spread_ns"] == pytest.approx(60) for w in c["pdl_windows"])


def test_the_traced_overlap_is_the_raw_one(tmp_path):
    write_run(tmp_path, tails=(1000,), overlaps={1000: 1100})  # tracing added 100 ns to the overlap
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert c["pdl_overlap_ns_traced"] == 1100 and c["pdl_model_ns"] == 0 and c["pdl_model_corrected_ns"] == 0


def test_without_the_inflation_only_the_raw_model_is_reported(tmp_path):
    write_run(tmp_path, tails=(500,))
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, None)  # no busy windows in this folder: no inflation
    assert c["pdl_model_ns"] == 500 and not any("corrected" in k or k == "pdl_inflation_ns" for k in c)


def test_a_broken_chain_is_reported(tmp_path):
    write_run(tmp_path, tails=(500,))
    truth = json.loads((tmp_path / "pdl-none.json").read_text())
    truth["windows"][1]["chain_ok"] = False
    (tmp_path / "pdl-none.json").write_text(json.dumps(truth))
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, 100)["pdl_chain_ok"] is False


def test_a_step_that_did_not_run_reads_empty(tmp_path):
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None) == {}


@pytest.mark.parametrize("which", ["pdl-node.json", "pdl-none.json"])
def test_a_skipped_step_says_why(tmp_path, which):
    (tmp_path / which).write_text(json.dumps({"mode": "node", "skipped": "sm_80: needs sm_90 or later"}))
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None) == {"pdl": "skipped: sm_80: needs sm_90 or later"}


def test_half_a_run_is_incomplete_not_a_figure(tmp_path):
    write_run(tmp_path, tails=(500,))
    (tmp_path / "pdl-none.json").unlink()
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None)["pdl"].startswith("incomplete")
    (tmp_path / "pdl-none.json").write_text(json.dumps({"windows": []}))
    assert pdl.pdl_calibration(tmp_path, None, None)["pdl"].startswith("incomplete")  # no empty-node cost to price


def test_the_gpu_half_imports_and_skips_without_cuda_or_triton(tmp_path):
    kernel, why = pdl.build_kernel()
    assert kernel is not None or why  # a kernel, or the reason: never a raise
    assert pdl.windows_spec([500, 1000]) == [("nopdl", 0, 1), ("pdl", 500, 1), ("pdl", 1000, 1)]
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("a GPU is present: the skip path is for boxes without one")
    pdl.main(["--mode", "none", "--out", str(tmp_path)])
    assert json.loads((tmp_path / "pdl-none.json").read_text())["skipped"] == "no CUDA device"
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None)["pdl"] == "skipped: no CUDA device"


def test_a_timer_too_coarse_for_a_tail_drops_that_window_and_an_export_short_stops_it(tmp_path):
    """%globaltimer's step is the implementation's: a tail under TICKS_PER_TAIL steps of it is not the tail the run
    asked for, so that window is left out, and with every tail too short the calibration is refused. The exports pair
    with the windows by order: one missing would move every tail onto another's trace."""
    write_run(tmp_path, tails=(500, 2000))
    for name, tick in (("pdl-node.json", 32), ("pdl-none.json", 400)):  # the untraced run's timer the coarser
        rec = json.loads((tmp_path / name).read_text())
        (tmp_path / name).write_text(json.dumps({**rec, "timer_step_ns": tick}))
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, None)
    assert [w["tail_ns"] for w in c["pdl_windows"]] == [2000] and c["pdl_timer_step_ns"] == 400
    rec = json.loads((tmp_path / "pdl-none.json").read_text())
    (tmp_path / "pdl-none.json").write_text(json.dumps({**rec, "timer_step_ns": 1000}))
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None)["pdl"].startswith("refused: %globaltimer steps 1000 ns")
    (tmp_path / "pdl-node.3.sqlite").unlink()
    assert pdl.pdl_calibration(tmp_path, FLOOR_US, None)["pdl"] == "incomplete: 2 node exports for 3 traced windows"


def test_the_chain_prices_its_windows_with_the_cells_own_function(tmp_path):
    write_run(tmp_path, tails=(500,), overlaps={500: 50})  # an overlap shorter than node tracing's inflation
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    # a copy of the model once charged 1,050 here: it did not clamp the inflated overlap at zero as the cells do
    assert c["pdl_model_ns"] == 950 and c["pdl_model_corrected_ns"] == 1000


def test_the_release_is_a_plateau_of_windows_whose_dependence_held(tmp_path):
    write_run(tmp_path, tails=(500, 1000, 2000))
    truth = json.loads((tmp_path / "pdl-none.json").read_text())
    truth["windows"][3] |= {"chain_ok": False, "node_device_ns": 10_100.0}  # the longest tail let a node run early:
    (tmp_path / "pdl-none.json").write_text(json.dumps(truth))              # its 50 ns exposed are not the release
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert c["pdl_release_ns"] == pytest.approx(250) and c["pdl_chain_ok"] is False
    assert [w["pdl_model_floored_ns"] for w in c["pdl_windows"]] == [500, 250, 250]
    truth["windows"][3]["chain_ok"] = True  # held, but the two longest tails disagree: no plateau, no release
    (tmp_path / "pdl-none.json").write_text(json.dumps(truth))
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert "pdl_release_ns" not in c and "disagree" in c["pdl_release_note"]
    assert not any("floored" in k for w in c["pdl_windows"] for k in w)
    one = tmp_path / "one"
    one.mkdir()
    write_run(one, tails=(500,))
    assert "one PDL window" in pdl.pdl_calibration(one, FLOOR_US, 100)["pdl_release_note"]


def test_the_wide_windows_report_how_the_release_grows_with_the_grid_beside_the_release(tmp_path):
    write_run(tmp_path, tails=(500, 2000))
    for grid, device, n in ((264, 10_600.0, 4), (8, 10_400.0, 5)):  # written out of grid order: the summary sorts
        for name, more in (("pdl-node.json", {}), ("pdl-none.json", {"node_device_ns": device, "work_ns": 10_050.0})):
            rec = json.loads((tmp_path / name).read_text())
            rec["windows"].append({"variant": "pdl-wide", "tail_ns": 2000, "grid": grid, "chain_ok": True, **more})
            (tmp_path / name).write_text(json.dumps(rec))
        chain_export(tmp_path / f"pdl-node.{n}.sqlite", 2000)
    c = pdl.pdl_calibration(tmp_path, FLOOR_US, 100)
    assert c["pdl_release_ns"] == pytest.approx(250) and [w["tail_ns"] for w in c["pdl_windows"]] == [500, 2000]
    assert [(w["grid"], round(w["pdl_wide_exposed_ns_per_node"])) for w in c["pdl_wide_windows"]] == [(8, 350),
                                                                                                        (264, 550)]
    assert c["pdl_wide_exposed_ns_per_node"] == pytest.approx(550) and c["pdl_wide_grid"] == 264  # the largest grid
    assert pdl.windows_spec([500], [8, 264]) == [("nopdl", 0, 1), ("pdl", 500, 1), ("pdl-wide", 2000, 8),
                                                 ("pdl-wide", 2000, 264)]
    assert pdl.wide_grids("8,64,2sm,1024,1,", 132) == [8, 64, 264, 1024] and pdl.wide_grids("", 132) == []
