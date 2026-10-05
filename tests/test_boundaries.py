"""Rules the layout promises (README.md): the shared core never imports the study, and the study's one headline rule
is the grid's."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "servebench").rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_servebench_never_imports_anatomy(path):
    for node in ast.walk(ast.parse(path.read_text())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
            [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        assert not any(n == "anatomy" or n.startswith("anatomy.") for n in names), f"{path} imports {names}"


def test_the_headline_level_is_the_grids():
    from anatomy.grid import levels_for
    from anatomy.metrics import HEADLINE
    from servebench.engines.base import Release

    for d in sorted((ROOT / "configs" / "engines").glob("*/*")):
        release = Release.load(d.parent.name, d.name)
        assert levels_for("headline", release) == [HEADLINE[release.engine]], d


def test_a_torn_last_line_is_skipped(tmp_path):
    from servebench.results import read_jsonl
    p = tmp_path / "driver.jsonl"
    p.write_text('{"kind": "info"}\n{"kind": "long", "batch": 1}\n{"kind": "lo')  # killed mid-write
    assert [r["kind"] for r in read_jsonl(p)] == ["info", "long"]


def test_the_acceptance_numbers_are_configs_tolerances():
    """One place for when a cell counts: configs/tolerances.yaml, never constants scattered in the code."""
    import yaml

    from anatomy import metrics, noise, tidy
    t = yaml.safe_load((metrics.CONFIGS / "tolerances.yaml").read_text())
    assert (metrics.AGREE, metrics.RANGE_OK, metrics.OVERHEAD_OK, metrics.CPU_THROTTLED, metrics.BEYOND_K,
            metrics.NOISE_FLOOR, tidy.BUSY_INVARIANT, tidy.HOSTLOAD_STRETCH, tidy.STEP_NOISE,
            tidy.ANCHOR_RATIO, tidy.MIN_STEPS, tidy.MISSED_ANCHOR) == \
        (t["agree"], t["range_ok"], t["overhead_ok"], t["cpu_throttled"], t["beyond_k"], t["noise_floor"],
         t["busy_invariant"], t["hostload_stretch"], t["step_noise"], tuple(t["anchor_ratio"]),
         t["min_steps"], t["missed_anchor"])
    assert noise.BANDS == {k: tuple(v) for k, v in t["bands"].items()}   # the rule's bands, read from the same file
    assert all(lo <= t[k] <= hi for k, (lo, hi) in noise.BANDS.items())  # every set value inside its band


def test_no_fixture_carries_a_machine_s_environment():
    """A real Nsight export stores the recording process's whole environment (tokens, home paths, ssh peers): a fixture
    keeps only the tables its test reads."""
    root = Path(__file__).parent / "fixtures"
    leaks = (b"HOME=", b"TOKEN", b"SSH_CONNECTION", b"SSH_CLIENT", b"/home/", b"USER=")
    for f in (p for p in root.rglob("*") if p.is_file()):
        data = f.read_bytes()
        assert not [x for x in leaks if x in data], f.relative_to(root)
