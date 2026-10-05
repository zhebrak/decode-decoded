"""A unit's record files (servebench.results): written whole or not at all, read strictly, and what makes a unit
done."""
import json

import pytest

from servebench.results import (
    ALWAYS_RUN,
    Unit,
    is_done,
    is_noise,
    is_sentinel,
    keyed_by_block,
    read_json,
    read_status,
    unit_dir,
    update_meta,
    write_json,
    write_meta,
    write_status,
)


def test_records_are_written_atomically_and_read_strictly(tmp_path):
    u = Unit(engine="vllm", version="0.30.0", arm="defaults", level="L1", model="Qwen/Qwen3-0.6B", capture="timing",
             batches=(1,))
    d = unit_dir("S1-x", u, root=tmp_path)
    write_meta(d, u, git="abc")
    update_meta(d, contention={"1": {"cpu_share": [0.9]}})
    write_status(d, True, 12.34)
    assert not list(d.glob("*.tmp"))                       # the temporary file is renamed over the record
    assert json.loads((d / "meta.json").read_text())["git"] == "abc" and is_done(d)
    assert read_status(tmp_path / "nowhere") is None and not is_done(tmp_path / "nowhere")   # absent: not done
    (d / "status.json").write_text('{"ok": tr')             # a torn record (which the atomic write never leaves)
    with pytest.raises(ValueError, match="status.json"):   # is an error with its path, never a silent default
        is_done(d)


def test_a_torn_tmp_file_never_counts(tmp_path):
    """A kill between the temporary file and the rename leaves <name>.tmp beside no record: the unit is not done."""
    (tmp_path / "status.json.tmp").write_text('{"ok": true')
    assert read_json(tmp_path / "status.json") is None and not is_done(tmp_path)
    write_json(tmp_path / "status.json", {"ok": True})
    assert is_done(tmp_path)


def test_block_predicates_keep_their_meanings():
    """Three rules, not one: a sentinel or noise block is never narrowed away or rerun (ALWAYS_RUN); a probe block
    is keyed by its block too, so its unit never takes an S1 unit's directory, but it is rerun and skipped like any
    other unit of its own grid."""
    assert is_sentinel("sentinel-start") and is_sentinel("sentinel-redo1-end") and not is_sentinel("noise-start-p1")
    assert is_noise("noise-end-p3") and not is_noise("sentinel-start") and not is_noise(None)
    assert all(keyed_by_block(b) for b in ("sentinel-start", "noise-start-p1", "probe-slots"))
    assert not keyed_by_block("main") and not keyed_by_block("")
    assert ALWAYS_RUN == ("sentinel-", "noise-") and not "probe-slots".startswith(ALWAYS_RUN)
