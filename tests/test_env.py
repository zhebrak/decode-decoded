def test_the_model_revision_is_found_under_the_default_cache(tmp_path, monkeypatch):
    """Without HF_HOME the cache is ~/.cache/huggingface, ~ expanded (unexpanded, every revision reads None, and
    TRT-LLM's server is handed the repo id instead of its local snapshot)."""
    from servebench import env
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    ref = tmp_path / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/refs/main"
    ref.parent.mkdir(parents=True)
    ref.write_text("c1899de\n")
    assert env.model_revision("Qwen/Qwen3-0.6B") == "c1899de"


def test_the_engine_s_cpus_are_the_gpu_s_own_node(tmp_path, monkeypatch):
    from servebench import env
    monkeypatch.setattr(env.os, "sched_getaffinity", lambda pid: set(range(224)), raising=False)
    monkeypatch.setattr(env, "gpu_query", lambda *f, device="0": {"pci.bus_id": "00000000:18:00.0"})
    dev = tmp_path / "bus/pci/devices/0000:18:00.0"
    dev.mkdir(parents=True)
    (dev / "local_cpulist").write_text("0-55,112-167\n")
    cpus = env.gpu_local_cpus(sys_root=tmp_path)
    assert cpus[:3] == [0, 1, 2] and len(cpus) == 112 and 56 not in cpus
    monkeypatch.setattr(env, "gpu_query", lambda *f, device="0": {})  # no nvidia-smi: every allowed CPU
    assert len(env.gpu_local_cpus(sys_root=tmp_path)) == 224


def test_cgroup_reads_the_quota_and_its_throttled_periods(tmp_path):
    from servebench import cgroup
    (tmp_path / "cpu.stat").write_text("usage_usec 10\nnr_periods 50\nnr_throttled 2\nthrottled_usec 1500\n")
    (tmp_path / "cpu.max").write_text("680000 100000\n")
    assert cgroup.cpu_quota(tmp_path) == 6
    before = cgroup.cpu_stat(tmp_path)
    assert before == {"periods": 50, "throttled": 2}
    (tmp_path / "cpu.stat").write_text("usage_usec 99\nnr_periods 90\nnr_throttled 6\nthrottled_usec 9000\n")
    assert cgroup.since(before, tmp_path) == {"periods": 40, "throttled": 4}
    # a wave of 40 periods with 4 throttled and one of 60 with none: 4 of 100; no quota anywhere reads None
    assert cgroup.throttled_share([{"cpu": {"periods": 40, "throttled": 4}}, {"cpu": {"periods": 60, "throttled": 0}},
                                   {}, None]) == 0.04
    assert cgroup.throttled_share([{}, None]) is None
    (tmp_path / "cpu.max").write_text("max 100000\n")  # no quota: nothing recorded, whatever cpu.stat says
    assert cgroup.cpu_quota(tmp_path) is None and cgroup.cpu_stat(tmp_path) == {}
    assert cgroup.since(before, tmp_path) == {} and cgroup.cpu_stat(tmp_path / "none") == {}
