import pytest

from servebench.physics import H100_SXM, PEAKS, floors, load_shape, shape_of

# (parameters in billions, weight bytes one step streams in GB, KV bytes per token), from the HF configs committed
# under configs/models. Counting all parameters gives 16.4 / 29.5 / 65.5 GB for 8B / 14B / 32B: that includes the
# untied input embedding, which a decode step only gathers B rows of.
QWEN3 = {
    "Qwen3-0.6B": (0.596, 1.19, 114_688),
    "Qwen3-1.7B": (1.721, 3.44, 114_688),
    "Qwen3-4B": (4.022, 8.04, 147_456),
    "Qwen3-8B": (8.191, 15.14, 147_456),
    "Qwen3-14B": (14.768, 27.98, 163_840),
    "Qwen3-32B": (32.762, 63.97, 262_144),
}


@pytest.mark.parametrize("model", QWEN3)
def test_qwen3_counts(model):
    params, weight_gb, kv = QWEN3[model]
    s = load_shape(f"Qwen/{model}")
    assert s.params / 1e9 == pytest.approx(params, abs=0.001)
    assert s.weight_bytes / 1e9 == pytest.approx(weight_gb, abs=0.005)
    assert s.kv_bytes_per_token == kv


def test_tied_counts_the_one_matrix_once():
    tied, untied = load_shape("Qwen3-4B"), load_shape("Qwen3-8B")
    assert tied.tied and not untied.tied
    assert tied.resident_bytes == tied.weight_bytes  # the LM head is the embedding
    assert untied.resident_bytes - untied.weight_bytes == untied.head_params * 2  # the input table: held, not streamed


def test_weight_read_floors_at_the_datasheet_bandwidth():
    """Weight-read floor at 3.35 TB/s and empty context: 0.36 ms (0.6B) ... 19.09 ms (32B)."""
    for model, ms in {"Qwen3-0.6B": 0.36, "Qwen3-8B": 4.52, "Qwen3-14B": 8.35, "Qwen3-32B": 19.09}.items():
        s = load_shape(model)
        assert s.weight_bytes / H100_SXM.bandwidth * 1e3 == pytest.approx(ms, abs=0.01)


def test_step_bytes_and_flops_at_context():
    s = load_shape("Qwen3-8B")
    contexts = [1024] * 64
    assert s.step_bytes(contexts) == s.weight_bytes + 64 * 1025 * 147_456
    # 64 sequences at 1k context read about 9.7 GB of KV
    assert (s.step_bytes(contexts) - s.weight_bytes) / 1e9 == pytest.approx(9.7, abs=0.05)
    dense = 2 * (s.body_params + s.head_params) * 64
    assert s.step_flops(contexts) == dense + 4 * 36 * 4096 * 64 * 1025


def test_intensity_is_about_the_batch():
    """Weight matmuls do ~2 FLOP per 2 bytes per sequence, so intensity is roughly B at short context."""
    s = load_shape("Qwen3-32B")
    f = floors(s, [1] * 64)
    assert f["intensity"] == pytest.approx(64, rel=0.05)


def test_max_batch_at_the_window_peak_context():
    """At 1,184 tokens and 0.9 x 80 GB, 8B fits ~318, 14B ~218, 32B ~20."""
    usable = 0.9 * 80e9
    assert load_shape("Qwen3-8B").max_batch(1184, usable) == 318
    assert load_shape("Qwen3-14B").max_batch(1184, usable) == 218
    assert load_shape("Qwen3-32B").max_batch(1184, usable) == 20
    assert load_shape("Qwen3-0.6B").max_batch(1184, usable) >= 256
    assert load_shape("Qwen3-32B").max_batch(1184, 60e9) == 0


def test_llama_second_family():
    assert load_shape("Llama-3.2-1B-Instruct").params / 1e9 == pytest.approx(1.236, abs=0.001)
    assert load_shape("Llama-3.2-1B-Instruct").kv_bytes_per_token == 32_768  # head_dim 64
    assert load_shape("Llama-3.1-8B-Instruct").params / 1e9 == pytest.approx(8.030, abs=0.001)


def test_moe_and_incomplete_configs_are_refused():
    assert shape_of({"num_experts": 128, "num_hidden_layers": 48}) is None
    assert shape_of({"hidden_size": 8}) is None


def test_peaks():
    assert H100_SXM.form == "SXM5"
    assert H100_SXM.flops == 989.5e12 and H100_SXM.bandwidth == 3.35e12
    assert PEAKS["NVIDIA H100 PCIe"].bandwidth == 2.0e12  # never mixed into the SXM charts
