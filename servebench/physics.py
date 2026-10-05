"""What a decode step must move and compute, from a model's config.json: the bytes and FLOPs every engine is held to,
so MBU, MFU and the speed-of-light floor are counted the same way for all of them.

Per decode step (one new token for each of B sequences, sequence i at context c_i):
- weight bytes: every matmul weight once: q, k, v, o, gate, up, down in every layer, and the LM head. The input
  embedding is a gather of B rows, not a stream, so an untied model's embedding table is left out, and a tied model's
  single matrix counts once, as the LM head (counting all parameters reads 8% high for Qwen3-8B);
- KV bytes: sequence i reads c_i tokens of K and V and writes one;
- FLOPs: 2 per matmul weight per sequence (LM head included), plus attention, 4 x layers x q_dim per (query, key)
  pair: a query at position c_i reads c_i + 1 keys (its own included).
Norm weights, activations and embedding rows are left out: a lower bound, by well under 1%.

Adapted from an earlier harness's model shapes (Shape), which already counted weights this way, extended with
per-step counts at a given context, resident memory and batch feasibility, and the GPU peak tables.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from servebench import ROOT

DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
MODELS_DIR = ROOT / "configs" / "models"


@dataclass(frozen=True)
class Shape:
    """What the byte and FLOP counts need from a dense decoder's config. q_dim and kv_dim are heads x head_dim."""
    layers: int
    hidden: int
    q_dim: int
    kv_dim: int
    intermediate: int
    vocab: int
    dtype_bytes: int
    tied: bool

    @property
    def body_params(self):
        """Matmul weights over all layers (q, k, v, o, gate, up, down): norms and biases are not matmuls."""
        h = self.hidden
        return self.layers * (2 * h * self.q_dim + 2 * h * self.kv_dim + 3 * h * self.intermediate)

    @property
    def head_params(self):
        return self.hidden * self.vocab

    @property
    def params(self):
        """All matmul and embedding parameters: an untied model holds a second vocab x hidden matrix."""
        return self.body_params + self.head_params * (1 if self.tied else 2)

    @property
    def weight_bytes(self):
        """Weight bytes one decode step streams from HBM: body and LM head, never the input embedding table."""
        return (self.body_params + self.head_params) * self.dtype_bytes

    @property
    def resident_bytes(self):
        """Weight bytes the GPU holds: what is left for the KV cache comes after these."""
        return self.params * self.dtype_bytes

    @property
    def kv_bytes_per_token(self):
        return 2 * self.layers * self.kv_dim * self.dtype_bytes  # K and V

    def step_bytes(self, contexts):
        """HBM bytes one decode step moves, for sequences at the given context lengths (tokens already cached)."""
        return self.weight_bytes + sum(c + 1 for c in contexts) * self.kv_bytes_per_token

    def step_flops(self, contexts):
        """FLOPs of one decode step: matmuls (LM head included) and attention, for the given context lengths."""
        matmul = 2 * (self.body_params + self.head_params) * len(contexts)
        return matmul + 4 * self.layers * self.q_dim * sum(c + 1 for c in contexts)

    def max_batch(self, context, memory_bytes):
        """The most sequences of `context` tokens whose KV fits beside the weights in memory_bytes: an upper bound,
        since engines also hold activations, graph pools and allocator slack (the grid takes each engine's own
        reported KV capacity on the box)."""
        return max(0, int((memory_bytes - self.resident_bytes) // (context * self.kv_bytes_per_token)))


def shape_of(config):
    """A Shape from a config.json dict, or None when it is not a dense decoder this module can count."""
    if any(config.get(k) for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        return None
    try:
        heads = config["num_attention_heads"]
        kv_heads = config.get("num_key_value_heads") or heads
        head_dim = config.get("head_dim") or config["hidden_size"] // heads
        dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16").removeprefix("torch.")
        return Shape(layers=config["num_hidden_layers"], hidden=config["hidden_size"], q_dim=heads * head_dim,
                     kv_dim=kv_heads * head_dim, intermediate=config["intermediate_size"], vocab=config["vocab_size"],
                     dtype_bytes=DTYPE_BYTES[dtype], tied=bool(config.get("tie_word_embeddings", False)))
    except KeyError:
        return None


def load_shape(model):
    """The Shape of a model: a path to a config.json, or a name whose config is committed under configs/models
    ('Qwen/Qwen3-8B' and 'Qwen3-8B' both read configs/models/Qwen3-8B.json)."""
    path = Path(model)
    if not path.is_file():
        path = MODELS_DIR / f"{str(model).rstrip('/').rsplit('/', 1)[-1]}.json"
    return shape_of(json.loads(path.read_text()))


@dataclass(frozen=True)
class Peak:
    """Datasheet peaks of one card, keyed by its nvidia-smi name. flops is dense bf16 tensor throughput (half of the
    with-sparsity figure datasheets lead with); bandwidth is HBM bytes per second; memory is what nvidia-smi reports
    as total."""
    name: str
    form: str
    flops: float
    bandwidth: float
    memory: int


_MIB = 2**20
# Sources: NVIDIA H100 datasheet and product briefs (H100 SXM 1,979 / PCIe 1,513 / NVL 1,671 bf16 TFLOPS with
# sparsity), H200 and A100 datasheets. The memory figures are nvidia-smi totals.
PEAKS = {p.name: p for p in (
    Peak("NVIDIA H100 80GB HBM3", "SXM5", 989.5e12, 3.35e12, 81_559 * _MIB),
    Peak("NVIDIA H100 PCIe", "PCIe", 756.5e12, 2.0e12, 81_559 * _MIB),
    Peak("NVIDIA H100 NVL", "NVL", 835.5e12, 3.9e12, 95_830 * _MIB),
    Peak("NVIDIA H200", "SXM", 989.5e12, 4.8e12, 143_771 * _MIB),
    Peak("NVIDIA B200", "SXM", 2250e12, 7.7e12, 183_359 * _MIB),  # bandwidth per Lenovo Press LP2226 (often "8 TB/s")
    Peak("NVIDIA A100-SXM4-80GB", "SXM4", 312e12, 2.039e12, 81_920 * _MIB),
    Peak("NVIDIA A100-SXM4-40GB", "SXM4", 312e12, 1.555e12, 40_960 * _MIB),  # a harness-validation card
)}
H100_SXM = PEAKS["NVIDIA H100 80GB HBM3"]


def floors(shape, contexts, peak=H100_SXM, bandwidth=None):
    """The speed-of-light numbers for one decode step: bytes, FLOPs, the bandwidth floor in seconds (at `bandwidth`
    when given, e.g. the achievable bandwidth measured on the box, else the datasheet peak) and arithmetic intensity."""
    moved, flops = shape.step_bytes(contexts), shape.step_flops(contexts)
    bw = bandwidth or peak.bandwidth
    return {"bytes": moved, "flops": flops, "floor_s": moved / bw, "intensity": flops / moved}
