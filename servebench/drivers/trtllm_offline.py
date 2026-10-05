"""L1 driver for TensorRT-LLM (PyTorch backend): tensorrt_llm.LLM with the arm's LLM-API arguments, so the PyExecutor
runs with its overlap scheduler and CUDA graphs as trtllm-serve runs them, minus HTTP and detokenisation. Prompts are
token ids; outputs stream with token_ids cumulative per request. API read from the 1.2.1 image's source (the package
cannot be imported without a driver: it initialises NVML at import): LLM.generate_async(inputs, sampling_params,
streaming=True), SamplingParams(max_tokens, ignore_eos, detokenize); no temperature or top_k set, which the LLM API
decodes greedily. No min_tokens, on any engine: ignore_eos with max_tokens fixes the length without a penalty pass.

The profile range: PyExecutor's own hook (TLLM_PROFILE_START_STOP) is keyed by iteration number, which a sweep cannot
predict, so this process calls cudaProfilerStart/Stop itself. With the executor in-process that is exact; with an
MPI-spawned worker it relies on nsys capturing the whole process tree, which it does (in validation every decode step
of a 288-token generation was in the trace).
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench.drivers.common import main  # noqa: E402


def eos_id(model):
    """The model's end-of-sequence id from its config.json (the first, where it lists several)."""
    import json

    path = Path(model)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model, local_files_only=True))
    eos = json.loads((path / "config.json").read_text()).get("eos_token_id")
    return eos[0] if isinstance(eos, list) else eos


KV_POOL = re.compile(r"for max tokens in paged KV cache \((\d+)\)")


def kv_tokens(log, wait_s=30.0, poll_s=1.0):
    """(TensorRT-LLM's KV token capacity, where it came from). No LLM-API call reports it short of iteration statistics
    (enable_iter_perf_stats), which add work to every step measured, so it is read from the executor's own log: it
    logs each paged-KV allocation, an estimation pass and then the pool ("Allocated 1.77 GiB for max tokens in paged KV
    cache (16608)", then "... 71.84 GiB ... (672544)" at 1.2.1). The pool is the last
    of at least two such lines; a single one may be the estimation pass alone, whose capacity would skip batches the
    pool holds, so it gives None, as does a line that has not reached the file within wait_s (the worker's output can
    sit in its buffer). With None the guard does not act, and a batch the pool cannot hold queues: the cadence's stall
    check rejects it."""
    deadline = time.monotonic() + wait_s
    while True:
        found = KV_POOL.findall(Path(log).read_text(errors="replace")) if Path(log).exists() else []
        if len(found) >= 2:
            return int(found[-1]), "the executor's log: the paged KV pool"
        if time.monotonic() >= deadline:
            return None, f"the executor's log held {len(found)} of its two KV allocation lines"
        time.sleep(poll_s)


class TRTLLMAdapter:
    def __init__(self, a):
        from tensorrt_llm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.log = Path(a.out).with_name("container.log")  # this unit's log, beside --out, in a container or not
        self.llm = LLM(model=a.model, **a.engine_args)
        if a.profile == "cuda":
            # The executor runs the GPU work; this process otherwise never touches CUDA, so its first cudaProfilerStart
            # would create a context first and open the capture late (in validation: 0, 18 and 36 steps of 64 captured).
            # The context is made now, after the executor has taken its memory.
            import torch

            torch.cuda.init()
            torch.empty(1, device="cuda")
        # generate_async needs end_id when the LLM holds no tokenizer, as 1.2.1's can ("tokenizer is
        # required to reset end_id"); ignore_eos leaves it unused, so the model's own eos is given explicitly
        self.end_id = eos_id(a.model)

    async def stream(self, prompt_ids, gen_len, rid):
        params = self.SamplingParams(max_tokens=gen_len, ignore_eos=True, detokenize=False, end_id=self.end_id,
                                     pad_id=self.end_id)
        async for out in self.llm.generate_async(prompt_ids, sampling_params=params, streaming=True):
            yield len(out.outputs[0].token_ids)

    async def tokens(self, prompt_ids, n, rid):
        """The n greedy token ids one prompt generates (outputs[0].token_ids, the request's whole output)."""
        params = self.SamplingParams(max_tokens=n, detokenize=False, end_id=self.end_id, pad_id=self.end_id)
        out = await self.llm.generate_async(prompt_ids, sampling_params=params)
        return list(out.outputs[0].token_ids)

    async def start_profile(self):
        import torch

        torch.cuda.cudart().cudaProfilerStart()

    async def stop_profile(self):
        import torch

        torch.cuda.cudart().cudaProfilerStop()

    def info(self):
        args = getattr(self.llm, "args", None)
        keep = ("max_batch_size", "max_num_tokens", "max_seq_len", "disable_overlap_scheduler", "stream_interval")
        out = {k: getattr(args, k, None) for k in keep} if args else {}
        if args is not None:
            out["cuda_graph_config"] = repr(getattr(args, "cuda_graph_config", None))
        out["end_id"] = self.end_id
        out["tokenizer_loaded"] = getattr(self.llm, "tokenizer", None) is not None
        out["kv_tokens"], out["kv_tokens_source"] = kv_tokens(self.log)  # the driver's guard skips what it cannot hold
        return out

    def close(self):
        self.llm.shutdown()


if __name__ == "__main__":
    main(TRTLLMAdapter)
