"""L1 driver for vLLM: AsyncLLM with the arm's engine arguments, the scheduler and async scheduling exactly as
`vllm serve` runs them, minus HTTP and detokenisation (detokenize=False, token-id prompts). Output chunks are DELTA
outputs, one per engine step per sequence. API checked against the v0.30.0 image: AsyncLLM.from_engine_args,
generate(prompt, sampling_params, request_id), async start_profile/stop_profile (profiler_config.profiler=cuda calls
cudaProfilerStart/Stop in the worker), SamplingParams(output_kind=DELTA, detokenize, ignore_eos). The KV capacity is
cache_config.num_gpu_blocks, which the engine-core handshake writes back into the frontend's config
(v1/engine/core_client.py).

    python3 servebench/drivers/vllm_offline.py --engine-args '{...}' --model Qwen/Qwen3-8B --batches 1,8 --out o.jsonl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench.drivers.common import main  # noqa: E402


class VLLMAdapter:
    def __init__(self, a):
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.inputs import TokensPrompt
        from vllm.sampling_params import RequestOutputKind
        from vllm.v1.engine.async_llm import AsyncLLM

        self.SamplingParams, self.TokensPrompt, self.DELTA = SamplingParams, TokensPrompt, RequestOutputKind.DELTA
        self.args = AsyncEngineArgs(model=a.model, **a.engine_args)
        self.llm = AsyncLLM.from_engine_args(self.args)

    async def stream(self, prompt_ids, gen_len, rid):
        # no min_tokens: in 0.30.0 it adds EOS to the stop set and turns on a per-step logits-bias pass outside the
        # graph (sampling_params.py, v1/worker/gpu/sample/sampler.py); ignore_eos with max_tokens fixes the length
        params = self.SamplingParams(max_tokens=gen_len, ignore_eos=True, temperature=0.0, detokenize=False,
                                     output_kind=self.DELTA)
        n = 0
        async for out in self.llm.generate(self.TokensPrompt(prompt_token_ids=prompt_ids), params, rid):
            n += len(out.outputs[0].token_ids)
            yield n

    async def tokens(self, prompt_ids, n, rid):
        """The n greedy token ids one prompt generates (the agreement check: servebench.drivers.common.probe)."""
        params = self.SamplingParams(max_tokens=n, temperature=0.0, detokenize=False, output_kind=self.DELTA)
        ids = []
        async for out in self.llm.generate(self.TokensPrompt(prompt_token_ids=prompt_ids), params, rid):
            ids += out.outputs[0].token_ids
        return ids

    async def start_profile(self):
        await self.llm.start_profile()

    async def stop_profile(self):
        await self.llm.stop_profile()

    def info(self):
        cfg = self.llm.vllm_config
        cache = cfg.cache_config
        blocks = getattr(cache, "num_gpu_blocks", None)
        return {"num_gpu_blocks": blocks, "block_size": cache.block_size,
                "kv_tokens": blocks * cache.block_size if blocks else None,
                "cudagraph_mode": str(getattr(cfg.compilation_config, "cudagraph_mode", None)),
                "cudagraph_capture_sizes": getattr(cfg.compilation_config, "cudagraph_capture_sizes", None),
                "async_scheduling": getattr(cfg.scheduler_config, "async_scheduling", None),
                "max_num_seqs": cfg.scheduler_config.max_num_seqs,
                "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens}

    def close(self):
        self.llm.shutdown()


if __name__ == "__main__":
    main(VLLMAdapter)
