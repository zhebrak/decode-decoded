"""L1 driver for SGLang: sgl.Engine with the arm's server arguments, so the scheduler, the overlap scheduler and the
detokeniser process run exactly as under launch_server, minus HTTP. skip_tokenizer_init with input_ids prompts keeps
tokenisation and detokenisation out of the loop. (sglang.benchmark.one_batch is not this level: it drives the
ModelRunner with no scheduler and no overlap; it is the L0 runner.)

The Engine owns its event loop (it creates one at construction when none is running), so the sweep runs on
engine.loop, and the profile range is opened through the tokenizer manager's async start_profile with the
CUDA_PROFILER activity (cudaProfilerStart/Stop in the scheduler process). API checked against the v0.5.20 image.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench.drivers.common import main  # noqa: E402


class SGLangAdapter:
    def __init__(self, a):
        import sglang as sgl

        args = {"skip_tokenizer_init": True, **a.engine_args}
        self.engine = sgl.Engine(model_path=a.model, **args)
        self._info = self._collect_info()

    async def stream(self, prompt_ids, gen_len, rid):
        # no min_new_tokens: it keeps BatchedMinNewTokensPenalizer active, a B x vocab pass every step
        params = {"max_new_tokens": gen_len, "ignore_eos": True, "temperature": 0.0}
        gen = await self.engine.async_generate(input_ids=prompt_ids, sampling_params=params, stream=True, rid=rid)
        async for chunk in gen:
            yield chunk["meta_info"]["completion_tokens"]

    async def tokens(self, prompt_ids, n, rid):
        """The n greedy token ids one prompt generates, whole (stream=False: output_ids; skip_tokenizer_init returns
        ids, no text)."""
        params = {"max_new_tokens": n, "temperature": 0.0}
        out = await self.engine.async_generate(input_ids=prompt_ids, sampling_params=params, stream=False, rid=rid)
        return list(out["output_ids"])

    async def _profile(self, start):
        from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType

        kind = ProfileReqType.START_PROFILE if start else ProfileReqType.STOP_PROFILE
        req = ProfileReq(req_type=kind, activities=["CUDA_PROFILER"]) if start else ProfileReq(req_type=kind)
        await self.engine.tokenizer_manager.start_profile(req) if start else \
            await self.engine.tokenizer_manager.stop_profile()

    async def start_profile(self):
        await self._profile(True)

    async def stop_profile(self):
        await self._profile(False)

    def _collect_info(self):
        """Read at construction, before the sweep runs on engine.loop: Engine.get_server_info() itself calls
        loop.run_until_complete and fails inside the running loop (v0.5.20 engine.py). The scheduler's init result and
        the resolved server args hold the same values."""
        try:
            sched = dict(self.engine._scheduler_init_result.scheduler_infos[0])
            sa = self.engine.tokenizer_manager.server_args
        except Exception as e:  # informational, but the KV-capacity guard reads it: say why it is missing
            return {"error": repr(e)}
        keep = ("max_running_requests", "cuda_graph_max_bs_decode", "cuda_graph_backend_decode",
                "disable_overlap_schedule", "chunked_prefill_size", "mem_fraction_static", "stream_interval")
        kv = sched.get("max_total_num_tokens")
        return {"max_total_num_tokens": kv, "kv_tokens": kv, **{k: getattr(sa, k, None) for k in keep}}

    def info(self):
        return self._info

    def run(self, coro):
        return self.engine.loop.run_until_complete(coro)

    def close(self):
        self.engine.shutdown()


if __name__ == "__main__":
    main(SGLangAdapter)
