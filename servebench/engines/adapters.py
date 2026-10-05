"""The four engines' command-line conventions. Each turns a merged arm (servebench.engines.base) into the server,
offline-driver or runner command its pinned image runs.

- vLLM: engine args as --kebab-case; a false boolean is --no-<flag> (argparse BooleanOptionalAction), a mapping is
  compact JSON (--compilation-config '{"cudagraph_mode":"NONE"}').
- SGLang: ServerArgs as --kebab-case; a true boolean is a bare flag, a false one is left out (store_true).
- TensorRT-LLM: trtllm-serve with the LLM API mapping as --extra_llm_api_options YAML, written into the cell's /out.
- llama.cpp: literal flags; the server's -np and -c are set per unit (slots = the largest batch, context per slot =
  prompt + generated + margin), and its thread pool to the engine's cores.

Every adapter takes the same arguments (the arm, the unit's plan, the port, the engine's core count); each uses what
its engine has a setting for.
"""
import json

import yaml

from servebench.containers import OUT
from servebench.engines.base import Engine, Plan, kebab

SLOT_MARGIN = 64
HOST = "127.0.0.1"  # every server on loopback: its only client is on the box (AIPerf, the stream client, the health
# check; nsys start and stop go through exec), and a rented VM's port open to the internet would let anyone's requests
# share the GPU it measures (vLLM's own default is every interface)


def repeated(plan):
    """Each batch `repeats` times in a row (at least once): L0 runners take a list and run every entry, so an L0 cell
    gets a median and a range like the others (servebench.runners)."""
    return [b for b in plan.batches for _ in range(max(plan.repeats, 1))]


def _flags(args, false_prefix):
    out = []
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                out.append(kebab(k))
            elif false_prefix:
                out.append("--no-" + k.replace("_", "-"))
        elif isinstance(v, (dict, list)):
            out += [kebab(k), json.dumps(v, sort_keys=True, separators=(",", ":"))]
        else:
            out += [kebab(k), str(v)]
    return out


class VLLM(Engine):
    name = "vllm"
    driver = "vllm_offline.py"

    def server_cmd(self, arm, plan, port, threads=None):
        return ["vllm", "serve", plan.model, "--host", HOST, "--port", str(port), *_flags(arm.args, True)]


class SGLang(Engine):
    name = "sglang"
    driver = "sglang_offline.py"

    def server_cmd(self, arm, plan, port, threads=None):
        return ["python3", "-m", "sglang.launch_server", "--model-path", plan.model, "--host", HOST,
                "--port", str(port), *_flags(arm.args, False)]

    def runner_cmd(self, arm, plan, threads=None):
        """L0: sglang.benchmark.one_batch drives the ModelRunner directly: no scheduler, no overlap."""
        runner_args = {"mem_fraction_static", "context_length", "cuda_graph_max_bs_decode",
                       "cuda_graph_backend_decode", "cuda_graph_backend_prefill", "random_seed"}
        keep = {k: v for k, v in arm.args.items() if k in runner_args}
        cmd = ["python3", "-m", "sglang.benchmark.one_batch", "--model-path", plan.model,
               "--batch-size", *map(str, repeated(plan)), "--input-len", str(plan.prompt_len),
               "--output-len", str(plan.gen_len), "--result-filename", f"{OUT}/one_batch.jsonl", *_flags(keep, False)]
        return cmd


class TRTLLM(Engine):
    name = "trtllm"
    driver = "trtllm_offline.py"
    options = f"{OUT}/llm_api_options.yaml"

    def server_files(self, arm):
        """Files the server reads from /out, written by the runner before the container starts."""
        return {"llm_api_options.yaml": yaml.safe_dump(arm.args, sort_keys=True)}

    def server_cmd(self, arm, plan, port, threads=None):
        return ["trtllm-serve", "serve", plan.model, "--backend", "pytorch", "--host", HOST, "--port", str(port),
                "--extra_llm_api_options", self.options]


class LlamaCpp(Engine):
    name = "llamacpp"
    driver = None

    @staticmethod
    def _literal(args):
        out = []
        for k, v in args.items():
            if v is None or v is False:
                continue
            out += [k] if v is True else [k, str(v)]
        return out

    @staticmethod
    def _threads(threads):
        """-t and -tb at the cores the engine runs on, so the pool matches what it may use: llama.cpp's own default
        counts every CPU the host shows (64 threads on a pod pinned to 4 cores; there, 1.6% of the step or less at 64
        and 256 slots)."""
        return ["-t", str(threads), "-tb", str(threads)] if threads else []

    def server_cmd(self, arm, plan, port, threads=None):
        """plan.model is the GGUF path; a slot for each sequence of the largest batch, each holding prompt and
        generation (a server per batch, server_per_batch, has exactly a slot per sequence of its batch)."""
        slots = max(plan.batches)
        server = self.release.extra.get("server_args") or {}
        return ["/app/llama-server", "-m", plan.model, "--host", HOST, "--port", str(port), "-np", str(slots),
                "-c", str(slots * (plan.prompt_len + plan.gen_len + SLOT_MARGIN)), *self._threads(threads),
                *self._literal(arm.args), *self._literal(server)]

    def runner_cmd(self, arm, plan: Plan, threads=None):
        """L0: llama-batched-bench feeds random tokens with no sampling and synchronises every step."""
        ctx = max(plan.batches) * (plan.prompt_len + plan.gen_len + SLOT_MARGIN)
        return ["/app/llama-batched-bench", "-m", plan.model, "-npp", str(plan.prompt_len), "-ntg", str(plan.gen_len),
                "-npl", ",".join(map(str, [min(plan.batches), *repeated(plan)])), "-c", str(ctx), "--output-format",
                "jsonl", *self._threads(threads), *self._literal(arm.args)]


ENGINES = {e.name: e for e in (VLLM, SGLang, TRTLLM, LlamaCpp)}


def engine_for(release):
    return ENGINES[release.engine](release)
