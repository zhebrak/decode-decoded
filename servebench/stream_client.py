"""A minimal streaming OpenAI-completions client, standard library only: one thread per request, every SSE chunk
timestamped. It is not the timing client (AIPerf is, with its multi-process design); it drives the L2 profile waves,
one wave inside one Nsight capture (anatomy.served), every stream's chunks timestamped so the wave's own cadence is
known. Threads, not asyncio: a single-process asyncio client queues under load and inflates latency, and B <= 64
threads block on sockets, not on the GIL.

Each request: POST /v1/completions with prompt token ids, stream, ignore_eos, max_tokens = gen_len, temperature 0 (no
min_tokens: it turns on a per-step penalty pass in vLLM and SGLang). Tokens so far are counted per chunk by
chunk_tokens (stream_interval is 1 on every engine).
"""
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlparse


def chunk_tokens(msg):
    """Tokens one SSE completions message carries. Token ids when the server sends them; otherwise one per chunk, except
    a closing chunk with finish_reason and empty text (llama.cpp sends one; it is not a token). A mid-stream chunk with
    empty text is a token (a partial UTF-8 character), and so is a last token that arrives with finish_reason (vLLM).
    Shared with the AIPerf raw-export reader, so L2 timing and L2 profile waves count alike."""
    choice = (msg.get("choices") or [{}])[0]
    ids = choice.get("token_ids") or (choice.get("logprobs") or {}).get("tokens")
    if ids:
        return len(ids)
    text, done = choice.get("text"), choice.get("finish_reason") is not None
    return 1 if text is not None and (text or not done) else 0


def advance(n, msg):
    """Tokens so far after one SSE message. A server sending continuous usage (stream_options; TensorRT-LLM's server
    needs it to count right while still detokenising) says so absolutely; otherwise the message's own count. A usage
    message with no choices closes the stream and is not a token. Counting by text alone undercounts wherever a server
    merges two steps into one chunk (vLLM's API server falling a step behind: 285 of 288 tokens in validation), so vLLM
    and SGLang are asked for token ids (return_token_ids), and llama-server for its own running count
    (timings_per_token: timings.predicted_n, which also covers a character held back over two tokens)."""
    if not msg.get("choices"):
        return n
    done = (msg.get("timings") or {}).get("predicted_n")
    done = done if done is not None else (msg.get("usage") or {}).get("completion_tokens")
    return max(n, done) if done is not None else n + chunk_tokens(msg)


def _post_stream(url, body, on_chunk, timeout=600, on_socket=lambda sock: None):
    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.connect()
        on_socket(conn.sock)  # retain the socket even if getresponse() detaches it (Connection: close)
        conn.request("POST", u.path or "/v1/completions", json.dumps(body), {"Content-Type": "application/json"})
        with conn.getresponse() as resp:
            if resp.status != 200:
                raise RuntimeError(f"{resp.status} {resp.read(300)!r}")
            n = 0
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                m = advance(n, json.loads(data))
                if m > n:
                    n = m
                    on_chunk(time.perf_counter_ns(), n)
            return n
    finally:
        conn.close()


class Wave:
    """B concurrent streams, each in its own thread: .chunks the timestamped chunks per stream, .errors the streams
    that failed (index, error), which the caller turns into a failed batch."""

    def __init__(self, url, prompts, gen_len, model="default", extra=None):
        self.url, self.prompts, self.gen_len, self.model = url, prompts, gen_len, model
        self.extra = extra or {}
        self.chunks = [[] for _ in prompts]
        self.errors = []
        self._lock, self._sockets = threading.Lock(), []
        self._cancelled = False

    def _socket(self, sock):
        with self._lock:
            if self._cancelled:
                sock.close()
                raise TimeoutError("wave deadline passed")
            self._sockets.append(sock)

    def cancel(self):
        """Wake blocked readers, including a server that sends bytes without ever completing an SSE line."""
        with self._lock:
            self._cancelled = True
            for sock in self._sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def _one(self, i):
        body = {"model": self.model, "prompt": self.prompts[i], "stream": True, "max_tokens": self.gen_len,
                "ignore_eos": True, "temperature": 0.0, **self.extra}
        try:
            timeout = 600 if self.deadline is None else max(0.001, min(600, self.deadline - time.monotonic()))
            _post_stream(self.url, body, lambda t, n: self.chunks[i].append((t, n)), timeout=timeout,
                         on_socket=self._socket)
        except Exception as e:  # noqa: BLE001  recorded: the caller decides
            self.errors.append((i, repr(e)))

    def start(self, timeout=None):
        self.deadline = None if timeout is None else time.monotonic() + timeout
        self.threads = [threading.Thread(target=self._one, args=(i,), daemon=True) for i in range(len(self.prompts))]
        self.t0 = time.perf_counter_ns()
        for t in self.threads:
            t.start()
        return self

    def join(self):
        try:
            for t in self.threads:
                t.join(None if self.deadline is None else max(0, self.deadline - time.monotonic()))
            if any(t.is_alive() for t in self.threads):
                raise TimeoutError("wave deadline passed")
        except BaseException:
            self.cancel()
            end = time.monotonic() + 1  # bounded socket cleanup, independent of the measurement's deadline
            for t in self.threads:
                t.join(max(0, end - time.monotonic()))
            raise
        self.t1 = time.perf_counter_ns()
        return self
