"""A fake OpenAI-compatible streaming server with deterministic pacing.

Used by the unit tests and to render sample reports without a GPU:

    python3 tests/fake_server.py --port 18443 --token-delay 0.004

Prompt tokens are the whitespace-separated words of the messages plus 4.
Time to first token is `ttft_base + prompt_tokens * prefill_per_token`;
each later chunk waits `token_delay * (1 + slowdown * (active - 1))`, where
`active` is the number of streams in flight (times
`1 + prompt_tokens / 1000 * decode_per_ktoken`), and carries `chunk_tokens`
tokens. Every stream carries a top-level `"capyctl_bench_sample": true`,
so capyctl-bench marks the results as sample data (meta.sample) and every
report rendered from them shows a "SAMPLE DATA" banner. GET /debug/memory prints a fake GiB figure that grows with the
prompt tokens in flight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeConfig:
    def __init__(self, ttft_base=0.004, prefill_per_token=2e-6, token_delay=0.002,
                 slowdown=0.15, chunk_tokens=1, extras=True, reasoning_tokens=0,
                 api_key=None, base_gib=10.0, gib_per_ktoken=0.05, decode_per_ktoken=0.0):
        self.ttft_base = ttft_base
        self.prefill_per_token = prefill_per_token
        self.token_delay = token_delay
        self.slowdown = slowdown
        self.chunk_tokens = chunk_tokens
        self.extras = extras
        self.reasoning_tokens = reasoning_tokens
        self.api_key = api_key
        self.base_gib = base_gib
        self.gib_per_ktoken = gib_per_ktoken
        self.decode_per_ktoken = decode_per_ktoken


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, config: FakeConfig):
        super().__init__(addr, Handler)
        self.config = config
        self.lock = threading.Lock()
        self.active = 0
        self.active_prompt_tokens = 0
        self.requests = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


def prompt_token_count(messages) -> int:
    words = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            words += len(content.split())
    return words + 4


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeServer

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        key = self.server.config.api_key
        if key and self.headers.get("Authorization") != f"Bearer {key}":
            self._send(401, b'{"error": {"message": "invalid api key"}}')
            return False
        return True

    def do_GET(self):
        if self.path == "/debug/memory":
            s = self.server
            gib = s.config.base_gib + s.active_prompt_tokens / 1000 * s.config.gib_per_ktoken
            self._send(200, f"{gib:.3f}\n".encode(), "text/plain")
            return
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._send(200, b'{"object": "list", "data": [{"id": "sample-model", "object": "model"}]}')
            return
        self._send(404, b'{"error": {"message": "not found"}}')

    def chunk(self, obj: dict) -> None:
        data = b"data: " + json.dumps(obj).encode() + b"\n\n"
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if not self._authorized():
            return
        if self.path != "/v1/chat/completions":
            self._send(404, b'{"error": {"message": "not found"}}')
            return
        req = json.loads(raw)
        cfg = self.server.config
        messages = req.get("messages") or []
        prompt_tokens = prompt_token_count(messages)
        max_tokens = int(req.get("max_tokens") or 16)
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        s = self.server
        with s.lock:
            s.active += 1
            s.active_prompt_tokens += prompt_tokens
            s.requests.append(req)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
                    "model": req.get("model", "sample-model"), "capyctl_bench_sample": True}
            time.sleep(cfg.ttft_base + prompt_tokens * cfg.prefill_per_token)
            produced = 0
            digest = hashlib.sha256()
            first = True
            while produced < max_tokens:
                if not first:
                    time.sleep(cfg.token_delay * cfg.chunk_tokens * (1 + cfg.slowdown * (s.active - 1))
                               * (1 + prompt_tokens / 1000 * cfg.decode_per_ktoken))
                first = False
                n = min(cfg.chunk_tokens, max_tokens - produced)
                text = "".join(f"w{(produced + k) % 97} " for k in range(n))
                digest.update(text.encode())
                if produced < cfg.reasoning_tokens:
                    delta = {"reasoning_content": text, "reasoning": text}
                else:
                    delta = {"content": text}
                if produced == 0:
                    delta["role"] = "assistant"
                produced += n
                self.chunk(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]))
            self.chunk(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "length"}]))
            if include_usage:
                final = dict(base, choices=[], usage={
                    "prompt_tokens": prompt_tokens, "completion_tokens": produced,
                    "total_tokens": prompt_tokens + produced})
                if cfg.extras:
                    drafted = produced * 2
                    final["sample_engine"] = {"drafted": drafted, "accepted": int(produced * 1.2),
                                           "token_sha": digest.hexdigest()[:16]}
                self.chunk(final)
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with s.lock:
                s.active -= 1
                s.active_prompt_tokens -= prompt_tokens


def start(config: FakeConfig | None = None, port: int = 0) -> FakeServer:
    srv = FakeServer(("127.0.0.1", port), config or FakeConfig())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=18443)
    ap.add_argument("--ttft-base", type=float, default=0.02)
    ap.add_argument("--prefill-per-token", type=float, default=2e-5)
    ap.add_argument("--token-delay", type=float, default=0.004)
    ap.add_argument("--slowdown", type=float, default=0.15)
    ap.add_argument("--chunk-tokens", type=int, default=1)
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--base-gib", type=float, default=10.0)
    ap.add_argument("--gib-per-ktoken", type=float, default=0.05)
    ap.add_argument("--decode-per-ktoken", type=float, default=0.0)
    a = ap.parse_args()
    cfg = FakeConfig(a.ttft_base, a.prefill_per_token, a.token_delay, a.slowdown, a.chunk_tokens,
                     not a.no_extras, 0, a.api_key, a.base_gib, a.gib_per_ktoken,
                     a.decode_per_ktoken)
    srv = FakeServer(("127.0.0.1", a.port), cfg)
    print(f"fake server on {srv.url}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
