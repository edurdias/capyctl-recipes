#!/usr/bin/env python3
"""capyctl-bench: a standard benchmark and report for CapyCTL recipes.

`run` measures an OpenAI-compatible endpoint (CapyCTL's inference endpoint)
with streaming chat completions, or with `--decisions` a decision model through
`POST /v1/systemone`, and writes one results JSON file.
`report` overlays any number of results files on the same charts and tables.

Standard library only. The PNG charts of `report` need matplotlib; run it with
`uv run --with matplotlib python3 capyctl_bench.py report ...`. Without
matplotlib the report still writes report.html, summary.md and data.csv.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import datetime as dt
import hashlib
import html
import http.client
import io
import json
import math
import os
import random
import re
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TOOL_NAME = "capyctl-bench"
TOOL_VERSION = "0.3.0"
# 2 adds `mode` ("chat" or "decisions") and the `decisions` section. A version 1 file is a chat
# run from before `mode` existed. 3 changes the decisions run: latency points carry per-question
# time, a `cache` section, closed-loop `concurrency` levels (requests instead of rounds), the
# request `manifest`, and accuracy sets of three kinds (classic, typed, di). `report` reads 1 to 3.
SCHEMA_VERSION = 3
READABLE_SCHEMAS = (1, 2, 3)
HERE = Path(__file__).resolve().parent
DEFAULT_PROMPTS = HERE / "prompts.json"
ASSETS = HERE / "assets"
RECIPES_URL = "github.com/edurdias/capyctl-recipes"

# Keys of a chat.completion.chunk defined by the OpenAI API. Anything else at
# the top level of a chunk is kept as an engine extra (for example a
# `tensorfold` object with draft counts).
STANDARD_CHUNK_KEYS = {
    "id", "object", "created", "model", "choices", "usage",
    "system_fingerprint", "service_tier", "prompt_logprobs", "kv_transfer_params",
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentile(values, q: float):
    """Linear-interpolation percentile (q in 0..100); None for no values."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def spread(values) -> dict:
    """Median, min, max and count of the non-None values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "n": len(vals),
    }


def find_draft_counts(extras):
    """Return (drafted, accepted) found anywhere in an engine extras object.

    Looks for numeric keys whose names contain "accept" (accepted tokens) and
    "draft" without "accept" (drafted tokens), for example
    `{"tensorfold": {"drafted": 120, "accepted": 85}}` or
    `{"num_draft_tokens": 120, "num_accepted_tokens": 85}`.
    """
    drafted = accepted = None
    stack = [extras]
    while stack:
        obj = stack.pop()
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    name = k.lower()
                    if "rate" in name or "ratio" in name or "len" in name:
                        continue
                    if "accept" in name and accepted is None:
                        accepted = v
                    elif "draft" in name and drafted is None:
                        drafted = v
        elif isinstance(obj, list):
            stack.extend(obj)
    if drafted is None or accepted is None:
        return None
    return drafted, accepted


def acceptance(requests) -> float | None:
    drafted = accepted = 0
    seen = False
    for r in requests:
        counts = find_draft_counts(r.get("extras") or {})
        if counts:
            seen = True
            drafted += counts[0]
            accepted += counts[1]
    if not seen or drafted <= 0:
        return None
    return accepted / drafted


REQUEST_METRICS = (
    "ttft_s", "total_s", "decode_tps", "tpot_ms", "prompt_tps",
    "output_bytes_per_s", "output_chars_per_s", "completion_tokens", "prompt_tokens",
    "memory_peak_gib",
)


def summarize_requests(requests) -> dict:
    ok = [r for r in requests if not r.get("error")]
    out = {m: spread(r.get(m) for r in ok) for m in REQUEST_METRICS}
    out["requests"] = len(requests)
    out["errors"] = len(requests) - len(ok)
    out["ttft_p50_s"] = percentile([r.get("ttft_s") for r in ok], 50)
    out["ttft_p95_s"] = percentile([r.get("ttft_s") for r in ok], 95)
    out["draft_acceptance"] = acceptance(ok)
    return out


# ---------------------------------------------------------------------------
# Endpoint and streaming
# ---------------------------------------------------------------------------

def read_api_key(path: str | None) -> str | None:
    """API key from a CapyCTL credentials file, else CAPYCTL_API_KEY.

    Accepts the standalone credentials file (an `api_key: <key>` line) and the
    server's JSON credentials file (an `"api_key": "<key>"` member).
    """
    if path:
        text = Path(path).expanduser().read_text()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and obj.get("api_key"):
                return str(obj["api_key"])
        except ValueError:
            pass
        for line in text.splitlines():
            m = re.match(r'^\s*"?api_key"?\s*:\s*"?([^"\s]+)"?\s*,?\s*$', line)
            if m:
                return m.group(1)
        raise SystemExit(f"no api_key line in {path}")
    return os.environ.get("CAPYCTL_API_KEY") or None


class Endpoint:
    def __init__(self, url: str, api_key: str | None = None, timeout: float = 900.0):
        u = urllib.parse.urlsplit(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise SystemExit(f"--endpoint must be an http(s) URL, got {url!r}")
        self.scheme = u.scheme
        self.host = u.hostname
        self.port = u.port or (443 if u.scheme == "https" else 80)
        self.base_path = u.path.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def redacted(self) -> str:
        """Scheme, port and path only: results files never name the host."""
        return f"{self.scheme}://<host>:{self.port}{self.base_path}"

    def connect(self):
        if self.scheme == "https":
            return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout,
                                               context=ssl.create_default_context())
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def headers(self, accept: str = "text/event-stream") -> dict:
        h = {"Content-Type": "application/json", "Accept": accept}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h


def _merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v


def _delta_text(delta: dict) -> str:
    # vLLM may send the same reasoning text as both `reasoning_content` and
    # `reasoning`; count it once.
    reasoning = delta.get("reasoning_content")
    if not isinstance(reasoning, str) or not reasoning:
        reasoning = delta.get("reasoning")
    if not isinstance(reasoning, str):
        reasoning = ""
    content = delta.get("content")
    if not isinstance(content, str):
        content = ""
    return reasoning + content


NO_TOKENS_ERROR = "no content or reasoning tokens in the stream"


def stream_chat(ep: Endpoint, payload: dict, record_chunks: bool = False) -> dict:
    """Send one streaming chat completion and measure it.

    Times are seconds from just before the request is written. A "token
    chunk" is a chunk whose delta carries content or reasoning text.
    """
    body = json.dumps(payload).encode()
    sent_at = time.time()
    t0 = time.perf_counter()
    token_times: list[float] = []
    parts: list[str] = []
    usage = None
    extras: dict = {}
    finish_reason = None
    error = None
    chunks = 0
    done = False
    conn = ep.connect()
    try:
        conn.request("POST", ep.base_path + "/chat/completions", body=body, headers=ep.headers())
        resp = conn.getresponse()
        if resp.status != 200:
            detail = resp.read(400).decode("utf-8", "replace").strip()
            error = f"HTTP {resp.status}: {detail}"
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                now = time.perf_counter() - t0
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("error"):
                    error = f"stream error: {json.dumps(obj['error'])[:400]}"
                    break
                chunks += 1
                extra = {k: v for k, v in obj.items() if k not in STANDARD_CHUNK_KEYS}
                if extra:
                    _merge(extras, extra)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    text = _delta_text(choice.get("delta") or {})
                    if text:
                        token_times.append(now)
                        parts.append(text)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
    except (OSError, http.client.HTTPException) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    total = time.perf_counter() - t0
    if error is None and not done:
        # The connection closed before [DONE]: something between (a proxy's idle bound, for
        # example) cut the stream, so even a reply with tokens is incomplete.
        got = (f"{len(token_times)} token chunks" if token_times
               else f"no content or reasoning tokens, {chunks} chunk{'' if chunks == 1 else 's'}")
        error = f"stream ended without [DONE] after {total:.1f} s ({got})"

    text = "".join(parts)
    usage = usage or {}
    completion_tokens = usage.get("completion_tokens")
    usage_missing = completion_tokens is None
    if usage_missing:
        completion_tokens = len(token_times)
    prompt_tokens = usage.get("prompt_tokens")
    rec: dict = {
        "sent_at": round(sent_at, 6),
        "error": error,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "usage_missing": usage_missing,
        "token_chunks": len(token_times),
        "ttft_s": None,
        "total_s": round(total, 6),
        "decode_s": None,
        "decode_tps": None,
        "tpot_ms": None,
        "prompt_tps": None,
        "output_bytes": len(text.encode("utf-8")),
        "output_chars": len(text),
        "output_bytes_per_s": None,
        "output_chars_per_s": None,
        "extras": extras,
        "memory_peak_gib": None,
    }
    if not token_times and not error:
        rec["error"] = NO_TOKENS_ERROR
    if token_times:
        first, last = token_times[0], token_times[-1]
        rec["ttft_s"] = round(first, 6)
        window = last - first
        rec["decode_s"] = round(window, 6)
        if completion_tokens and completion_tokens > 1 and window > 0:
            rec["decode_tps"] = (completion_tokens - 1) / window
            rec["tpot_ms"] = window / (completion_tokens - 1) * 1000.0
            rec["output_bytes_per_s"] = rec["output_bytes"] / window
            rec["output_chars_per_s"] = rec["output_chars"] / window
        if prompt_tokens and first > 0:
            rec["prompt_tps"] = prompt_tokens / first
    if record_chunks:
        rec["chunk_times_s"] = [round(t, 6) for t in token_times]
    return rec


# Keys of a /v1/systemone response defined by the System One API; the rest are engine extras.
STANDARD_DECISION_KEYS = {"id", "object", "created", "model", "answers", "usage"}


def check_answers(questions: dict, answers) -> str | None:
    """None when every question has an answer of its type, else what is wrong."""
    if not isinstance(answers, dict):
        return "no answers object in the response"
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            return f"no answer for {qid!r}"
        t = q["type"]
        if t == "choice":
            ok = a.get("choice") in q["criteria"]
        else:
            v = a.get(t)
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        if not ok:
            return f"the answer for {qid!r} is not a {t} answer: {json.dumps(a)[:200]}"
    return None


def post_decision(ep: Endpoint, payload: dict, keep_answers: bool = False) -> dict:
    """Send one `POST /systemone` and time it: request written to the whole body read."""
    body = json.dumps(payload).encode()
    sent_at = time.time()
    t0 = time.perf_counter()
    error = None
    obj = None
    conn = ep.connect()
    try:
        conn.request("POST", ep.base_path + "/systemone", body=body, headers=ep.headers("application/json"))
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            error = f"HTTP {resp.status}: {raw[:400].decode('utf-8', 'replace').strip()}"
        else:
            try:
                obj = json.loads(raw)
            except ValueError:
                error = f"response is not JSON: {raw[:200].decode('utf-8', 'replace')!r}"
    except (OSError, http.client.HTTPException) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    latency = time.perf_counter() - t0
    if error is None and not isinstance(obj, dict):
        error = "response is not a JSON object"
    usage = (obj or {}).get("usage") if isinstance(obj, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    answers = obj.get("answers") if isinstance(obj, dict) else None
    if error is None:
        error = check_answers(payload["questions"], answers)
    rec: dict = {
        "sent_at": round(sent_at, 6),
        "error": error,
        "latency_s": round(latency, 6),
        "questions": len(payload["questions"]),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "extras": {k: v for k, v in obj.items() if k not in STANDARD_DECISION_KEYS} if isinstance(obj, dict) else {},
        "memory_peak_gib": None,
    }
    if keep_answers and error is None:
        rec["answers"] = answers
    return rec


# ---------------------------------------------------------------------------
# Memory sampler
# ---------------------------------------------------------------------------

class MemorySampler(threading.Thread):
    """Runs a shell command every `interval` seconds; its output's last number is GiB used."""

    def __init__(self, command: str, interval: float = 0.5):
        super().__init__(daemon=True)
        self.command = command
        self.interval = interval
        self.samples: list[list[float]] = []
        self.errors = 0
        self.last_error = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def sample_once(self):
        try:
            # The command is the operator's own shell pipeline (often ssh), so it runs in a shell.
            out = subprocess.run(self.command, shell=True, capture_output=True, text=True,
                                 timeout=max(5.0, self.interval * 10))
            nums = re.findall(r"-?\d+(?:\.\d+)?", out.stdout)
            if out.returncode != 0 or not nums:
                raise ValueError(f"exit {out.returncode}, output {out.stdout.strip()[:80]!r} {out.stderr.strip()[:120]!r}")
            return float(nums[-1])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.errors += 1
            self.last_error = str(exc)
            return None

    def run(self):
        while not self._stop_event.is_set():
            start = time.time()
            value = self.sample_once()
            if value is not None:
                with self._lock:
                    self.samples.append([round(start, 3), value])
            self._stop_event.wait(max(0.0, self.interval - (time.time() - start)))

    def stop(self):
        self._stop_event.set()
        self.join(timeout=15)

    def peak(self, start: float, end: float):
        """Highest sample taken while [start, end] ran; the nearest sample if none fell inside."""
        with self._lock:
            samples = list(self.samples)
        vals = [v for t, v in samples if start - self.interval <= t <= end]
        if vals:
            return max(vals)
        near = [(min(abs(t - start), abs(t - end)), v) for t, v in samples]
        near = [nv for nv in near if nv[0] <= max(5.0, self.interval * 4)]
        return min(near)[1] if near else None


class NoSampler:
    samples: list = []
    errors = 0
    last_error = None

    def peak(self, start, end):
        return None

    def stop(self):
        pass


# ---------------------------------------------------------------------------
# Deterministic prompts
# ---------------------------------------------------------------------------

WORDS = (
    "the a an of to and in on for with by from as at that this these those it its "
    "system engine model memory request token stream server client cache queue batch "
    "layer kernel device driver process thread signal record report value number "
    "river mountain forest harbor village garden market library bridge window station "
    "morning evening winter summer season weather journey letter story picture method "
    "careful quiet steady bright narrow simple distant ancient modern common gentle "
    "builds measures carries follows changes returns opens holds keeps shows writes reads "
    "slowly quickly often rarely always never early later again together outside inside "
    "after before during while because although when where which every each several many"
).split()


def filler_text(sentences: int, seed: int) -> str:
    """Deterministic pseudo-English text of `sentences` sentences."""
    state = (seed * 2654435761 + 1) % 2147483648 or 1

    def nxt() -> int:
        nonlocal state
        state = (1103515245 * state + 12345) % 2147483648
        return state >> 8

    out = []
    for _ in range(sentences):
        n = 8 + nxt() % 9
        words = [WORDS[nxt() % len(WORDS)] for _ in range(n)]
        words[0] = words[0].capitalize()
        out.append(" ".join(words) + ".")
    return " ".join(out)


def context_messages(sentences: int, seed: int) -> list[dict]:
    text = filler_text(sentences, seed)
    return [{
        "role": "user",
        "content": (
            f"Document {seed}. Read the notes below, then answer the request at the end.\n\n"
            f"{text}\n\n"
            "Request: write a long, detailed summary of the notes above, paragraph by paragraph."
        ),
    }]


def parse_size(token: str) -> int:
    """'0.5k' -> 500, '128k' -> 128000, '2048' -> 2048 (k is 1000 tokens)."""
    t = token.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(k|m)?", t)
    if not m:
        raise argparse.ArgumentTypeError(f"bad context size {token!r}")
    mult = {"k": 1000, "m": 1000000}.get(m.group(2) or "", 1)
    return int(round(float(m.group(1)) * mult))


def size_label(tokens: float) -> str:
    if tokens >= 500:
        v = tokens / 1000
        return (f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:g}") + "k"
    return f"{tokens:g}"


def parse_sweep(spec: str) -> list[int]:
    sizes = [parse_size(s) for s in spec.split(",") if s.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("no sizes given")
    return sizes


def parse_levels(spec: str) -> list[int]:
    levels: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            levels.extend(range(int(a), int(b) + 1))
        else:
            levels.append(int(part))
    if not levels or min(levels) < 1:
        raise argparse.ArgumentTypeError(f"bad --concurrency {spec!r}")
    return levels


def load_prompts(path: Path) -> list[dict]:
    obj = json.loads(Path(path).read_text())
    items = obj["prompts"] if isinstance(obj, dict) else obj
    prompts = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            prompts.append({"id": f"p{i}", "messages": [{"role": "user", "content": item}]})
        elif "messages" in item:
            prompts.append({"id": item.get("id", f"p{i}"), "messages": item["messages"]})
        else:
            prompts.append({"id": item.get("id", f"p{i}"),
                            "messages": [{"role": "user", "content": item["prompt"]}]})
    if not prompts:
        raise SystemExit(f"no prompts in {path}")
    return prompts


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, args, ep: Endpoint, sampler):
        self.args = args
        self.ep = ep
        self.sampler = sampler
        self.extra_body = json.loads(args.extra_body) if args.extra_body else {}
        if not isinstance(self.extra_body, dict):
            raise SystemExit("--extra-body must be a JSON object")

    def payload(self, messages, max_tokens: int) -> dict:
        body = {
            "model": self.args.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        _merge(body, json.loads(json.dumps(self.extra_body)))
        return body

    def one(self, messages, max_tokens: int) -> dict:
        return stream_chat(self.ep, self.payload(messages, max_tokens), self.args.record_chunks)

    # -- context sweep -----------------------------------------------------

    def calibrate(self) -> dict:
        """Fit prompt tokens = overhead + per_sentence * sentences from two probes."""
        sizes = (20, 200)
        points = []
        for n in sizes:
            rec = stream_chat(self.ep, self.payload(context_messages(n, 900000 + n), 1))
            # Only the prompt token count matters here. The one generated token may be a format
            # token the engine's parser holds back (gpt-oss channel markers on SGLang), which
            # leaves the stream without content or reasoning text.
            error = None if rec["error"] == NO_TOKENS_ERROR else rec["error"]
            if error or not rec["prompt_tokens"]:
                raise SystemExit(f"calibration request failed: {error or 'no usage.prompt_tokens'}")
            points.append((n, rec["prompt_tokens"]))
        (n1, t1), (n2, t2) = points
        per = (t2 - t1) / (n2 - n1)
        if per <= 0:
            raise SystemExit("calibration failed: prompt tokens did not grow with the prompt")
        return {"per_sentence_tokens": per, "overhead_tokens": t1 - per * n1,
                "probes": [{"sentences": n, "prompt_tokens": t} for n, t in points]}

    def context_sweep(self) -> dict:
        a = self.args
        cal = self.calibrate()
        log(f"calibrated: {cal['per_sentence_tokens']:.2f} tokens per sentence, "
            f"{cal['overhead_tokens']:.0f} tokens of overhead")

        def sentences_for(target: int) -> int:
            return max(1, round((target - cal["overhead_tokens"]) / cal["per_sentence_tokens"]))

        for w in range(a.warmup):
            self.one(context_messages(sentences_for(a.context_sweep[0]), 800000 + w), a.max_tokens)
        points = []
        for target in a.context_sweep:
            n = sentences_for(target)
            reqs = []
            for r in range(a.runs):
                rec = self.one(context_messages(n, target * 100 + r + 1), a.max_tokens)
                rec["run"] = r
                reqs.append(rec)
                log(f"context {size_label(target):>6} run {r + 1}/{a.runs}: " + describe(rec))
            points.append({
                "label": size_label(target),
                "target_prompt_tokens": target,
                "sentences": n,
                "requests": reqs,
                "summary": summarize_requests(reqs),
            })
            log(f"context {size_label(target):>6}: " + describe_summary(points[-1]["summary"]))
        return {
            "settings": {"targets": a.context_sweep, "runs": a.runs, "max_tokens": a.max_tokens,
                         "warmup": a.warmup, "unit": "k = 1000 prompt tokens"},
            "calibration": cal,
            "points": points,
        }

    # -- concurrency ------------------------------------------------------

    def round_(self, c: int, prompts: list[dict], offset: int, max_tokens: int) -> dict:
        results: list = [None] * c
        barrier = threading.Barrier(c)

        def worker(i: int):
            p = prompts[(offset + i) % len(prompts)]
            try:
                barrier.wait(timeout=60)
            except threading.BrokenBarrierError:
                pass
            rec = stream_chat(self.ep, self.payload(p["messages"], max_tokens), self.args.record_chunks)
            rec["prompt_id"] = p["id"]
            rec["stream"] = i
            results[i] = rec

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return summarize_round({"requests": results})

    def concurrency(self) -> dict:
        a = self.args
        prompts = load_prompts(a.prompts)
        # Every round, warm-up included, takes the next prompts of the set, across levels too:
        # a prompt sent twice can hit the engine's prefix cache and look faster than new
        # traffic. Found with long documents (2026-10-17): each level restarted at the first
        # prompt, so later levels re-sent prompts the earlier ones had cached.
        needed = sum(c * (a.warmup + a.rounds) for c in a.concurrency)
        if needed > len(prompts):
            log(f"note: {needed} requests over {len(prompts)} prompts; prompts repeat, and a "
                "repeated prompt can hit the engine's prefix cache")
        offset = 0
        levels = []
        for c in a.concurrency:
            for w in range(a.warmup):
                self.round_(c, prompts, offset, a.concurrency_max_tokens)
                offset += c
            rounds = []
            for r in range(a.rounds):
                rnd = self.round_(c, prompts, offset, a.concurrency_max_tokens)
                offset += c
                rnd["round"] = r
                rounds.append(rnd)
                agg = rnd["aggregate_tps"]
                log(f"concurrency {c} round {r + 1}/{a.rounds}: "
                    f"{agg:.1f} tok/s aggregate" if agg else
                    f"concurrency {c} round {r + 1}/{a.rounds}: all requests failed")
            levels.append(summarize_level({"concurrency": c, "rounds": rounds}))
        return {
            "settings": {"levels": a.concurrency, "rounds": a.rounds, "warmup": a.warmup,
                         "max_tokens": a.concurrency_max_tokens,
                         "prompt_set": str(Path(a.prompts).name), "prompts": [p["id"] for p in prompts]},
            "levels": levels,
        }


def summarize_round(rnd: dict) -> dict:
    """Fill a round's totals from its requests (all started together)."""
    results = rnd["requests"]
    ok = [r for r in results if not r["error"]]
    wall = None
    if ok:
        first = min(r["sent_at"] for r in ok)
        last = max(r["sent_at"] + r["total_s"] for r in ok)
        wall = last - first
    tokens = sum(r["completion_tokens"] or 0 for r in ok)
    rnd.update({
        "wall_s": wall,
        "completion_tokens": tokens,
        "aggregate_tps": tokens / wall if wall else None,
        "errors": len(results) - len(ok),
        "memory_peak_gib": max((r["memory_peak_gib"] for r in results
                                if r.get("memory_peak_gib") is not None), default=None),
    })
    return rnd


def summarize_level(level: dict) -> dict:
    rounds = level["rounds"]
    reqs = [q for rnd in rounds for q in rnd["requests"]]
    summary = summarize_requests(reqs)
    summary["aggregate_tps"] = spread(rnd["aggregate_tps"] for rnd in rounds)
    summary["per_stream_tps"] = summary["decode_tps"]
    summary["memory_peak_gib"] = spread(rnd["memory_peak_gib"] for rnd in rounds)
    level["summary"] = summary
    return level


def apply_memory(result: dict, sampler) -> None:
    """Set memory_peak_gib on every request from the samples, then re-summarize.

    Done after the run so a sample still running when a request ends counts.
    A round's peak covers the whole round and is set on each of its requests.
    """
    for p in (result.get("context_sweep") or {}).get("points") or []:
        for r in p["requests"]:
            r["memory_peak_gib"] = sampler.peak(r["sent_at"], r["sent_at"] + r["total_s"])
        p["summary"] = summarize_requests(p["requests"])
    for lv in (result.get("concurrency") or {}).get("levels") or []:
        for rnd in lv["rounds"]:
            reqs = rnd["requests"]
            start = min(r["sent_at"] for r in reqs)
            end = max(r["sent_at"] + r["total_s"] for r in reqs)
            peak = sampler.peak(start, end)
            for r in reqs:
                r["memory_peak_gib"] = peak
            summarize_round(rnd)
        summarize_level(lv)


def describe_summary(sm: dict) -> str:
    bits = [f"median {fmt(sm['decode_tps']['median'])} tok/s", f"ttft {fmt(sm['ttft_s']['median'])} s"]
    if sm["errors"]:
        bits.append(f"{sm['errors']} failed")
    return ", ".join(bits)


def describe(rec: dict) -> str:
    if rec["error"]:
        return f"error: {rec['error']}"
    bits = [f"prompt {rec['prompt_tokens']} tok", f"ttft {rec['ttft_s']:.3f} s"]
    if rec["decode_tps"]:
        bits.append(f"{rec['decode_tps']:.1f} tok/s")
    return ", ".join(bits)


# ---------------------------------------------------------------------------
# run --decisions: decision models (POST /v1/systemone)
# ---------------------------------------------------------------------------

# The latency grid, the cache test and the throughput levels ask the first N of these about the
# filler state. Mixed types, as an application would send them; what they ask does not matter.
LATENCY_QUESTIONS = [
    ("topic", {"type": "choice", "instructions": "What is the main subject of the notes?",
               "criteria": {"technology": "Computers, engines, servers, software or data",
                            "nature": "Rivers, mountains, forests, gardens or the weather",
                            "places": "Villages, markets, libraries, stations or harbors",
                            "other": "Anything else"}}),
    ("formal", {"type": "noul", "instructions": "Are the notes written in a formal register?",
                "criteria": {"true": "Formal, as in a report or a manual",
                             "false": "Casual, as in a chat or a diary"}}),
    ("clarity", {"type": "score", "instructions": "How easy are the notes to follow?",
                 "criteria": ["Hard to follow", "Somewhat clear", "Very clear"]}),
    ("season", {"type": "choice", "instructions": "Which season do the notes mention most?",
                "criteria": {"winter": "Winter or cold weather", "summer": "Summer or warm weather",
                             "none": "No season stands out"}}),
    ("question", {"type": "noul", "instructions": "Do the notes ask the reader a question?",
                  "criteria": {"true": "At least one sentence is a question to the reader",
                               "false": "Every sentence is a statement"}}),
]

# Without usage.input_tokens in the responses, state lengths are estimated from characters.
CHARS_PER_TOKEN = 4.0

DATASETS_URL = "https://datasets-server.huggingface.co"
DATASET_CACHE = "~/.cache/capyctl-bench/datasets"

# Probabilities are clipped to this before a logarithm (NLL of AG News and BoolQ, KL of Typed
# Decisions; the same epsilon as the public Typed Decisions scorer of SaylorTwift/typed-decisions-evals).
PROB_EPS = 1e-9
# Expected calibration error: top label, this many equal-width confidence bins over [0, 1].
ECE_BINS = 10

# The decision-index kit whose request construction and scoring the `di` sets reproduce.
DI_KIT = {"repo": "apolinario/decision-index", "url": "https://github.com/apolinario/decision-index",
          "commit": "f5a6e4fd85fed05e77878dfc4966d0568bfb45ec", "edition": "0.3"}
# The kit hashes every request as this payload; the results keep the same hash per row.
DI_REFERENCE_MODEL = "jev-1.13.0"
# HTTP 400/413/422 messages the kit counts as "unsupported" (a capacity limit) instead of an error.
# Both count against coverage; the results keep them apart.
CAPACITY_MARKERS = ("options per choice", "a choice needs at least two options", "a score takes 2 to 10 levels",
                    "the canvas holds", "maximum context length", "maximum model length",
                    "longer than the maximum model length", "context window", "too many tokens")

API_BANK_INSTRUCTIONS = ("Select the single API tool to invoke for the dialogue prefix in state. Use the published "
                         "catalog in state. Do not infer arguments or use the current/future API result.")

# Built-in accuracy sets, of three kinds:
#
# classic: AG News and BoolQ, one question per request, sampled rows. typed: Typed Decisions, every
# test case with its five questions in one request. di: Decision Index components, rebuilt from
# pinned upstream sources exactly as the decision-index kit builds them.
#
# classic and typed rows come from the Hugging Face datasets-server JSON rows API (no parquet reader
# needed). That API serves a repository's current revision only, so the pin is the revision the split
# was read at plus the sha256 of its rows (canonical JSON, one line per row, in row order): a split
# that changed fails the run instead of being measured silently. A di set's pin is the sha256 of the
# kit's own normalized file for the benchmark (artifacts/benchmark-suite/normalized/<track>.jsonl,
# recorded in the kit's hub/manifest.json): the rows built here are byte-identical to the kit's.
ACCURACY_SETS = {
    "ag_news": {
        "kind": "classic",
        "title": "AG News",
        "repo": "fancyzhx/ag_news",
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "config": "default",
        "split": "test",
        "sha256": "edaf2c78a140e85223420031ef116d2036a2affc1102ff2e97d47ae947b8b609",
        "license": "see the dataset card",
        "approx_payload_chars": 700,
        "type": "choice",
        "state_field": "text",
        "label_field": "label",
        "labels": ["World", "Sports", "Business", "Sci/Tech"],
        "question": {"type": "choice",
                     "instructions": "Which section of a news site does this article belong to?",
                     "criteria": {"World": "International news, politics, conflicts and diplomacy",
                                  "Sports": "Sports events, teams, athletes and competitions",
                                  "Business": "Companies, markets, the economy and finance",
                                  "Sci/Tech": "Science, technology, computing, the internet and space"}},
    },
    "boolq": {
        "kind": "classic",
        "title": "BoolQ",
        "repo": "google/boolq",
        "revision": "35b264d03638db9f4ce671b711558bf7ff0f80d5",
        "config": "default",
        "split": "validation",
        "sha256": "c81b6520f1a51a8b96de45420b3d1f78184aee00373c77852132b297f2e567ef",
        "license": "CC BY-SA 3.0",
        "approx_payload_chars": 900,
        "type": "noul",
        "state_field": "passage",
        "question_field": "question",
        "label_field": "answer",
        # The question of each row becomes the instructions; these criteria are the same for all.
        "criteria": {"true": "The passage shows that the answer is yes",
                     "false": "The passage shows that the answer is no"},
    },
    "typed_decisions": {
        "kind": "typed",
        "title": "Typed Decisions",
        "repo": "LocalLLaMA/typed-decisions",
        "revision": "7189f46b02248bd2569df7325c9df8e3b24e90a7",
        "config": "all",
        "split": "test",
        "sha256": "ec99662b8129a775b0c12a8e4889fa70c99830c58db43924177a44ef2d4836b9",
        "license": "Apache-2.0",
        "approx_payload_chars": 2340,
    },
    "api_bank": {
        "kind": "di",
        "title": "API-Bank",
        "catalog_id": 3,
        "track": "API-Bank-tool-selection",
        "metric": "accuracy",
        "chance": 0.0189,
        "source": {"repo": "AlibabaResearch/DAMO-ConvAI", "commit": "188835d4f9948563a6b9c8ac50cd0f3ae4021ed6",
                   "path": "api-bank"},
        "tree_url": "https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1",
        "raw_url": "https://raw.githubusercontent.com/{repo}/{commit}/{path}",
        "serialization": "ascii",
        "sha256": "33d2cdbe003f271f05c4296f9ca890cc5c7f93a4ac372fa2558c3ccd9ef081ff",
        "rows": 508,
        "excluded": [],
        "license": "see api-bank/LICENSE in DAMO-ConvAI",
        "approx_payload_chars": 27500,
    },
    "gpqa_diamond": {
        "kind": "di",
        "title": "GPQA Diamond",
        "short": "GPQA",
        "catalog_id": 25,
        "track": "GPQA-Diamond",
        "metric": "accuracy",
        "chance": 0.25,
        "source": {"url": "https://github.com/idavidrein/gpqa/raw/main/dataset.zip",
                   "sha256": "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc",
                   "member": "dataset/gpqa_diamond.csv",
                   # Public in the upstream README; the zip is encrypted to keep the questions out of
                   # crawls, not to restrict access.
                   "password": "deserted-untie-orchid"},
        "serialization": "ascii",
        "sha256": "96e00f1248b6af1d991ff10745b49e7049a3558781e6488d21c9eccac7a3e610",
        "rows": 198,
        # The kit's hub/excluded-questions.json: two rows with byte-identical options.
        "excluded": ["GPQA-Diamond:test:89", "GPQA-Diamond:test:126"],
        "license": "CC BY 4.0; the authors ask that the questions not be published in plain text",
        "approx_payload_chars": 760,
    },
    "home_appliances": {
        "kind": "di",
        "title": "Home appliances",
        "short": "Home",
        "catalog_id": 9,
        "track": "Home-Appliance",
        "metric": "case exact accuracy",
        "chance": 0.0,
        "source": {"generator": "di_home_appliance.py (the kit's decision_index/suite/build/home_appliance.py)"},
        "serialization": "canon",
        "sha256": "13862aa65d24c0adeca03185997d1e984080ee96dc020dba16c0ffd64f8a5e06",
        "rows": 160,
        # Edition 0.2.1 onward (decision_index/data/release-v2.1/home-appliances-subset.json): test rows
        # deduplicated by state (lowest row id kept) and rows whose state is a dev row's dropped.
        "kept": 88,
        "kept_run_ids_sha256": "f76d0ef2493916ba799a4e78e42d734100e152f2b73d6978ab2370a3866d6e87",
        "excluded": [],
        "license": "MIT (generated by the kit's own code)",
        "approx_payload_chars": 8200,
    },
}
# Names `--accuracy-sets` takes besides the set names.
SUITES = {
    "classic": ["ag_news", "boolq"],
    "typed": ["typed_decisions"],
    "decision-index": ["api_bank", "gpqa_diamond", "home_appliances"],
    "all": list(ACCURACY_SETS),
}


def parse_question_counts(spec: str) -> list[int]:
    counts = parse_levels(spec)
    if max(counts) > len(LATENCY_QUESTIONS):
        raise argparse.ArgumentTypeError(f"at most {len(LATENCY_QUESTIONS)} questions per request")
    return counts


def parse_sets(spec: str) -> list[str]:
    names: list[str] = []
    for token in (s.strip() for s in spec.split(",")):
        for name in SUITES.get(token, [token] if token else []):
            if name not in ACCURACY_SETS:
                raise argparse.ArgumentTypeError(
                    f"accuracy sets are {', '.join(ACCURACY_SETS)} (or {', '.join(SUITES)}); got {spec!r}")
            if name not in names:
                names.append(name)
    if not names:
        raise argparse.ArgumentTypeError(f"no accuracy set in {spec!r}")
    return names


def latency_questions(n: int) -> dict:
    return dict(LATENCY_QUESTIONS[:n])


def canonical_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def rows_hash(rows) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(canonical_row(row).encode() + b"\n")
    return h.hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sample_indices(n_rows: int, k: int, seed: int, name: str) -> list[int]:
    """k row indices of n_rows, in row order: the k smallest sha256(seed:name:index).

    Defined by a hash, not by Python's random, so every Python picks the same rows.
    """
    order = sorted(range(n_rows), key=lambda i: hashlib.sha256(f"{seed}:{name}:{i}".encode()).digest())
    return sorted(order[:k])


def http_get(url: str, attempts: int = 10) -> bytes:
    """GET a URL, retrying rate limits and server errors with backoff.

    The datasets-server rate-limits a client after a few dozen pages; a full download waits
    out its Retry-After (or 2, 4, ... 60 s) instead of failing.
    """
    host = urllib.parse.urlsplit(url).netloc
    for k in range(attempts):
        wait = min(60.0, 2.0 ** (k + 1))
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"{TOOL_NAME}/{TOOL_VERSION}"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or k == attempts - 1:
                raise SystemExit(f"GET {url}: HTTP {exc.code}")
            after = exc.headers.get("Retry-After") if exc.headers else None
            if after and after.strip().isdigit():
                wait = min(120.0, float(after))
            log(f"HTTP {exc.code} from {host}; retrying in {wait:.0f} s")
        except (urllib.error.URLError, OSError) as exc:
            if k == attempts - 1:
                raise SystemExit(f"GET {url}: {exc}")
        time.sleep(wait)
    raise AssertionError("unreachable")


def http_get_json(url: str, attempts: int = 10):
    try:
        return json.loads(http_get(url, attempts))
    except ValueError as exc:
        raise SystemExit(f"GET {url}: not JSON ({exc})")


def fetch_rows(base_url: str, spec: dict) -> list[dict]:
    """Every row of the split, in order, from the datasets-server rows API (100 per page)."""
    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        query = urllib.parse.urlencode({"dataset": spec["repo"], "config": spec["config"],
                                        "split": spec["split"], "offset": len(rows), "length": 100})
        page = http_get_json(f"{base_url.rstrip('/')}/rows?{query}")
        total = page["num_rows_total"]
        if not page.get("rows"):
            raise SystemExit(f"{spec['repo']} {spec['split']}: no rows at offset {len(rows)} of {total}")
        for r in page["rows"]:
            if r.get("truncated_cells"):
                raise SystemExit(f"{spec['repo']} row {r['row_idx']}: the rows API truncated {r['truncated_cells']}")
            if r["row_idx"] != len(rows):
                raise SystemExit(f"{spec['repo']}: expected row {len(rows)}, got {r['row_idx']}")
            rows.append(r["row"])
    return rows


def load_dataset(name: str, base_url: str, cache_dir: Path) -> list[dict]:
    """A datasets-server set (classic or typed), from the cache when it matches the pin, else downloaded."""
    spec = ACCURACY_SETS[name]
    path = cache_dir / f"{name}-{spec['split']}-{spec['revision'][:12]}.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if rows_hash(rows) == spec["sha256"]:
            return rows
        log(f"{path} does not match its pin; downloading it again")
    log(f"downloading {spec['repo']} {spec['split']} from {base_url} ...")
    rows = fetch_rows(base_url, spec)
    got = rows_hash(rows)
    if got != spec["sha256"]:
        raise SystemExit(f"{spec['repo']} {spec['split']}: rows sha256 {got} is not the pinned "
                         f"{spec['sha256']} (revision {spec['revision']}); the dataset changed since it "
                         "was pinned. Pin the new revision and hash in ACCURACY_SETS, in their own change.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_row(r) + "\n" for r in rows))
    return rows


# --- Decision Index components, rebuilt as the kit builds them --------------------------------------

def dump_ascii(x) -> str:
    """The kit's normalized-row serialization (layout.dump_ascii)."""
    return json.dumps(x, ensure_ascii=True, separators=(",", ":"))


def kit_dumps(x) -> str:
    """The kit's payload serialization (suite.io.dumps)."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def di_payload_sha256(row: dict) -> str:
    """The kit's `_evaluation.payload_sha256` of a row: the same request has the same hash there."""
    return sha256_hex(kit_dumps({"model": DI_REFERENCE_MODEL, "state": row["state"],
                                 "questions": row["questions"]}).encode())


def di_run_id(spec: dict, row: dict) -> str:
    return f"{spec['catalog_id']}:{spec['track']}:{row['id']}"


def build_api_bank(spec: dict) -> list[dict]:
    """decision_index/suite/build/adapters_selection.py `apibank`: one row per API turn of every
    lv1-lv2 sample dialogue; the options are all 53 catalog tools, shuffled per row."""
    src = spec["source"]
    tree = http_get_json(spec["tree_url"].format(**src))
    if tree.get("truncated"):
        raise SystemExit(f"{src['repo']}: the git tree listing is truncated")
    root = src["path"] + "/"
    blobs = [t["path"] for t in tree["tree"] if t.get("type") == "blob" and t["path"].startswith(root)]
    apis = sorted(p for p in blobs if p.startswith(root + "apis/") and p.count("/") == 2 and p.endswith(".py"))
    # the kit's rglob order: by path components
    samples = sorted((p for p in blobs if p.startswith(root + "lv1-lv2-samples/") and p.endswith(".jsonl")),
                     key=lambda p: p.split("/"))
    log(f"downloading {len(apis) + len(samples)} API-Bank files of {src['repo']} at {src['commit'][:12]} ...")

    def get(p):
        return http_get(spec["raw_url"].format(repo=src["repo"], commit=src["commit"], path=urllib.parse.quote(p)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = dict(zip(apis + samples, pool.map(get, apis + samples)))
    tools = {}
    for p in apis:
        if p.rsplit("/", 1)[1] in ("api.py", "__init__.py"):
            continue
        cls = next((n for n in ast.parse(files[p].decode()).body if isinstance(n, ast.ClassDef)), None)
        if not cls:
            continue
        vals = {}
        for n in cls.body:
            if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("description", "input_parameters", "output_parameters")):
                try:
                    vals[n.targets[0].id] = ast.literal_eval(n.value)
                except (ValueError, SyntaxError):
                    vals[n.targets[0].id] = None
        tools[cls.name] = {"name": cls.name, "description": vals.get("description", ""),
                           "input_parameters": vals.get("input_parameters", {}),
                           "output_parameters": vals.get("output_parameters", {})}
    names = sorted(tools)
    rows = []
    for p in samples:
        fname = p.rsplit("/", 1)[1]
        turns = [json.loads(x) for x in files[p].decode().splitlines() if x.strip()]
        for pos in [i for i, t in enumerate(turns) if t.get("role") == "API"]:
            gold = turns[pos].get("api_name")
            if gold not in tools:
                continue
            order = list(range(len(names)))
            random.Random(f"20260918:{fname}:{pos}").shuffle(order)
            labels = {f"option_{i}": names[j] for i, j in enumerate(order)}
            goldkey = next(k for k, v in labels.items() if v == gold)
            rows.append({
                "id": f"API-Bank-tool-selection:{fname.rsplit('.', 1)[0]}:api-{pos}",
                "family": "API-Bank-tool-selection", "split": "evaluation-only",
                "state": {"dialogue": turns[:pos], "available_tools": [tools[k] for k in names],
                          "task": "API-Bank current API tool selection"},
                "questions": {"tool": {"type": "choice", "instructions": API_BANK_INSTRUCTIONS, "criteria": labels}},
                "expected": {"tool": goldkey},
                "metadata": {"provenance": {"source": "artifacts/benchmark-suite/raw/repos/apibank/" + p,
                                            "source_sha256": sha256_hex(files[p]), "turn_index": pos},
                             "gold_tool": gold,
                             "candidate_policy": "full published API catalog; source available subset is not "
                                                 "documented in this sample format",
                             "adaptation_scope": "Tool-name selection only; arguments and execution are not claimed."}})
    return rows


def build_gpqa_diamond(spec: dict) -> list[dict]:
    """decision_index/suite/build/normalize_text.py `gpqa`: the question as instructions, an empty
    state, the four published options shuffled per row and keyed A to D."""
    src = spec["source"]
    log(f"downloading {src['url']} ...")
    data = http_get(src["url"])
    if sha256_hex(data) != src["sha256"]:
        raise SystemExit(f"{src['url']}: sha256 {sha256_hex(data)} is not the pinned {src['sha256']}")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        text = z.read(src["member"], pwd=src["password"].encode()).decode("utf-8-sig")
    rows = []
    for i, r in enumerate(csv.DictReader(io.StringIO(text))):
        options = [r["Correct Answer"]] + [r[f"Incorrect Answer {n}"] for n in (1, 2, 3)]
        order = list(range(4))
        random.Random(f"20260918:GPQA:{i}").shuffle(order)
        criteria = {chr(65 + k): options[j] for k, j in enumerate(order)}
        rows.append(dict(
            id=f"GPQA-Diamond:test:{i}", family="GPQA-Diamond", split="test", state="",
            questions={"answer": dict(type="choice", instructions=r["Question"], criteria=criteria)},
            expected={"answer": chr(65 + order.index(0))},
            metadata=dict(benchmark_number=25, source_file="data/sources/gpqa/dataset.zip", source_sha256=src["sha256"],
                          source_id=i, member=src["member"], source="https://github.com/idavidrein/gpqa",
                          shuffle_seed="20260918:GPQA:<source-index>", source_option_indices=order,
                          handling="LOCAL EVALUATION ONLY; do not upload benchmark text or tokens to public artifacts",
                          primary_metric="accuracy")))
    return rows


def home_appliance_rows(split: str) -> list[dict]:
    import di_home_appliance as home
    rows = [home.make_row(h, i) for h in range(len(home.HOUSEHOLDS)) for i in range(10)]
    return [r for r in rows if r["split"] == split]


def build_home_appliances(spec: dict) -> list[dict]:
    """decision_index/suite/build/home_appliance.py: generated, so no download."""
    return home_appliance_rows("test")


DI_BUILDERS = {"api_bank": build_api_bank, "gpqa_diamond": build_gpqa_diamond, "home_appliances": build_home_appliances}


def di_serialize(rows: list[dict], spec: dict) -> bytes:
    if spec["serialization"] == "canon":
        return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
                       + "\n" for r in rows).encode()
    return "".join(dump_ascii(r) + "\n" for r in rows).encode()


def home_appliances_kept(rows: list[dict]) -> list[dict]:
    """Edition 0.2.1's cut: of identical test states the lowest row id stays, and test rows whose state
    is a dev row's are dropped (states compared as JSON with sorted keys)."""
    def key(r):
        return json.dumps(r["state"], sort_keys=True)
    dev = {key(r) for r in home_appliance_rows("dev")}
    seen: set = set()
    keep = set()
    for r in sorted(rows, key=lambda r: r["id"]):
        k = key(r)
        if k not in seen and k not in dev:
            keep.add(r["id"])
        seen.add(k)
    return [r for r in rows if r["id"] in keep]


def load_di_rows(name: str, cache_dir: Path) -> tuple[list[dict], list[dict]]:
    """(every built row, the rows scored): built from the pinned sources, checked byte for byte
    against the kit's normalized file, then cut by the kit's exclusions or subset."""
    spec = ACCURACY_SETS[name]
    path = cache_dir / f"{name}-{spec['sha256'][:12]}.jsonl"
    data = path.read_bytes() if path.exists() else None
    if data is not None and sha256_hex(data) != spec["sha256"]:
        log(f"{path} does not match its pin; building it again")
        data = None
    if data is None:
        data = di_serialize(DI_BUILDERS[name](spec), spec)
        if sha256_hex(data) != spec["sha256"]:
            raise SystemExit(f"{spec['title']}: the rebuilt rows have sha256 {sha256_hex(data)}, not the kit's "
                             f"{spec['sha256']}; an upstream source changed since it was pinned")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rows = [json.loads(line) for line in data.decode().splitlines() if line]
    if len(rows) != spec["rows"]:
        raise SystemExit(f"{spec['title']}: {len(rows)} rows, expected {spec['rows']}")
    excluded = set(spec["excluded"])
    kept = [r for r in rows if r["id"] not in excluded]
    if name == "home_appliances":
        kept = home_appliances_kept(kept)
        ids = sorted(di_run_id(spec, r) for r in kept)
        if len(kept) != spec["kept"] or sha256_hex("\n".join(ids).encode()) != spec["kept_run_ids_sha256"]:
            raise SystemExit(f"{spec['title']}: the 0.2.1 cut kept {len(kept)} rows, not the kit's {spec['kept']}")
    return rows, kept


# --- Scoring -------------------------------------------------------------------------------------

def mean(values):
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def ece_bins(pairs, bins: int = ECE_BINS) -> tuple:
    """(top-label ECE, reliability bins) of (confidence, correct) pairs: `bins` equal-width bins over
    [0, 1] (a confidence of exactly 1 goes in the last); ECE is the count-weighted mean |accuracy -
    mean confidence| over the non-empty bins. Every bin is listed, empty ones with n = 0."""
    pairs = [(min(max(float(c), 0.0), 1.0), bool(ok)) for c, ok in pairs]
    groups: list[list] = [[] for _ in range(bins)]
    for c, ok in pairs:
        groups[min(int(c * bins), bins - 1)].append((c, ok))
    out, ece = [], 0.0
    for k, g in enumerate(groups):
        conf = mean(c for c, _ in g)
        acc = mean(1.0 if ok else 0.0 for _, ok in g)
        out.append({"lo": k / bins, "hi": (k + 1) / bins, "n": len(g), "confidence": conf, "accuracy": acc})
        if g:
            ece += len(g) / len(pairs) * abs(acc - conf)
    return (ece if pairs else None), out


def macro_f1(pairs) -> float | None:
    """Mean F1 over every label seen as gold or prediction (the decision-index kit's definition)."""
    pairs = list(pairs)
    labels = {x for p in pairs for x in p}
    scores = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in pairs)
        fp = sum(g != lab and p == lab for g, p in pairs)
        fn = sum(g == lab and p != lab for g, p in pairs)
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return mean(scores)


def argmax_first(values: list) -> int:
    """Index of the largest value; ties go to the first."""
    return max(range(len(values)), key=lambda k: (values[k], -k))


def class_probs(item: dict, spec: dict) -> list | None:
    """The answer's distribution over the set's classes (choice: its labels; noul: [yes, no])."""
    if spec["type"] == "noul":
        p = item.get("noul")
        return None if p is None else [p, 1.0 - p]
    probs = item.get("probabilities")
    if not isinstance(probs, dict):
        return None
    v = [float(probs.get(lab) or 0.0) for lab in spec["labels"]]
    s = sum(v)
    return [x / s for x in v] if s > 0 else None


def class_index(spec: dict, label) -> int:
    return spec["labels"].index(label) if spec["type"] == "choice" else (0 if label else 1)


def summarize_decisions(requests) -> dict:
    ok = [r for r in requests if not r.get("error")]
    lat = [r["latency_s"] for r in ok]
    per_q = [r["latency_s"] / r["questions"] for r in ok if r.get("questions")]
    return {
        "latency_s": spread(lat),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        # amortized: request latency over the questions it answered
        "per_question_p50_s": percentile(per_q, 50),
        "per_question_p95_s": percentile(per_q, 95),
        "input_tokens": spread(r.get("input_tokens") for r in ok),
        "memory_peak_gib": spread(r.get("memory_peak_gib") for r in ok),
        "requests": len(requests),
        "errors": len(requests) - len(ok),
    }


def summarize_accuracy(items: list[dict], spec: dict) -> dict:
    """Accuracy and calibration of a classic set, over the answered rows."""
    ok = [i for i in items if not i.get("error")]
    lat = [i["latency_s"] for i in ok]
    out = {
        "items": len(items),
        "answered": len(ok),
        "errors": len(items) - len(ok),
        "accuracy": mean(1.0 if i["correct"] else 0.0 for i in ok),
        "macro_f1": macro_f1((str(i["label"]), str(i["prediction"])) for i in ok) if ok else None,
        "brier": None, "brier_vector": None, "nll": None, "nll_eps": PROB_EPS,
        "ece": None, "ece_bins": [], "calibrated": 0,
        "latency_s": spread(lat),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "input_tokens": spread(i.get("input_tokens") for i in ok),
    }
    vecs = [(p, class_index(spec, i["label"])) for i in ok if (p := class_probs(i, spec)) is not None]
    out["calibrated"] = len(vecs)
    if vecs:
        out["brier_vector"] = mean(sum((pk - (1.0 if k == y else 0.0)) ** 2 for k, pk in enumerate(p)) for p, y in vecs)
        out["nll"] = mean(-math.log(max(p[y], PROB_EPS)) for p, y in vecs)
        out["ece"], out["ece_bins"] = ece_bins((p[t], t == y) for p, y in vecs for t in [argmax_first(p)])
    if spec["type"] == "noul" and ok:
        out["brier"] = mean((i["noul"] - (1.0 if i["label"] else 0.0)) ** 2 for i in ok)
    return out


def typed_probabilities(gold_q: dict, ans: dict) -> tuple[dict, bool]:
    """(the answer's distribution over the gold's options, whether it had to be derived).

    A noul answer is its probability of true and its complement. A choice or score answer without a
    probabilities map becomes one-hot on its choice (or on the level nearest its score): derived."""
    options = list(gold_q["probabilities"])
    probs = ans.get("probabilities") if isinstance(ans.get("probabilities"), dict) else None
    derived = False
    if gold_q["type"] == "noul":
        if probs is None:
            p = float(ans["noul"])
            probs = {"true": p, "false": 1.0 - p}
        else:
            names = {"yes": "true", "no": "false", "True": "true", "False": "false"}
            probs = {names.get(str(k), str(k)): v for k, v in probs.items()}
    elif probs is None:
        derived = True
        if gold_q["type"] == "choice":
            probs = {ans["choice"]: 1.0}
        else:
            probs = {str(min(len(options) - 1, max(0, round(float(ans["score"]))))): 1.0}
    p = {o: float(probs.get(o) or 0.0) for o in options}
    s = sum(p.values())
    return ({o: v / s for o, v in p.items()} if s > 0 else {o: 1.0 / len(options) for o in options}), derived


def score_typed_decision(qid: str, gold_q: dict, ans: dict) -> dict:
    """One Typed Decisions answer against its gold: the card's metrics plus the hard-label ones."""
    options = list(gold_q["probabilities"])
    p, derived = typed_probabilities(gold_q, ans)
    g = {o: float(gold_q["probabilities"][o]) for o in options}
    pmax, gmax = max(p.values()), max(g.values())
    top = next(o for o in options if p[o] >= pmax - 1e-12)
    label = str(gold_q["label"])
    d = {
        "qid": qid, "type": gold_q["type"], "label": label, "prediction": top,
        "confidence": round(pmax, 6),
        "correct": top == label,
        "correct_gold_ties": g[top] >= gmax - 1e-12,
        "kl": round(max(0.0, sum(g[o] * math.log(g[o] / max(p[o], PROB_EPS)) for o in options if g[o] > 0)), 6),
        "brier": round(sum((p[o] - g[o]) ** 2 for o in options), 6),
        "brier_hard": round(sum((p[o] - (1.0 if o == label else 0.0)) ** 2 for o in options), 6),
        "probabilities": {o: round(v, 6) for o, v in p.items()},
        "derived": derived,
    }
    if gold_q["type"] == "score":
        sc = ans.get("score")
        pred = float(sc) if isinstance(sc, (int, float)) and not isinstance(sc, bool) else \
            sum(float(o) * v for o, v in p.items())
        gold = float(gold_q["score"]) if isinstance(gold_q.get("score"), (int, float)) else \
            sum(float(o) * v for o, v in g.items())
        d["score"] = round(pred, 6)
        d["score_error"] = round(abs(pred - gold), 6)
    return d


def typed_aggregate(ds: list[dict]) -> dict:
    return {"decisions": len(ds),
            "accuracy": mean(1.0 if d["correct"] else 0.0 for d in ds),
            "accuracy_gold_ties": mean(1.0 if d["correct_gold_ties"] else 0.0 for d in ds),
            "kl": mean(d["kl"] for d in ds),
            "brier": mean(d["brier"] for d in ds),
            "brier_hard": mean(d["brier_hard"] for d in ds)}


def summarize_typed(items: list[dict]) -> dict:
    """Typed Decisions over the answered cases (a failed case loses its five decisions: coverage)."""
    ok = [i for i in items if not i.get("error")]
    ds = [d for i in ok for d in i["decisions"]]
    scores = [d for d in ds if "score_error" in d]
    lat = [i["latency_s"] for i in ok]
    out = {"cases": len(items), "answered": len(ok), "errors": len(items) - len(ok),
           "coverage": len(ok) / len(items) if items else None, **typed_aggregate(ds)}
    out["ece"], out["ece_bins"] = ece_bins((d["confidence"], d["correct"]) for d in ds) if ds else (None, [])
    # the convention of the public third-party scorer: a prediction tied for the gold's top counts as right
    out["ece_gold_ties"] = ece_bins((d["confidence"], d["correct_gold_ties"]) for d in ds)[0] if ds else None
    out.update({
        "score_mae": mean(d["score_error"] for d in scores),
        "score_within_1": mean(1.0 if d["score_error"] <= 1.0 else 0.0 for d in scores),
        "derived_probabilities": sum(1 for d in ds if d["derived"]),
        "kl_eps": PROB_EPS,
        "by_type": {t: typed_aggregate([d for d in ds if d["type"] == t]) for t in ("noul", "choice", "score")
                    if any(d["type"] == t for d in ds)},
        "by_workflow": {w: typed_aggregate([d for i in ok if i["workflow"] == w for d in i["decisions"]])
                        for w in sorted({i["workflow"] for i in ok})},
        "latency_s": spread(lat),
        "latency_p50_s": percentile(lat, 50),
        "latency_p95_s": percentile(lat, 95),
        "input_tokens": spread(i.get("input_tokens") for i in ok),
    })
    return out


def failure_status(error: str) -> str:
    """The kit's split of a failed request: a capacity rejection is unsupported, the rest errors."""
    if re.match(r"HTTP (400|413|422)\b", error) and any(m in error for m in CAPACITY_MARKERS):
        return "unsupported"
    return "error"


def index_value(score, answered: int, requests: int, chance: float) -> dict:
    """The kit's per-benchmark index value (scoring/index02.py `display_value`): the score times
    coverage (unanswered counts as wrong), then chance-corrected and clipped to [0, 1]."""
    cov = answered / requests if requests else 1.0
    raw = 0.0 if score is None else round(score, 4) * cov
    skill = min(1.0, max(0.0, (raw - chance) / (1 - chance))) if chance < 1 else raw
    return {"coverage": cov, "raw": raw, "skill": skill, "chance": chance}


def summarize_di(items: list[dict], spec: dict) -> dict:
    ok = [i for i in items if i["status"] == "ok"]
    fields = sum(i["fields"] for i in ok)
    field_acc = sum(i["fields_correct"] for i in ok) / fields if fields else None
    case_exact = mean(1.0 if i["correct"] else 0.0 for i in ok)
    score = case_exact if spec["metric"] == "case exact accuracy" else field_acc
    lat = [i["latency_s"] for i in ok]
    out = {"requests": len(items), "answered": len(ok),
           "unsupported": sum(1 for i in items if i["status"] == "unsupported"),
           "errors": sum(1 for i in items if i["status"] == "error"),
           "metric": spec["metric"], "score": score, "field_accuracy": field_acc, "case_exact_accuracy": case_exact,
           **index_value(score, len(ok), len(items), spec["chance"]),
           "latency_s": spread(lat), "latency_p50_s": percentile(lat, 50), "latency_p95_s": percentile(lat, 95),
           "input_tokens": spread(i.get("input_tokens") for i in ok)}
    roles: dict = {}
    for i in ok:
        for role, hit in i.get("role_hits") or []:
            roles.setdefault(role, []).append(hit)
    if roles:
        out["field_accuracy_by_role"] = {k: {"n": len(v), "accuracy": mean(1.0 if h else 0.0 for h in v)}
                                         for k, v in sorted(roles.items())}
    return out


def accuracy_request(spec: dict, row: dict) -> tuple[str, dict, object]:
    """(state, question, true answer) for one classic dataset row."""
    state = row[spec["state_field"]]
    if spec["type"] == "choice":
        return state, spec["question"], spec["labels"][row[spec["label_field"]]]
    text = row[spec["question_field"]].strip().rstrip("?")
    question = {"type": "noul", "instructions": text[:1].upper() + text[1:] + "?", "criteria": spec["criteria"]}
    return state, question, bool(row[spec["label_field"]])


def describe_decision(rec: dict) -> str:
    if rec["error"]:
        return f"error: {rec['error']}"
    return f"input {rec['input_tokens']} tok, {rec['latency_s'] * 1000:.1f} ms"


def summarize_throughput(level: dict) -> dict:
    """A closed-loop level: every request, over the wall from the first sent to the last finished."""
    reqs = level["requests"]
    ok = [r for r in reqs if not r["error"]]
    wall = (max(r["sent_at"] + r["latency_s"] for r in reqs) - min(r["sent_at"] for r in reqs)) if reqs else None
    summary = summarize_decisions(reqs)
    summary.update({
        "wall_s": wall,
        "requests_per_s": len(ok) / wall if wall else None,
        "questions_per_s": sum(r["questions"] for r in ok) / wall if wall else None,
        "error_rate": (len(reqs) - len(ok)) / len(reqs) if reqs else None,
    })
    level["summary"] = summary
    return level


def speed_manifest(extra_body: dict) -> dict:
    """What the latency grid, the cache test and the throughput levels send, so others can replay it."""
    qs = dict(LATENCY_QUESTIONS)
    return {
        "endpoint": "POST <endpoint>/systemone",
        "body": ["model", "state", "questions"] + [k for k in extra_body if k not in ("model", "state", "questions")],
        "serialization": "Python json.dumps defaults: ', ' and ': ' separators, non-ASCII escaped, UTF-8",
        "questions": qs,
        "question_order": list(qs),
        "question_types": {k: q["type"] for k, q in qs.items()},
        "options": {k: len(q["criteria"]) for k, q in qs.items()},
        "n_questions": "the first N of question_order",
        "state": "deterministic pseudo-English filler (filler_text), sized in sentences from the calibration; "
                 "every request its own seed, except the repeated block of the cache test",
    }


class DecisionRunner:
    def __init__(self, args, ep: Endpoint, extra_body: dict):
        self.args = args
        self.ep = ep
        self.extra_body = extra_body
        self.seed = 7000000
        self.cal: dict | None = None

    def next_seed(self) -> int:
        # One counter for the whole run, warm-ups included: no two requests share a state (but the
        # cache test's repeated block), so none can reuse another's prefix cache.
        self.seed += 1
        return self.seed

    def payload(self, state, questions: dict) -> dict:
        body = {"model": self.args.model, "state": state, "questions": questions}
        _merge(body, json.loads(json.dumps(self.extra_body)))
        return body

    def one(self, state, questions: dict, keep_answers: bool = False) -> dict:
        return post_decision(self.ep, self.payload(state, questions), keep_answers)

    def calibrate(self) -> dict:
        """State tokens per filler sentence: the slope of usage.input_tokens between two probes."""
        probes = []
        for n in (20, 200):
            rec = self.one(filler_text(n, 900000 + n), latency_questions(1))
            if rec["error"]:
                raise SystemExit(f"calibration request failed: {rec['error']}")
            probes.append({"sentences": n, "input_tokens": rec["input_tokens"]})
        t1, t2 = probes[0]["input_tokens"], probes[1]["input_tokens"]
        if t1 and t2 and t2 > t1:
            per = (t2 - t1) / (200 - 20)
            return {"method": "usage.input_tokens", "per_sentence_tokens": per,
                    "overhead_tokens": t1 - per * 20, "probes": probes}
        per = len(filler_text(200, 900200)) / 200 / CHARS_PER_TOKEN
        log(f"no usable usage.input_tokens: state lengths are estimated at {CHARS_PER_TOKEN:g} characters per token")
        return {"method": f"{CHARS_PER_TOKEN:g} characters per token", "per_sentence_tokens": per,
                "overhead_tokens": None, "probes": probes}

    def sentences_for(self, target: int) -> int:
        return max(1, round(target / self.cal["per_sentence_tokens"]))

    def unique(self, n: int, questions: dict) -> dict:
        return self.one(filler_text(n, self.next_seed()), questions)

    def latency(self) -> dict:
        a = self.args
        points = []
        for qn in a.question_counts:
            questions = latency_questions(qn)
            for target in a.state_sweep:
                n = self.sentences_for(target)
                for _ in range(a.warmup):
                    self.unique(n, questions)
                reqs = []
                for r in range(a.runs):
                    rec = self.unique(n, questions)
                    rec["run"] = r
                    reqs.append(rec)
                label = f"{qn}q {size_label(target)}"
                points.append({"label": label, "questions": qn, "target_state_tokens": target,
                               "state_label": size_label(target), "sentences": n,
                               "requests": reqs, "summary": summarize_decisions(reqs)})
                sm = points[-1]["summary"]
                log(f"latency {label:>9}: p50 {fmt(ms(sm['latency_p50_s']))} ms, "
                    f"p95 {fmt(ms(sm['latency_p95_s']))} ms, {fmt(ms(sm['per_question_p50_s']))} ms per question"
                    + (f", {sm['errors']} failed: {next(q['error'] for q in reqs if q['error'])}"
                       if sm["errors"] else ""))
        return {
            "settings": {"question_counts": a.question_counts, "state_targets": a.state_sweep,
                         "runs": a.runs, "warmup": a.warmup, "warmup_scope": "before every point",
                         "unit": "k = 1000 state tokens", "concurrency": 1,
                         "questions": latency_questions(max(a.question_counts))},
            "points": points,
        }

    def cache_test(self) -> dict:
        """Cold prefix (every request a new state) against a repeated state (the same request again
        and again, after one untimed first sight): what a prefix or result cache saves."""
        a = self.args
        questions = latency_questions(a.concurrency_questions)
        n = self.sentences_for(a.concurrency_state)
        for _ in range(a.warmup):
            self.unique(n, questions)
        cold = [self.unique(n, questions) for _ in range(a.cache_runs)]
        state = filler_text(n, self.next_seed())
        first = self.one(state, questions)
        repeated = [self.one(state, questions) for _ in range(a.cache_runs)]
        out = {"settings": {"questions": a.concurrency_questions, "state_tokens": a.concurrency_state,
                            "sentences": n, "runs": a.cache_runs, "warmup": a.warmup},
               "cold": {"requests": cold, "summary": summarize_decisions(cold)},
               "repeated": {"first": first, "requests": repeated, "summary": summarize_decisions(repeated)}}
        c, r = out["cold"]["summary"], out["repeated"]["summary"]
        log(f"cache: cold p50 {fmt(ms(c['latency_p50_s']))} ms, repeated p50 {fmt(ms(r['latency_p50_s']))} ms")
        return out

    def closed_loop(self, c: int, questions: dict, states: list[str]) -> list[dict]:
        """`c` workers send `states` back to back, each taking the next as soon as it is done."""
        results: list = [None] * len(states)
        lock = threading.Lock()
        nxt = [0]

        def worker(w: int):
            while True:
                with lock:
                    i = nxt[0]
                    nxt[0] += 1
                if i >= len(states):
                    return
                rec = self.one(states[i], questions)
                rec["stream"] = w
                rec["seq"] = i
                results[i] = rec

        threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(c)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def concurrency(self) -> dict:
        a = self.args
        questions = latency_questions(a.concurrency_questions)
        n = self.sentences_for(a.concurrency_state)
        levels = []
        for c in a.concurrency:
            self.closed_loop(c, questions, [filler_text(n, self.next_seed()) for _ in range(max(c, a.warmup))])
            states = [filler_text(n, self.next_seed()) for _ in range(a.throughput_requests)]
            lv = summarize_throughput({"concurrency": c, "requests": self.closed_loop(c, questions, states)})
            levels.append(lv)
            sm = lv["summary"]
            log(f"concurrency {c}: {fmt(sm['requests_per_s'])} requests/s, {fmt(sm['questions_per_s'])} questions/s, "
                f"p50 {fmt(ms(sm['latency_p50_s']))} ms, {sm['errors']} failed")
        return {
            "settings": {"levels": a.concurrency, "requests": a.throughput_requests,
                         "warmup": "max(streams, --warmup) requests before each level",
                         "mode": "closed loop: each stream sends its next request when its last one returns",
                         "questions": a.concurrency_questions, "state_tokens": a.concurrency_state, "sentences": n},
            "levels": levels,
        }

    def rows_limit(self, rows: list) -> list:
        lim = self.args.accuracy_limit
        return rows[:lim] if lim else rows

    def progress(self, title: str, j: int, total: int, done: list) -> None:
        if (j + 1) % 50 == 0 or j + 1 == total:
            acc = mean(1.0 if x["correct"] else 0.0 for x in done)
            log(f"accuracy {title}: {j + 1}/{total}, {fmt(acc * 100 if acc is not None else None, '%')} so far")

    def classic_set(self, name: str, spec: dict, cache: Path) -> dict:
        a = self.args
        rows = load_dataset(name, a.datasets_url, cache)
        idx = sample_indices(len(rows), min(a.accuracy_samples, len(rows)), a.accuracy_seed, name)
        items = []
        for j, i in enumerate(idx):
            state, question, truth = accuracy_request(spec, rows[i])
            rec = self.one(state, {"answer": question}, keep_answers=True)
            item = {"row_idx": i, "label": truth, "prediction": None, "correct": None,
                    "sent_at": rec["sent_at"], "latency_s": rec["latency_s"],
                    "input_tokens": rec["input_tokens"], "error": rec["error"]}
            if not rec["error"]:
                ans = rec["answers"]["answer"]
                if spec["type"] == "choice":
                    item["prediction"] = ans["choice"]
                    item["probabilities"] = ans.get("probabilities")
                else:
                    item["noul"] = float(ans["noul"])
                    item["prediction"] = item["noul"] >= 0.5
                item["correct"] = item["prediction"] == truth
            items.append(item)
            self.progress(spec["title"], j, len(idx), [x for x in items if not x["error"]])
        summary = summarize_accuracy(items, spec)
        summary["memory_peak_gib"] = None
        return {
            "name": name, "kind": "classic", "title": spec["title"], "type": spec["type"],
            "dataset": {"repo": spec["repo"], "revision": spec["revision"], "config": spec["config"],
                        "split": spec["split"], "rows": len(rows), "sha256": spec["sha256"],
                        "source": "datasets-server rows API", "license": spec["license"]},
            "question": spec["question"] if spec["type"] == "choice" else
            {"type": "noul", "instructions": f"each row's `{spec['question_field']}`, capitalized, ending in ?",
             "criteria": spec["criteria"]},
            "state": f"each row's `{spec['state_field']}`",
            "sample": {"n": len(idx), "seed": a.accuracy_seed,
                       "method": "the n rows with the smallest sha256(\"seed:set:row_idx\")"},
            "summary": summary,
            "items": items,
        }

    def typed_set(self, name: str, spec: dict, cache: Path) -> dict:
        a = self.args
        rows = load_dataset(name, a.datasets_url, cache)
        scored = self.rows_limit(rows)
        items = []
        for j, row in enumerate(scored):
            state, questions, gold = (json.loads(row[k]) for k in ("state", "questions", "gold"))
            rec = self.one(state, questions, keep_answers=True)
            item = {"row_idx": j, "id": row["id"], "workflow": row["workflow"],
                    "payload_sha256": sha256_hex(canonical_row({"state": state, "questions": questions}).encode()),
                    "sent_at": rec["sent_at"], "latency_s": rec["latency_s"], "input_tokens": rec["input_tokens"],
                    "error": rec["error"], "decisions": None}
            if not rec["error"]:
                item["decisions"] = [score_typed_decision(qid, gq, rec["answers"][qid]) for qid, gq in gold.items()]
            items.append(item)
            if (j + 1) % 50 == 0 or j + 1 == len(scored):
                acc = mean(1.0 if d["correct"] else 0.0 for x in items if not x["error"] for d in x["decisions"])
                log(f"accuracy {spec['title']}: {j + 1}/{len(scored)} cases, "
                    f"{fmt(acc * 100 if acc is not None else None, '%')} of decisions so far")
        summary = summarize_typed(items)
        summary["memory_peak_gib"] = None
        return {
            "name": name, "kind": "typed", "title": spec["title"],
            "dataset": {"repo": spec["repo"], "revision": spec["revision"], "config": spec["config"],
                        "split": spec["split"], "rows": len(rows), "sha256": spec["sha256"],
                        "source": "datasets-server rows API", "license": spec["license"]},
            "request": {"state": "each row's `state`, parsed from its JSON string",
                        "questions": "each row's `questions`, parsed from its JSON string, all five in one request",
                        "payload_sha256": "sha256 of {state, questions} as JSON with sorted keys"},
            "rows": {"n": len(scored), "of": len(rows), "partial": len(scored) < len(rows)},
            "summary": summary,
            "items": items,
        }

    def di_set(self, name: str, spec: dict, cache: Path) -> dict:
        built, kept = load_di_rows(name, cache)
        scored = self.rows_limit(kept)
        items = []
        for j, row in enumerate(scored):
            rec = self.one(row["state"], row["questions"], keep_answers=True)
            item = {"id": row["id"], "run_id": di_run_id(spec, row), "payload_sha256": di_payload_sha256(row),
                    "status": "ok" if not rec["error"] else failure_status(rec["error"]),
                    "sent_at": rec["sent_at"], "latency_s": rec["latency_s"], "input_tokens": rec["input_tokens"],
                    "error": rec["error"], "fields": len(row["expected"])}
            if not rec["error"]:
                # the kit scores the engine's `choice`, not the argmax of its probabilities
                hits = {k: rec["answers"][k].get("choice") == g for k, g in row["expected"].items()}
                item.update({"fields_correct": sum(hits.values()), "correct": all(hits.values())})
                if len(hits) == 1:
                    (k, g), = row["expected"].items()
                    item.update({"prediction": rec["answers"][k].get("choice"), "gold": g})
                roles = (row.get("metadata") or {}).get("question_roles")
                if roles:
                    item["role_hits"] = [[roles[k].split(":")[0], hit] for k, hit in hits.items()]
            items.append(item)
            self.progress(spec["title"], j, len(scored), [x for x in items if x["status"] == "ok"])
        summary = summarize_di(items, spec)
        summary["memory_peak_gib"] = None
        qs = [len(r["questions"]) for r in kept]
        opts = [len(q["criteria"]) for r in kept for q in r["questions"].values()]
        return {
            "name": name, "kind": "di", "title": spec["title"],
            "dataset": {"kit": DI_KIT, "catalog_id": spec["catalog_id"], "track": spec["track"],
                        "source": {k: v for k, v in spec["source"].items() if k != "password"},
                        "sha256": spec["sha256"], "rows": len(built), "kept": len(kept),
                        "excluded": spec["excluded"], "license": spec["license"]},
            "request": {"state": "the kit row's `state`, unchanged", "questions": "the kit row's `questions`, unchanged",
                        "questions_per_request": {"min": min(qs), "max": max(qs)},
                        "options_per_question": {"min": min(opts), "max": max(opts)},
                        "prediction": "the answer's `choice`", "payload_sha256": "the kit's _evaluation.payload_sha256"},
            "rows": {"n": len(scored), "of": len(kept), "partial": len(scored) < len(kept)},
            "summary": summary,
            "items": items,
        }

    def accuracy(self) -> dict:
        a = self.args
        cache = Path(a.dataset_cache).expanduser()
        sets = []
        for name in a.accuracy_sets:
            spec = ACCURACY_SETS[name]
            build = {"classic": self.classic_set, "typed": self.typed_set, "di": self.di_set}[spec["kind"]]
            sets.append(build(name, spec, cache))
        return {"settings": {"sets": a.accuracy_sets, "samples": a.accuracy_samples, "seed": a.accuracy_seed,
                             "limit": a.accuracy_limit, "prob_eps": PROB_EPS, "ece_bins": ECE_BINS},
                "sets": sets}


def ms(v):
    return None if v is None else v * 1000.0


def apply_decision_memory(dec: dict, sampler) -> None:
    """memory_peak_gib per latency and cache request, per throughput level and per accuracy set."""
    def per_request(reqs):
        for r in reqs:
            r["memory_peak_gib"] = sampler.peak(r["sent_at"], r["sent_at"] + r["latency_s"])

    for p in (dec.get("latency") or {}).get("points") or []:
        per_request(p["requests"])
        p["summary"] = summarize_decisions(p["requests"])
    cache = dec.get("cache") or {}
    for part in ("cold", "repeated"):
        if part in cache:
            per_request(cache[part]["requests"])
            cache[part]["summary"] = summarize_decisions(cache[part]["requests"])
    for lv in (dec.get("concurrency") or {}).get("levels") or []:
        reqs = lv["requests"]
        peak = sampler.peak(min(r["sent_at"] for r in reqs), max(r["sent_at"] + r["latency_s"] for r in reqs))
        for r in reqs:
            r["memory_peak_gib"] = peak
        summarize_throughput(lv)
    for st in (dec.get("accuracy") or {}).get("sets") or []:
        items = st["items"]
        if items:
            st["summary"]["memory_peak_gib"] = sampler.peak(
                min(i["sent_at"] for i in items), max(i["sent_at"] + i["latency_s"] for i in items))


def parse_meta(items) -> dict:
    meta = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--meta expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        meta[k.strip()] = v.strip()
    if "sample" in meta:
        meta["sample"] = meta["sample"].lower() in ("1", "true", "yes")
    return meta


# A server that answers with this top-level chunk key (the test fake server
# does) produces sample data: the results file gets meta.sample = true and
# every report rendered from it says so.
SAMPLE_MARK = "capyctl_bench_sample"
SAMPLE_BANNER = "SAMPLE DATA — not a measurement"


def all_requests(result: dict):
    for p in (result.get("context_sweep") or {}).get("points") or []:
        yield from p["requests"]
    for lv in (result.get("concurrency") or {}).get("levels") or []:
        for rnd in lv["rounds"]:
            yield from rnd["requests"]
    dec = result.get("decisions") or {}
    for p in (dec.get("latency") or {}).get("points") or []:
        yield from p["requests"]
    for part in ("cold", "repeated"):
        yield from ((dec.get("cache") or {}).get(part) or {}).get("requests") or []
    for lv in (dec.get("concurrency") or {}).get("levels") or []:
        yield from lv["requests"]


def plan_decisions(args) -> dict:
    """Requests and an estimated time of a decisions run with these options, without sending any.

    Time per request is an assumption, not a measurement: --plan-overhead-ms plus the request's
    estimated input tokens at --plan-prefill-tps. Dataset requests are sized from the measured mean
    payload of each set at 4 characters per token. Throughput levels are counted serially (an upper
    bound: an engine that batches finishes them sooner)."""
    def est(tokens: float) -> float:
        return args.plan_overhead_ms / 1000 + tokens / args.plan_prefill_tps

    def q_tokens(n: int) -> float:
        return len(json.dumps(latency_questions(n))) / CHARS_PER_TOKEN

    rows = [("calibration", 2, est(200 + q_tokens(1)) * 2, "2 requests")]
    lat = [(args.warmup + args.runs, t + q_tokens(qn)) for qn in args.question_counts for t in args.state_sweep]
    rows.append(("latency grid", sum(k for k, _ in lat), sum(k * est(tok) for k, tok in lat),
                 f"{len(lat)} points x ({args.runs} runs + {args.warmup} warm-up)"))
    ref = args.concurrency_state + q_tokens(args.concurrency_questions)
    if args.cache_runs:
        k = args.warmup + 2 * args.cache_runs + 1
        rows.append(("cache test", k, k * est(ref), f"{args.cache_runs} cold + 1 + {args.cache_runs} repeated"))
    if args.concurrency and args.throughput_requests:
        k = sum(max(c, args.warmup) + args.throughput_requests for c in args.concurrency)
        rows.append(("throughput", k, k * est(ref),
                     f"{len(args.concurrency)} levels x {args.throughput_requests} (+ warm-up), serial upper bound"))
    if args.accuracy:
        for name in args.accuracy_sets:
            spec = ACCURACY_SETS[name]
            if spec["kind"] == "classic":
                k = args.accuracy_samples
            else:
                k = 400 if spec["kind"] == "typed" else spec.get("kept", spec["rows"] - len(spec["excluded"]))
                k = min(k, args.accuracy_limit) if args.accuracy_limit else k
            tok = spec["approx_payload_chars"] / CHARS_PER_TOKEN
            rows.append((spec["title"], k, k * est(tok), f"~{tok:,.0f} input tokens each"))
    return {"sections": [{"section": s, "requests": k, "seconds": t, "note": note} for s, k, t, note in rows],
            "requests": sum(r[1] for r in rows), "seconds": sum(r[2] for r in rows),
            "assumptions": {"overhead_ms": args.plan_overhead_ms, "prefill_tps": args.plan_prefill_tps}}


def print_plan(plan: dict) -> None:
    a = plan["assumptions"]
    print(f"Plan of one decisions run (no request sent). Time assumes {a['overhead_ms']:g} ms per request plus "
          f"its input tokens at {a['prefill_tps']:g} tok/s (--plan-overhead-ms, --plan-prefill-tps).")
    w = max(len(s["section"]) for s in plan["sections"])
    for s in plan["sections"]:
        print(f"  {s['section']:<{w}}  {s['requests']:>6,} requests  ~{s['seconds'] / 60:6.1f} min  {s['note']}")
    print(f"  {'total':<{w}}  {plan['requests']:>6,} requests  ~{plan['seconds'] / 60:6.1f} min")


# Options of one mode only; their defaults are None so a mismatch can be refused.
DECISION_ONLY = ("question_counts", "state_sweep", "concurrency_questions", "concurrency_state", "cache_runs",
                 "throughput_requests", "accuracy_sets", "accuracy_samples", "accuracy_seed", "accuracy_limit",
                 "datasets_url", "dataset_cache")
ACCURACY_ONLY = ("accuracy_sets", "accuracy_samples", "accuracy_seed", "accuracy_limit", "datasets_url", "dataset_cache")
DECISION_DEFAULTS = {
    "question_counts": [1, 3, 5], "state_sweep": [200, 2000, 8000], "concurrency_questions": 3,
    "concurrency_state": 2000, "concurrency": [1, 2, 4, 8], "cache_runs": 100, "throughput_requests": 200,
    "accuracy_sets": list(ACCURACY_SETS), "accuracy_samples": 500, "accuracy_seed": 0, "accuracy_limit": 0,
    "datasets_url": DATASETS_URL, "dataset_cache": DATASET_CACHE, "runs": 100, "warmup": 5,
}


def flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def check_mode_options(args) -> None:
    """Refuse options of the other mode, then fill the defaults of this one."""
    if args.decisions:
        if args.context_sweep is not None:
            raise SystemExit("--context-sweep is for chat runs; --decisions measures latency by state length")
        if args.record_chunks:
            raise SystemExit("--record-chunks is for chat runs: a decision streams nothing")
        if args.rounds is not None:
            raise SystemExit("--rounds is for chat runs: decision throughput is closed-loop, see --throughput-requests")
        if not args.accuracy:
            given = [flag(n) for n in ACCURACY_ONLY if getattr(args, n) is not None]
            if given:
                raise SystemExit(f"{', '.join(given)} need --accuracy")
        for name, default in DECISION_DEFAULTS.items():
            if getattr(args, name) is None:
                setattr(args, name, default)
        if args.concurrency_questions > len(LATENCY_QUESTIONS) or args.concurrency_questions < 1:
            raise SystemExit(f"--concurrency-questions is 1 to {len(LATENCY_QUESTIONS)}")
        return
    given = [flag(n) for n in DECISION_ONLY if getattr(args, n) is not None]
    if args.accuracy:
        given.insert(0, "--accuracy")
    if args.plan:
        given.insert(0, "--plan")
    if given:
        raise SystemExit(f"{', '.join(given)} need --decisions")
    if not args.context_sweep and not args.concurrency:
        raise SystemExit("give --context-sweep, --concurrency or both (or --decisions)")
    if args.runs is None:
        args.runs = 3
    if args.warmup is None:
        args.warmup = 1
    if args.rounds is None:
        args.rounds = 5


def parse_extra_body(text: str | None) -> dict:
    extra = json.loads(text) if text else {}
    if not isinstance(extra, dict):
        raise SystemExit("--extra-body must be a JSON object")
    return extra


def cmd_run(args) -> int:
    check_mode_options(args)
    if args.plan:
        print_plan(plan_decisions(args))
        return 0
    key = read_api_key(args.api_key_file)
    ep = Endpoint(args.endpoint, key, timeout=args.timeout)
    extra_body = parse_extra_body(args.extra_body)
    sampler = NoSampler()
    if args.memory_cmd:
        sampler = MemorySampler(args.memory_cmd, args.memory_interval)
        if sampler.sample_once() is None:
            raise SystemExit(f"--memory-cmd failed: {sampler.last_error}")
        sampler.start()
    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": "decisions" if args.decisions else "chat",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                 "python": ".".join(map(str, sys.version_info[:3]))},
        "label": args.label,
        "model": args.model,
        "endpoint": ep.redacted(),
        "started_at": utc_now(),
        "finished_at": None,
        "meta": parse_meta(args.meta),
    }
    memory_settings = {"memory_command": bool(args.memory_cmd),
                       "memory_interval_s": args.memory_interval if args.memory_cmd else None}
    if args.decisions:
        result["settings"] = {"extra_body": extra_body, **memory_settings}
        result["decisions"] = {"manifest": speed_manifest(extra_body), "calibration": None, "latency": None,
                               "cache": None, "concurrency": None, "accuracy": None}
    else:
        result["settings"] = {"temperature": args.temperature, "extra_body": extra_body, **memory_settings,
                              "record_chunks": args.record_chunks}
        result["context_sweep"] = None
        result["concurrency"] = None
    result["memory"] = None
    try:
        if args.decisions:
            dec = result["decisions"]
            runner = DecisionRunner(args, ep, extra_body)
            runner.cal = dec["calibration"] = runner.calibrate()
            log(f"calibrated: {runner.cal['per_sentence_tokens']:.2f} state tokens per sentence "
                f"({runner.cal['method']})")
            dec["latency"] = runner.latency()
            if args.cache_runs:
                dec["cache"] = runner.cache_test()
            if args.concurrency and args.throughput_requests:
                dec["concurrency"] = runner.concurrency()
            if args.accuracy:
                dec["accuracy"] = runner.accuracy()
        else:
            runner = Runner(args, ep, sampler)
            if args.context_sweep:
                result["context_sweep"] = runner.context_sweep()
            if args.concurrency:
                result["concurrency"] = runner.concurrency()
    finally:
        sampler.stop()
    result["finished_at"] = utc_now()
    if any((r.get("extras") or {}).get(SAMPLE_MARK) for r in all_requests(result)):
        result["meta"]["sample"] = True
        log("the server marked its answers as sample data: meta.sample is true")
    if args.memory_cmd:
        if args.decisions:
            apply_decision_memory(result["decisions"], sampler)
        else:
            apply_memory(result, sampler)
        result["memory"] = {"unit": "GiB", "interval_s": args.memory_interval,
                            "errors": sampler.errors, "samples": sampler.samples}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    log(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------
# report: data model
# ---------------------------------------------------------------------------

# CapyCTL site colors. Light is the default everywhere; print is always light.
# Series colors: the first series is the CapyCTL orange; the rest a fixed
# palette (stone, blue-grey, teal, ...), tuned per theme for contrast.
THEMES = {
    "light": {
        "bg": "#fafaf9", "surface": "#f5f5f4", "text": "#1c1917", "muted": "#57534e",
        "rule": "#e7e5e4", "accent": "#ea580c", "accent_text": "#c2410c",
        "palette": ["#ea580c", "#57534e", "#0369a1", "#0f766e", "#6d28d9", "#a16207", "#be185d", "#4d7c0f"],
        "logo": ASSETS / "capyctl-header-light.webp",
    },
    "dark": {
        "bg": "#1c1917", "surface": "#292524", "text": "#e7e5e4", "muted": "#a8a29e",
        "rule": "#44403c", "accent": "#fb923c", "accent_text": "#fb923c",
        "palette": ["#fb923c", "#d6d3d1", "#7dd3fc", "#2dd4bf", "#c4b5fd", "#facc15", "#f472b6", "#a3e635"],
        "logo": ASSETS / "capyctl-header-dark.webp",
    },
}
PALETTE = THEMES["light"]["palette"]
MARKERS = ["circle", "square", "triangle", "diamond", "circle", "square", "triangle", "diamond"]
MPL_MARKERS = {"circle": "o", "square": "s", "triangle": "^", "diamond": "D"}

# (key, title, unit, better, digits hint)
CONTEXT_METRICS = [
    ("decode_tps", "Generation", "tok/s", "higher"),
    ("prompt_tps", "Prompt processing", "tok/s", "higher"),
    ("ttft_s", "Time to first token", "s", "lower"),
    ("tpot_ms", "Time per output token", "ms", "lower"),
    ("total_s", "Total time", "s", "lower"),
    ("memory_peak_gib", "Peak memory", "GiB", "lower"),
    ("output_bytes_per_s", "Output bytes per second", "B/s", "higher"),
]
CONCURRENCY_METRICS = [
    ("aggregate_tps", "Aggregate generation", "tok/s", "higher"),
    ("per_stream_tps", "Per-stream generation, median", "tok/s", "higher"),
    ("ttft", "Time to first token, p50 and p95", "s", "lower"),
    ("draft_acceptance", "Draft acceptance", "%", "higher"),
]


class Series:
    def __init__(self, idx: int, path: Path, data: dict, label: str):
        self.idx = idx
        self.path = path
        self.data = data
        self.label = label

    @property
    def context_points(self) -> list[dict]:
        cs = self.data.get("context_sweep") or {}
        return cs.get("points") or []

    @property
    def levels(self) -> list[dict]:
        cc = self.data.get("concurrency") or {}
        return cc.get("levels") or []

    @property
    def mode(self) -> str:
        return self.data.get("mode") or "chat"

    @property
    def decisions(self) -> dict:
        return self.data.get("decisions") or {}

    @property
    def lat_points(self) -> list[dict]:
        return (self.decisions.get("latency") or {}).get("points") or []

    @property
    def dec_levels(self) -> list[dict]:
        return (self.decisions.get("concurrency") or {}).get("levels") or []

    @property
    def acc_sets(self) -> list[dict]:
        return (self.decisions.get("accuracy") or {}).get("sets") or []

    @property
    def cache(self) -> dict:
        return self.decisions.get("cache") or {}


def is_sample(series: list[Series]) -> bool:
    return any((s.data.get("meta") or {}).get("sample") in (True, "true") for s in series)


def parse_relabel(items) -> dict[str, str]:
    """`--relabel FILE_LABEL=NEW_LABEL` pairs, as {file label: new label}."""
    out: dict[str, str] = {}
    for item in items or []:
        old, sep, new = item.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise SystemExit(f"--relabel expects FILE_LABEL=NEW_LABEL, got {item!r}")
        out[old.strip()] = new.strip()
    return out


def load_series(paths, relabel: dict[str, str] | None = None) -> list[Series]:
    """The results files as series. `relabel` renames series by the label in their file, so the files
    stay as `run` wrote them."""
    series = []
    seen: dict[str, int] = {}
    relabel = dict(relabel or {})
    used: set = set()
    file_labels: list[str] = []
    for i, p in enumerate(paths):
        p = Path(p)
        data = json.loads(p.read_text())
        v = data.get("schema_version")
        if v not in READABLE_SCHEMAS:
            raise SystemExit(f"{p}: schema_version {v!r}, this tool reads "
                             f"{', '.join(map(str, READABLE_SCHEMAS))}")
        label = data.get("label") or p.stem
        file_labels.append(label)
        if label in relabel:
            used.add(label)
            label = data["label"] = relabel[label]
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        series.append(Series(i, p, data, label))
    unknown = sorted(set(relabel) - used)
    if unknown:
        raise SystemExit(f"--relabel: no results file is labelled {', '.join(map(repr, unknown))}; "
                         f"the labels are {', '.join(map(repr, file_labels))}")
    if len({s.mode for s in series}) > 1:
        raise SystemExit("chat and decisions results measure different things; report them separately")
    return series


def is_decisions(series: list[Series]) -> bool:
    return series[0].mode == "decisions"


def ctx_x(point: dict) -> float:
    med = point["summary"]["prompt_tokens"]["median"]
    return med if med else point["target_prompt_tokens"]


def ctx_value(point: dict, key: str) -> dict:
    return point["summary"].get(key) or {"median": None, "min": None, "max": None, "n": 0}


def cc_value(level: dict, key: str) -> dict:
    s = level["summary"]
    if key == "draft_acceptance":
        v = s.get("draft_acceptance")
        v = v * 100 if v is not None else None
        return {"median": v, "min": v, "max": v, "n": 1 if v is not None else 0}
    if key in ("ttft_p50_s", "ttft_p95_s"):
        v = s.get(key)
        return {"median": v, "min": v, "max": v, "n": 1 if v is not None else 0}
    return s.get(key) or {"median": None, "min": None, "max": None, "n": 0}


def has_metric_ctx(series: list[Series], key: str) -> bool:
    return any(ctx_value(p, key)["median"] is not None for s in series for p in s.context_points)


def has_metric_cc(series: list[Series], key: str) -> bool:
    keys = ("ttft_p50_s",) if key == "ttft" else (key,)
    return any(cc_value(lv, k)["median"] is not None for s in series for lv in s.levels for k in keys)


def chart_specs(series: list[Series]) -> list[dict]:
    """Every chart in the report: the same list drives SVG, PNG and CSV."""
    if is_decisions(series):
        return decision_chart_specs(series)
    specs = []
    for key, title, unit, better in CONTEXT_METRICS:
        if not has_metric_ctx(series, key):
            continue
        lines = []
        for s in series:
            pts = [(ctx_x(p), ctx_value(p, key)["median"], p["label"]) for p in s.context_points]
            pts = [pt for pt in pts if pt[1] is not None]
            if pts:
                lines.append({"series": s, "name": s.label, "dash": False, "points": pts})
        specs.append({"id": f"context-{key}", "section": "context", "metric": key, "title": title,
                      "unit": unit, "better": better, "xlog": True,
                      "xlabel": "Prompt tokens", "lines": lines})
    # One stream count (C1 only) has no line against streams: its numbers are in the tables.
    several_levels = len({lv["concurrency"] for s in series for lv in s.levels}) > 1
    for key, title, unit, better in CONCURRENCY_METRICS:
        if not several_levels or not has_metric_cc(series, key):
            continue
        lines = []
        for s in series:
            sub = [("ttft_p50_s", " p50", False), ("ttft_p95_s", " p95", True)] if key == "ttft" else [(key, "", False)]
            for k, suffix, dash in sub:
                pts = [(lv["concurrency"], cc_value(lv, k)["median"], str(lv["concurrency"])) for lv in s.levels]
                pts = [pt for pt in pts if pt[1] is not None]
                if pts:
                    lines.append({"series": s, "name": s.label + suffix, "dash": dash, "points": pts})
        specs.append({"id": f"concurrency-{key}", "section": "concurrency", "metric": key, "title": title,
                      "unit": unit, "better": better, "xlog": False,
                      "xlabel": "Concurrent streams", "lines": lines})
    return specs


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(v, unit: str = "") -> str:
    if v is None:
        return "–"
    if unit == "%":
        return f"{v:.1f}%"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def pct_change(v, base) -> str:
    if v is None or base in (None, 0):
        return ""
    p = (v - base) / base * 100
    return f"{p:+.1f}%"


# ---------------------------------------------------------------------------
# SVG charts (no dependencies)
# ---------------------------------------------------------------------------

def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m in (1, 2, 2.5, 5, 10):
        if span / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 10))
        if t >= hi:
            break
        t += step
    return ticks


def tick_fmt(v: float, step: float) -> str:
    """Axis label with just enough decimals for the tick step; 20k for 20,000."""
    if step >= 1000:
        return f"{v / 1000:g}k" if v else "0"
    decimals = 0
    while decimals < 6 and abs(step * 10 ** decimals - round(step * 10 ** decimals)) > 1e-6:
        decimals += 1
    return f"{v:,.{decimals}f}"


def log_ticks(xmin: float, xmax: float) -> list[float]:
    """Ticks at 1000 * 2^n tokens (0.5k, 1k, 2k, ...) covering the data."""
    lo = math.floor(math.log2(max(xmin, 1) / 1000))
    hi = math.ceil(math.log2(max(xmax, 1) / 1000))
    ticks = [1000 * 2 ** e for e in range(lo, hi + 1)]
    while len(ticks) > 9:
        ticks = ticks[::2]
    return ticks


def marker_svg(shape: str, x: float, y: float, cls: str, tip: str) -> str:
    t = f"<title>{html.escape(tip)}</title>"
    if shape == "square":
        return f'<rect class="{cls}" x="{x - 3.4:.1f}" y="{y - 3.4:.1f}" width="6.8" height="6.8">{t}</rect>'
    if shape == "triangle":
        return (f'<polygon class="{cls}" points="{x:.1f},{y - 4.4:.1f} {x + 4:.1f},{y + 3:.1f} '
                f'{x - 4:.1f},{y + 3:.1f}">{t}</polygon>')
    if shape == "diamond":
        return (f'<polygon class="{cls}" points="{x:.1f},{y - 4.4:.1f} {x + 4.4:.1f},{y:.1f} '
                f'{x:.1f},{y + 4.4:.1f} {x - 4.4:.1f},{y:.1f}">{t}</polygon>')
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="3.6">{t}</circle>'


def svg_chart(spec: dict) -> str:
    W, H = 460, 280
    ml, mr, mt, mb = 54, 14, 12, 42
    pw, ph = W - ml - mr, H - mt - mb
    xs = [p[0] for ln in spec["lines"] for p in ln["points"]]
    ys = [p[1] for ln in spec["lines"] for p in ln["points"]]
    if not xs:
        return ""
    unit = spec["unit"]
    if spec.get("xdomain"):
        # a fixed linear axis (reliability: confidence 0 to 100%)
        x0, x1 = spec["xdomain"]
        xticks = spec["xticks"]
        def X(v):
            return ml + (v - x0) / (x1 - x0) * pw
        xlabels = [f"{t:g}" for t in xticks]
    elif spec["xlog"]:
        xticks = log_ticks(min(xs), max(xs))
        x0, x1 = math.log2(min(xticks + xs)), math.log2(max(xticks + xs))
        def X(v):
            return ml + (math.log2(v) - x0) / ((x1 - x0) or 1) * pw
        xlabels = [size_label(t) for t in xticks]
    else:
        xticks = sorted(set(int(v) for v in xs))
        x0, x1 = min(xticks), max(xticks)
        if x0 == x1:
            x0, x1 = x0 - 1, x1 + 1
        def X(v):
            return ml + (v - x0) / (x1 - x0) * pw
        xlabels = [str(t) for t in xticks]
    if spec.get("ylog") and min(ys) > 0:
        e0, e1 = math.floor(math.log10(min(ys))), math.ceil(math.log10(max(ys)))
        if e1 == e0:
            e1 += 1
        yticks = [10.0 ** e for e in range(e0, e1 + 1)]
        def Y(v):
            return mt + ph - (math.log10(v) - e0) / (e1 - e0) * ph
        def ytick_label(t):
            return f"{t:g}"
    else:
        ymax = spec["ydomain"][1] if spec.get("ydomain") else (max(ys) * 1.08 if max(ys) > 0 else 1)
        yticks = nice_ticks(0, ymax)
        y1 = yticks[-1] or 1
        ystep = yticks[1] - yticks[0] if len(yticks) > 1 else 1
        def Y(v):
            return mt + ph - v / y1 * ph
        def ytick_label(t):
            return tick_fmt(t, ystep)
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{html.escape(spec["title"])} ({html.escape(unit)})" xmlns="http://www.w3.org/2000/svg">']
    for t in yticks:
        y = Y(t)
        out.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end">{ytick_label(t)}</text>')
    for t, lab in zip(xticks, xlabels):
        x = X(t)
        out.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="{mt + ph}" y2="{mt + ph + 4}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt + ph + 18}" text-anchor="middle">{lab}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{mt + ph}" y2="{mt + ph}"/>')
    out.append(f'<text class="axlabel" x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">'
               f'{html.escape(spec["xlabel"])}</text>')
    out.append(f'<text class="axlabel" transform="translate(13 {mt + ph / 2:.1f}) rotate(-90)" '
               f'text-anchor="middle">{html.escape(unit)}</text>')
    if spec.get("diagonal"):
        out.append(f'<path class="ln diag" d="M{X(x0):.1f},{Y(x0):.1f} L{X(x1):.1f},{Y(x1):.1f}" '
                   f'stroke-dasharray="2 4"/>')
    for ln in spec["lines"]:
        i = ln["series"].idx % len(PALETTE)
        pts = sorted(ln["points"])
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for k, (x, y, _) in enumerate(pts))
        dash = ' stroke-dasharray="5 4"' if ln["dash"] else ""
        out.append(f'<path class="ln s{i}" d="{d}"{dash}/>')
        for x, y, lab in pts:
            tip = f"{ln['name']}: {fmt(y, unit)} {'' if unit == '%' else unit} at {lab}"
            out.append(marker_svg(MARKERS[i], X(x), Y(y), f"mk s{i}", tip))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# PNG charts (matplotlib, optional)
# ---------------------------------------------------------------------------

def bar_top(vmax: float, label_pts: float, axes_pts: float) -> float:
    """Top of a summary bar panel, so the tallest bar's labels (`label_pts` high, in points)
    end inside the panel (`axes_pts` high) and clear of the unit caption above it."""
    share = min(label_pts / axes_pts, 0.8) if axes_pts > 0 else 0.5
    return vmax / (1 - share) * 1.04


def y_top(values) -> float:
    """Top of a linear y axis from 0: 8% above the highest value, so no point sits on the edge."""
    vals = [v for v in values if v is not None]
    top = max(vals) * 1.08 if vals else 0
    return top if top > 0 else 1


def load_matplotlib(what: str):
    """pyplot, or None (with a message naming what was skipped) without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log(f"matplotlib is not installed: {what} skipped. Render them with\n"
            "  uv run --with matplotlib python3 capyctl_bench.py report ...")
        return None
    return plt


def mpl_theme(plt, theme: str) -> dict:
    from matplotlib import font_manager
    T = THEMES[theme]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Inter", "Helvetica", "Arial", "DejaVu Sans") if f in installed), "sans-serif")
    plt.rcParams.update({
        "font.family": family,
        "font.size": 11,
        "axes.facecolor": T["surface"], "figure.facecolor": T["bg"],
        "axes.edgecolor": T["rule"], "axes.labelcolor": T["muted"],
        "xtick.color": T["muted"], "ytick.color": T["muted"],
        "text.color": T["text"], "grid.color": T["rule"],
        "legend.facecolor": T["surface"], "legend.edgecolor": T["rule"],
    })
    return T


def mpl_axes_x(ax, spec: dict) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    xs = [p[0] for ln in spec["lines"] for p in ln["points"]]
    if spec.get("xdomain"):
        ax.set_xlim(*spec["xdomain"])
        ax.set_xticks(spec["xticks"])
    elif spec["xlog"]:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(log_ticks(min(xs), max(xs))))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: size_label(v)))
    else:
        ax.set_xticks(sorted(set(int(x) for x in xs)))


def write_pngs(specs: list[dict], out_dir: Path, title: str, theme: str = "light",
               sample: bool = False) -> list[Path]:
    plt = load_matplotlib("charts/*.png")
    if plt is None:
        return []
    charts = out_dir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    T = mpl_theme(plt, theme)
    palette = T["palette"]
    written = []
    for spec in specs:
        if not spec["lines"]:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        for ln in spec["lines"]:
            i = ln["series"].idx % len(PALETTE)
            pts = sorted(ln["points"])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=palette[i],
                    marker=MPL_MARKERS[MARKERS[i]], markersize=5, linewidth=2,
                    linestyle="--" if ln["dash"] else "-", label=ln["name"])
        mpl_axes_x(ax, spec)
        if spec.get("diagonal"):
            lo, hi = spec["xdomain"]
            ax.plot([lo, hi], [lo, hi], color=T["muted"], linewidth=1, linestyle=":", label="perfectly calibrated")
        ax.set_ylim(*(spec.get("ydomain") or (0, y_top(p[1] for ln in spec["lines"] for p in ln["points"]))))
        ax.grid(True, axis="y", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["unit"])
        ax.set_title(f"{spec['title']} ({spec['unit']}, {spec['better']} is better)",
                     loc="left", fontsize=13, fontweight="bold", color=T["text"])
        ax.legend(loc="best", fontsize=9, frameon=True)
        if sample:
            fig.text(0.99, 0.985, SAMPLE_BANNER, ha="right", va="top", fontsize=12, fontweight="bold",
                     color="#ffffff", bbox={"boxstyle": "square,pad=0.4", "facecolor": "#b91c1c", "edgecolor": "none"})
        fig.text(0.99, 0.01, f"{title} · measured through CapyCTL · {TOOL_NAME} {TOOL_VERSION}",
                 ha="right", va="bottom", fontsize=7.5, color=T["muted"])
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = charts / f"{spec['id']}.png"
        fig.savefig(path, facecolor=T["bg"])
        plt.close(fig)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Tables, CSV, Markdown
# ---------------------------------------------------------------------------

CTX_TABLE = [
    ("prompt_tokens", "Prompt tokens", "tok"),
    ("decode_tps", "Generation", "tok/s"),
    ("prompt_tps", "Prompt processing", "tok/s"),
    ("ttft_s", "TTFT", "s"),
    ("tpot_ms", "TPOT", "ms"),
    ("total_s", "Total", "s"),
    ("output_bytes_per_s", "Output", "B/s"),
    ("output_chars_per_s", "Output", "chars/s"),
    ("memory_peak_gib", "Peak memory", "GiB"),
]
CC_TABLE = [
    ("aggregate_tps", "Aggregate", "tok/s"),
    ("per_stream_tps", "Per stream", "tok/s"),
    ("ttft_p50_s", "TTFT p50", "s"),
    ("ttft_p95_s", "TTFT p95", "s"),
    ("tpot_ms", "TPOT", "ms"),
    ("total_s", "Total", "s"),
    ("draft_acceptance", "Draft acceptance", "%"),
    ("memory_peak_gib", "Peak memory", "GiB"),
]


def csv_rows(series: list[Series]):
    if is_decisions(series):
        yield from decision_csv_rows(series)
        return
    for s in series:
        for p in s.context_points:
            for key, _, unit in CTX_TABLE:
                v = ctx_value(p, key)
                yield [s.label, "context", p["label"], ctx_x(p), key, unit,
                       v["median"], v["min"], v["max"], v["n"]]
        for lv in s.levels:
            for key, _, unit in CC_TABLE:
                v = cc_value(lv, key)
                yield [s.label, "concurrency", str(lv["concurrency"]), lv["concurrency"], key, unit,
                       v["median"], v["min"], v["max"], v["n"]]


def write_csv(series: list[Series], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "section", "point", "x", "metric", "unit", "median", "min", "max", "n"])
        for row in csv_rows(series):
            w.writerow(["" if c is None else (f"{c:.6g}" if isinstance(c, float) else c) for c in row])


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def md_head(series: list[Series], title: str, how: str) -> list[str]:
    """Title, sample warning, what a cell is, and the series table."""
    out = [f"# {title}", ""]
    if is_sample(series):
        out += [f"> **{SAMPLE_BANNER}.** These numbers come from the test fake server.", ""]
    out.append(f"Measured through CapyCTL with {TOOL_NAME} {TOOL_VERSION}. {how}")
    out.append("")
    out.append(md_table(["Series", "Model", "Measured", "Meta"], [
        [s.label, f"`{s.data.get('model', '')}`", (s.data.get("started_at") or "")[:10],
         ", ".join(f"{k}={v}" for k, v in (s.data.get("meta") or {}).items()) or "–"]
        for s in series]))
    return out


def summary_md(series: list[Series], title: str) -> str:
    if is_decisions(series):
        return decisions_summary_md(series, title)
    base = series[0]
    out = md_head(series, title, "Each cell is the median of the runs at that point; percentages compare "
                  f"with {base.label}.")
    labels = sorted({p["label"]: p["target_prompt_tokens"] for s in series for p in s.context_points}.items(),
                    key=lambda kv: kv[1])
    if labels:
        out += ["", "## Context sweep"]
        for key, title_, unit, better in CONTEXT_METRICS:
            if not has_metric_ctx(series, key):
                continue
            rows = []
            for lab, _ in labels:
                row = [lab]
                base_v = next((ctx_value(p, key)["median"] for p in base.context_points if p["label"] == lab), None)
                for s in series:
                    v = next((ctx_value(p, key)["median"] for p in s.context_points if p["label"] == lab), None)
                    cell = fmt(v)
                    if s is not base and v is not None and base_v:
                        cell += f" ({pct_change(v, base_v)})"
                    row.append(cell)
                rows.append(row)
            out += ["", f"### {title_} ({unit}, {better} is better)", "",
                    md_table(["Context"] + [s.label for s in series], rows)]
    levels = sorted({lv["concurrency"] for s in series for lv in s.levels})
    if levels:
        out += ["", "## Concurrency"]
        for key, title_, unit in [(k, t, u) for k, t, u in CC_TABLE if k not in ("total_s",)]:
            if not any(cc_value(lv, key)["median"] is not None for s in series for lv in s.levels):
                continue
            rows = []
            for c in levels:
                row = [str(c)]
                base_v = next((cc_value(lv, key)["median"] for lv in base.levels if lv["concurrency"] == c), None)
                for s in series:
                    v = next((cc_value(lv, key)["median"] for lv in s.levels if lv["concurrency"] == c), None)
                    cell = fmt(v, unit)
                    if s is not base and v is not None and base_v:
                        cell += f" ({pct_change(v, base_v)})"
                    row.append(cell)
                rows.append(row)
            out += ["", f"### {title_} ({unit})", "", md_table(["Streams"] + [s.label for s in series], rows)]
    out += ["", f"Produced by `{TOOL_NAME} report`; settings and per-run spread are in report.html.", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
%(theme_css)s
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%%}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-feature-settings:"tnum" 1}
.wrap{max-width:1200px;margin:0 auto;padding:0 24px 64px}
header.top{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 0;border-bottom:1px solid var(--rule)}
header.top img{height:32px;width:auto;display:block}
.actions{display:flex;gap:8px;flex-wrap:wrap}
.actions a,.actions button{font:inherit;font-size:13px;color:var(--text);background:var(--surface);border:1px solid var(--rule);border-radius:6px;padding:6px 12px;text-decoration:none;cursor:pointer}
.actions a:hover,.actions button:hover{border-color:var(--accent);color:var(--accent-text)}
.actions button.theme{display:inline-flex;align-items:center;justify-content:center;padding:6px 9px}
.theme svg{width:16px;height:16px;display:block;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round}
.theme .sun,:root[data-theme="dark"] .theme .moon{display:none}
:root[data-theme="dark"] .theme .sun{display:block}
.logo-dark,:root[data-theme="dark"] .logo-light{display:none!important}
:root[data-theme="dark"] .logo-dark{display:block!important}
h1{font-size:28px;line-height:1.2;margin:32px 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:44px 0 6px;padding-top:4px}
h3{font-size:15px;margin:0 0 8px}
p.lede{color:var(--muted);margin:0 0 4px}
.sample{background:#b91c1c;color:#fff;font-weight:700;letter-spacing:.04em;text-align:center;padding:10px 14px;border-radius:6px;margin:16px 0 0;-webkit-print-color-adjust:exact;print-color-adjust:exact}
.note{border-left:3px solid var(--accent);background:var(--surface);padding:10px 14px;border-radius:0 6px 6px 0;margin:20px 0;color:var(--text)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;margin:12px 0 4px;font-size:14px}
.legend span{display:inline-flex;align-items:center;gap:8px}
.legend i{display:inline-block;width:18px;height:3px;border-radius:2px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:14px 14px 8px;min-width:0}
.card h3 small{color:var(--muted);font-weight:400}
svg.chart{width:100%%;height:auto;display:block;font-family:inherit}
svg .grid{stroke:var(--rule);stroke-width:1}
svg .axis{stroke:var(--muted);stroke-width:1}
svg .tick{fill:var(--muted);font-size:12px}
svg .axlabel{fill:var(--muted);font-size:12px}
svg .ln{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
svg .ln.diag{stroke:var(--muted);stroke-width:1.2}
%(series_css)s
.meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px;margin-top:16px}
.meta dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 12px;font-size:13px}
.meta dt{color:var(--muted)}
.meta dd{margin:0;overflow-wrap:anywhere}
.tablewrap{overflow-x:auto;margin:12px 0 24px;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%%;font-size:13px}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule)}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--muted);font-weight:500;vertical-align:bottom}
thead th small{display:block;font-weight:400}
tbody tr:last-child td{border-bottom:0}
td small{display:block;color:var(--muted);font-size:11px}
td.err{color:#f87171}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:baseline}
.method{color:var(--text);max-width:860px}
.method li{margin:4px 0}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em}
footer{margin-top:48px;padding-top:16px;border-top:1px solid var(--rule);color:var(--muted);font-size:13px}
@media screen and (max-width:760px){.wrap{padding:0 16px 48px}.grid2{grid-template-columns:minmax(0,1fr)}h1{font-size:23px}
header.top{flex-direction:column;align-items:flex-start}svg .tick,svg .axlabel{font-size:14px}}
@media print{
%(print_css)s
body{font-size:11pt}
.wrap{max-width:none;padding:0}
.actions{display:none}
:root[data-theme="dark"] .logo-light{display:block!important}
:root[data-theme="dark"] .logo-dark{display:none!important}
.grid2{gap:10px}
.card,.tablewrap,.meta>div{break-inside:avoid}
h2,h3{break-after:avoid}
.tablewrap{overflow:visible}
table{font-size:7.5pt}
th,td{padding:4px 5px}
a{color:inherit}
@page{margin:14mm}
}
"""


def theme_vars(name: str) -> str:
    T = THEMES[name]
    out = (f"--bg:{T['bg']};--surface:{T['surface']};--text:{T['text']};--muted:{T['muted']};"
           f"--rule:{T['rule']};--accent:{T['accent']};--accent-text:{T['accent_text']};color-scheme:{name};")
    return out + "".join(f"--c{i}:{c};" for i, c in enumerate(T["palette"]))


def page_css(extra: str = "") -> str:
    """Light by default (whatever the OS prefers), dark when the page sets data-theme="dark"."""
    theme = f':root{{{theme_vars("light")}}}:root[data-theme="dark"]{{{theme_vars("dark")}}}'
    print_css = f':root,:root[data-theme="dark"]{{{theme_vars("light")}}}'
    rules = "".join(
        f"svg .ln.s{i}{{stroke:var(--c{i})}}svg .mk.s{i}{{fill:var(--c{i});stroke:var(--surface);stroke-width:1}}"
        f"svg .bar.s{i}{{fill:var(--c{i})}}.sw{i}{{background:var(--c{i})}}"
        for i in range(len(PALETTE)))
    return CSS % {"theme_css": theme, "print_css": print_css, "series_css": rules} + extra


THEME_KEY = "capyctl-bench-theme"
# Runs before the body renders: light unless the viewer chose dark before.
THEME_HEAD = ("<script>(function(){var t='light';try{var s=localStorage.getItem('%s');"
              "if(s==='dark'||s==='light')t=s}catch(e){}"
              "document.documentElement.setAttribute('data-theme',t)})();"
              "function capyToggleTheme(){var r=document.documentElement;"
              "var t=r.getAttribute('data-theme')==='dark'?'light':'dark';r.setAttribute('data-theme',t);"
              "try{localStorage.setItem('%s',t)}catch(e){}}</script>") % (THEME_KEY, THEME_KEY)
THEME_BUTTON = (
    '<button type="button" class="theme" onclick="capyToggleTheme()" '
    'aria-label="Switch between light and dark theme" title="Light or dark theme">'
    '<svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>'
    '<svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
    '<path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>'
    '</button>')


def esc(v) -> str:
    return html.escape(str(v))


def cell(v: dict, unit: str = "") -> str:
    if v["median"] is None:
        return "<td>–</td>"
    rng = ""
    if v["n"] and v["n"] > 1 and v["min"] is not None and v["max"] != v["min"]:
        rng = f"<small>{fmt(v['min'], unit)}–{fmt(v['max'], unit)}</small>"
    return f"<td>{fmt(v['median'], unit)}{rng}</td>"


def series_tables(s: Series) -> str:
    if s.mode == "decisions":
        return decision_series_tables(s)
    out = []
    i = s.idx % len(PALETTE)
    if s.context_points:
        cols = [c for c in CTX_TABLE if c[0] != "memory_peak_gib" or
                any(ctx_value(p, "memory_peak_gib")["median"] is not None for p in s.context_points)]
        accept = any(p["summary"].get("draft_acceptance") is not None for p in s.context_points)
        head = "".join(f"<th>{esc(t)}<small>{esc(u)}</small></th>" for _, t, u in cols)
        if accept:
            head += "<th>Draft acceptance<small>%</small></th>"
        rows = []
        for p in s.context_points:
            sm = p["summary"]
            tds = "".join(cell(ctx_value(p, k), u) for k, _, u in cols)
            if accept:
                a = sm.get("draft_acceptance")
                tds += f"<td>{fmt(a * 100 if a is not None else None, '%')}</td>"
            ok = sm["requests"] - sm["errors"]
            runs = f'<td class="err">{ok}/{sm["requests"]}</td>' if sm["errors"] else f"<td>{ok}/{sm['requests']}</td>"
            rows.append(f"<tr><td>{esc(p['label'])}</td>{tds}{runs}</tr>")
        out.append(f'<h3><span class="swatch sw{i}"></span>{esc(s.label)}: context sweep</h3>'
                   f'<div class="tablewrap"><table><thead><tr><th>Context</th>{head}<th>Runs<small>ok/sent</small></th>'
                   f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    if s.levels:
        cols = [c for c in CC_TABLE if any(cc_value(lv, c[0])["median"] is not None for lv in s.levels)]
        head = "".join(f"<th>{esc(t)}<small>{esc(u)}</small></th>" for _, t, u in cols)
        rows = []
        for lv in s.levels:
            sm = lv["summary"]
            tds = "".join(cell(cc_value(lv, k), u) for k, _, u in cols)
            ok = sm["requests"] - sm["errors"]
            req = f'<td class="err">{ok}/{sm["requests"]}</td>' if sm["errors"] else f"<td>{ok}/{sm['requests']}</td>"
            rows.append(f"<tr><td>{lv['concurrency']}</td>{tds}{req}</tr>")
        out.append(f'<h3><span class="swatch sw{i}"></span>{esc(s.label)}: concurrency</h3>'
                   f'<div class="tablewrap"><table><thead><tr><th>Streams</th>{head}<th>Requests<small>ok/sent</small></th>'
                   f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return "".join(out)


def meta_card(s: Series) -> str:
    d = s.data
    i = s.idx % len(PALETTE)
    rows = [("Model", d.get("model")), ("Endpoint", d.get("endpoint")),
            ("Started", d.get("started_at")), ("Finished", d.get("finished_at"))]
    rows += [(k, v) for k, v in (d.get("meta") or {}).items()]
    st = d.get("settings") or {}
    if s.mode == "chat":
        rows.append(("Temperature", st.get("temperature")))
    if st.get("extra_body"):
        rows.append(("Extra body", json.dumps(st["extra_body"])))
    rows += decision_meta_rows(d)
    cs = (d.get("context_sweep") or {}).get("settings")
    if cs:
        rows.append(("Context sweep", f"{len(cs['targets'])} points, {cs['runs']} runs each, "
                                      f"max_tokens {cs['max_tokens']}, warm-up {cs['warmup']}"))
    cc = (d.get("concurrency") or {}).get("settings")
    if cc:
        rows.append(("Concurrency", f"streams {','.join(map(str, cc['levels']))}, {cc['rounds']} rounds, "
                                    f"max_tokens {cc['max_tokens']}, prompts {cc['prompt_set']}"))
    mem = d.get("memory")
    rows.append(("Memory", f"sampled every {mem['interval_s']} s ({len(mem['samples'])} samples)" if mem else "not sampled"))
    tool = d.get("tool") or {}
    rows.append(("Tool", f"{tool.get('name', TOOL_NAME)} {tool.get('version', '?')}"))
    dl = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v if v is not None else '–')}</dd>" for k, v in rows)
    return (f'<div class="card"><h3><span class="swatch sw{i}"></span>{esc(s.label)}</h3>'
            f"<dl>{dl}</dl></div>")


def settings_differ(series: list[Series]) -> list[str]:
    notes = []
    def key(s, path):
        obj = s.data
        for p in path:
            obj = (obj or {}).get(p) if isinstance(obj, dict) else None
        return json.dumps(obj, sort_keys=True)
    checks = [("model", ("model",)), ("temperature", ("settings", "temperature")),
              ("extra body", ("settings", "extra_body")),
              ("context sweep settings", ("context_sweep", "settings")),
              ("concurrency settings", ("concurrency", "settings")),
              ("latency settings", ("decisions", "latency", "settings")),
              ("concurrency settings", ("decisions", "concurrency", "settings")),
              ("accuracy settings", ("decisions", "accuracy", "settings"))]
    for name, path in checks:
        if len({key(s, path) for s in series}) > 1:
            notes.append(name)
    return notes


def method_html(series: list[Series]) -> str:
    items = decision_method_items() if is_decisions(series) else [
        "Every request is a streaming chat completion sent to the CapyCTL inference endpoint "
        "with <code>stream_options.include_usage</code>; token counts come from the final usage chunk.",
        "<b>Time to first token</b> (TTFT): request sent to the first chunk carrying content or reasoning text.",
        "<b>Generation</b> tok/s: (completion tokens − 1) / (last token chunk − first token chunk). "
        "<b>TPOT</b> is the inverse, in milliseconds per token.",
        "<b>Prompt processing</b> tok/s: prompt tokens / TTFT. It includes queueing and the first decode step, "
        "so it understates the engine's raw prefill rate at short prompts.",
        "<b>Total time</b>: request sent to the end of the stream.",
        "<b>Output bytes per second</b>: UTF-8 bytes of content and reasoning text over the generation window.",
        "<b>Peak memory</b>: the highest value the memory command printed while the request or round ran.",
        "Context sweep: one request at a time. Each request gets its own deterministic filler text "
        "(a different seed per run), so no run reuses another's prefix cache. The prompt length is sized from "
        "two calibration requests; the table shows the prompt tokens the engine reported. k is 1000 tokens.",
        "Concurrency: N requests start together from a fixed prompt set; <b>aggregate</b> tok/s is the round's "
        "completion tokens over the time from the first request sent to the last stream finished. "
        "Per-stream tok/s is the median over every request at that level; TTFT p50/p95 likewise.",
        "Each chart point is a median. Tables show the median with the min–max range across runs below it.",
    ]
    diff = settings_differ(series)
    warn = ""
    if diff:
        warn = (f'<p class="note">The series differ in {esc(", ".join(diff))}. Compare them with that in mind.</p>')
    return f'<ul class="method">{"".join(f"<li>{i}</li>" for i in items)}</ul>{warn}'


def logo_tags() -> str:
    """Both CapyCTL logos as data URIs; CSS shows the one that fits the theme."""
    tags = []
    for theme in ("light", "dark"):
        path = THEMES[theme]["logo"]
        if path.exists():
            uri = "data:image/webp;base64," + base64.b64encode(path.read_bytes()).decode()
            tags.append(f'<img class="logo-{theme}" src="{uri}" alt="CapyCTL">')
    return "".join(tags) or "<strong>CapyCTL</strong>"


def page_head(title: str, css: str) -> str:
    return f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
{THEME_HEAD}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap">
<style>{css}</style>
</head>"""


def sample_banner_html(series: list[Series]) -> str:
    if not is_sample(series):
        return ""
    return f'<p class="sample" role="note">{esc(SAMPLE_BANNER)}</p>'


def report_html(series: list[Series], specs: list[dict], title: str, has_zip: bool,
                has_summary: bool = False) -> str:
    legend = "".join(
        f'<span><i class="sw{s.idx % len(PALETTE)}"></i>{esc(s.label)}</span>' for s in series)
    models = sorted({s.data.get("model", "") for s in series})
    dates = sorted({(s.data.get("started_at") or "")[:10] for s in series} - {""})
    lede = f"{esc(', '.join(models))} · {esc(' to '.join([dates[0], dates[-1]]) if len(dates) > 1 else (dates[0] if dates else ''))}"

    def section(name: str, heading: str, intro: str) -> str:
        cards = []
        for sp in specs:
            if sp["section"] != name:
                continue
            svg = svg_chart(sp)
            if not svg:
                continue
            extra = " Dashed lines are p95." if any(ln["dash"] for ln in sp["lines"]) else ""
            cards.append(f'<div class="card"><h3>{esc(sp["title"])} <small>{esc(sp["unit"])}, '
                         f'{esc(sp["better"])} is better.{extra}</small></h3>{svg}</div>')
        if not cards:
            return ""
        return (f'<h2 id="{name}">{esc(heading)}</h2><p class="lede">{intro}</p>'
                f'<div class="legend">{legend}</div><div class="grid2">{"".join(cards)}</div>')

    if is_decisions(series):
        charts = (section("latency", "Latency",
                          "One request at a time, state length growing; client side, request sent to the whole "
                          "answer read. x axis is log scale.")
                  + cache_section_html(series)
                  + section("concurrency", "Throughput",
                            "Closed loop: N streams, each sending its next request as soon as its last one returns; "
                            "every request its own state.")
                  + quality_section_html(series)
                  + section("calibration", "Calibration",
                            "Reliability: each confidence bin's accuracy at its mean confidence (10 equal-width bins of "
                            "the top label's probability). The dotted diagonal is perfect calibration."))
    else:
        charts = (section("context", "Context sweep",
                          "One request at a time, prompt length growing. x axis is log scale.")
                  + section("concurrency", "Concurrency", "N streams started together from a fixed prompt set."))
    tables = "".join(series_tables(s) for s in series)
    links = [THEME_BUTTON, '<button type="button" onclick="window.print()">Print or save as PDF</button>']
    if has_summary:
        links.append('<a href="summary/summary.html">Summary</a>')
    links += ['<a href="summary.md">summary.md</a>', '<a href="data.csv">data.csv</a>']
    if has_zip:
        links.append('<a href="report.zip">report.zip</a>')
    return f"""{page_head(title, page_css())}
<body>
<div class="wrap">
<header class="top">{logo_tags()}<nav class="actions">{"".join(links)}</nav></header>
<h1>{esc(title)}</h1>
<p class="lede">{lede}</p>
{sample_banner_html(series)}
<p class="note">Measured through CapyCTL: every request went to the CapyCTL inference endpoint, not to the engine directly.
Each series is one results file; the same prompts, settings and host are what make them comparable.</p>
{charts}
<h2 id="tables">Full tables</h2>
<p class="lede">Median per point, with the min–max range across runs below it.</p>
{tables}
<h2 id="runs">Runs</h2>
<div class="meta">{"".join(meta_card(s) for s in series)}</div>
<h2 id="method">Method</h2>
{method_html(series)}
<footer>Generated {esc(utc_now())} by {TOOL_NAME} {TOOL_VERSION} from {len(series)} results file{'s' if len(series) != 1 else ''}.</footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Shareable summary (one page: HTML, wide and tall PNGs in both themes)
# ---------------------------------------------------------------------------

SUMMARY_SIZES = {"wide": (1200, 675), "tall": (1080, 1350)}


def headline_groups(series: list[Series]) -> tuple[str, list[dict]]:
    """Bars of the headline panel: aggregate tok/s at the fewest and most streams all series ran.

    These use the prompt set (real prompts). Without a concurrency sweep: one-stream generation
    tok/s at ~1k and at the largest common context of the sweep (filler prompts, which a drafter
    accepts easily, so they stay out of the headline when real prompts exist).
    """
    groups = []
    levels = None
    for s in series:
        lv = {l["concurrency"] for l in s.levels if cc_value(l, "aggregate_tps")["median"] is not None}
        levels = lv if levels is None else levels & lv
    if levels:
        for c in dict.fromkeys([min(levels), max(levels)]):
            groups.append({"label": f"{c} stream{'s' if c > 1 else ''}",
                           "bars": [(s, next(cc_value(l, "aggregate_tps")["median"] for l in s.levels
                                             if l["concurrency"] == c)) for s in series]})
        return "Generation, real prompts", groups

    def ctx_val(s, lab):
        return next((ctx_value(p, "decode_tps")["median"] for p in s.context_points if p["label"] == lab), None)

    common = None
    for s in series:
        labs = {p["label"]: p["target_prompt_tokens"] for p in s.context_points
                if ctx_value(p, "decode_tps")["median"] is not None}
        common = labs if common is None else {k: v for k, v in common.items() if k in labs}
    if common:
        short = min(common, key=lambda k: abs(math.log(common[k] / 1000)))
        long_ = max(common, key=lambda k: common[k])
        for lab in dict.fromkeys([short, long_]):
            groups.append({"label": f"{lab} prompt", "bars": [(s, ctx_val(s, lab)) for s in series]})
        return "Generation, one stream", groups
    return "", []


def summary_panels(series: list[Series]) -> list[dict]:
    """2 to 4 panels, chosen from what the results contain."""
    if is_decisions(series):
        return decision_summary_panels(series)
    specs = {sp["id"]: sp for sp in chart_specs(series)}
    panels = []
    title, groups = headline_groups(series)
    if groups:
        panels.append({"kind": "bars", "title": title, "unit": "tok/s", "groups": groups})
    lines = [("context-prompt_tps", "Prompt processing"), ("context-ttft_s", "Time to first token"),
             ("concurrency-aggregate_tps", "Throughput vs streams")]
    if title == "Generation, real prompts":
        # The headline has the stream counts; the sweep's one-stream generation goes in a line.
        lines.insert(0, ("context-decode_tps", "Generation by prompt length"))
    for sid, short in lines:
        sp = specs.get(sid)
        if not sp or not sp["lines"]:
            continue
        sp = dict(sp, title=short)
        if sid == "context-ttft_s":
            ys = [p[1] for ln in sp["lines"] for p in ln["points"] if p[1]]
            sp["ylog"] = bool(ys) and min(ys) > 0 and max(ys) / min(ys) > 50
        panels.append({"kind": "line", "title": short, "unit": sp["unit"], "spec": sp})
    return panels[:4]


def summary_text(series: list[Series], title: str | None) -> dict:
    first = series[0].data
    models = list(dict.fromkeys(s.data.get("model", "") for s in series))
    labels = [s.label for s in series]
    heading = title or f"{', '.join(models)} on {', '.join(labels[:-1]) + ' and ' + labels[-1] if len(labels) > 1 else labels[0]}"
    hw = list(dict.fromkeys(v for s in series for k, v in (s.data.get("meta") or {}).items()
                            if k.lower() in ("hardware", "gpu", "host_hardware")))
    dates = sorted({(s.data.get("started_at") or "")[:10] for s in series} - {""})
    sub = [" / ".join(hw)] if hw else []
    sub.append("measured through CapyCTL")
    if dates:
        sub.append(dates[-1] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}")
    if is_decisions(series):
        notes = decision_notes(first)
    else:
        notes = []
        cs = (first.get("context_sweep") or {}).get("settings")
        if cs:
            # The points' own labels: an adapted run's 512 ... 131072 reads 0.5k ... 128k.
            pts = sorted((first.get("context_sweep") or {}).get("points") or [],
                         key=lambda p: p["target_prompt_tokens"])
            t = cs["targets"]
            lo, hi = (pts[0]["label"], pts[-1]["label"]) if pts else (size_label(min(t)), size_label(max(t)))
            notes.append(f"prompts {lo}–{hi} tokens, max_tokens {cs['max_tokens']}, "
                         f"{cs['runs']} runs per point")
        cc = (first.get("concurrency") or {}).get("settings")
        if cc:
            lv = cc["levels"]
            streams = f"{lv[0]} stream{'s' if lv[0] > 1 else ''}" if len(set(lv)) == 1 else f"{min(lv)}–{max(lv)} streams"
            notes.append(f"{streams}, max_tokens {cc['max_tokens']}, {cc['rounds']} rounds")
        notes.append(f"temperature {fmt_plain((first.get('settings') or {}).get('temperature'))}, medians")
    sample = is_sample(series)
    if sample:
        notes.insert(0, "sample data from the test fake server")
    return {"title": heading, "subtitle": " · ".join(sub), "footnote": "; ".join(notes), "url": RECIPES_URL,
            "sample": sample}


def fmt_plain(v) -> str:
    return "–" if v is None else f"{v:g}"


def bar_note(series: list[Series], s: Series, v, base, points: bool = False) -> str:
    """Change vs the first series on N-series bars (%, or percentage points for a share);
    nothing for one series or the baseline."""
    if len(series) < 2 or s is series[0] or v is None or base is None:
        return ""
    if points:
        return f"{v - base:+.1f} pt"
    return pct_change(v, base) if base else ""


def panel_note(panel: dict, series: list[Series], s: Series, v, base) -> str:
    return bar_note(series, s, v, base, panel.get("delta") == "pt")


def panel_better(panel: dict) -> str:
    return f"{panel.get('better', 'higher')} is better"


# Advance widths in em of DejaVu Sans Bold, matplotlib's default font, for bar labels.
_EM = {**{d: 0.696 for d in "0123456789"}, "+": 0.838, "-": 0.415, "–": 0.5, "%": 1.002,
       ".": 0.38, ",": 0.38, " ": 0.348}


def text_width_em(text: str) -> float:
    return sum(_EM.get(ch, 0.72) for ch in text)


def fit_font(texts, size: float, slot: float, min_size: float = 6.0) -> float:
    """Largest font size up to `size` at which every text fits `slot` (same units as size)."""
    w = max((text_width_em(t) for t in texts if t), default=0)
    if not w:
        return size
    return max(min_size, min(size, slot / w))


def bar_labels(panel: dict, series: list[Series]) -> tuple[list[str], list[str]]:
    vals, notes = [], []
    for g in panel["groups"]:
        base = g["bars"][0][1]
        for s, v in g["bars"]:
            if v is not None:
                vals.append(fmt(v, panel["unit"]))
                notes.append(panel_note(panel, series, s, v, base))
    return vals, notes


def svg_bars(panel: dict, series: list[Series]) -> str:
    W, H = 460, 300
    ml, mr, mt, mb = 12, 12, 46, 40
    pw, ph = W - ml - mr, H - mt - mb
    vals = [v for g in panel["groups"] for _, v in g["bars"] if v is not None]
    if not vals:
        return ""
    vmax = max(vals) or 1
    n = len(series)
    gw = pw / len(panel["groups"])
    bw = min(64, gw * 0.8 / n)
    out = [f'<svg class="chart bars" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{esc(panel["title"])} ({esc(panel["unit"])})" xmlns="http://www.w3.org/2000/svg">']
    out.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{mt + ph}" y2="{mt + ph}"/>')
    texts, notes = bar_labels(panel, series)
    vsize = fit_font(texts, 20, bw * 0.86)
    psize = fit_font(notes, 14, bw * 0.86)
    for gi, g in enumerate(panel["groups"]):
        gx = ml + gi * gw + (gw - bw * n) / 2
        base = g["bars"][0][1]
        for k, (s, v) in enumerate(g["bars"]):
            if v is None:
                continue
            i = s.idx % len(PALETTE)
            h = v / vmax * ph
            x = gx + k * bw + bw * 0.08
            w = bw * 0.84
            y = mt + ph - h
            out.append(f'<rect class="bar s{i}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="3">'
                       f'<title>{esc(s.label)}: {fmt(v, panel["unit"])}'
                       f'{"" if panel["unit"] == "%" else " " + esc(panel["unit"])}</title></rect>')
            note = panel_note(panel, series, s, v, base)
            ty = y - (psize * 1.2 + 7 + 3 if note else 8)
            out.append(f'<text class="barval" x="{x + w / 2:.1f}" y="{ty:.1f}" text-anchor="middle" '
                       f'style="font-size:{vsize:.1f}px">{fmt(v, panel["unit"])}</text>')
            if note:
                out.append(f'<text class="barpct" x="{x + w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" '
                           f'style="font-size:{psize:.1f}px">{note}</text>')
        out.append(f'<text class="axlabel grp" x="{ml + gi * gw + gw / 2:.1f}" y="{mt + ph + 24}" '
                   f'text-anchor="middle">{esc(g["label"])}</text>')
    out.append("</svg>")
    return "".join(out)


SUMMARY_CSS = """
.s-head h1{font-size:34px;margin:28px 0 6px}
.s-head p{font-size:17px;color:var(--muted);margin:0}
.s-legend{font-size:16px;margin:18px 0 0}
.s-legend i{width:14px;height:14px;border-radius:3px}
.s-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}
.s-grid .card.hero{grid-row:span 1}
.s-grid .card.full{grid-column:1/-1}
.board td.b{background:color-mix(in srgb,var(--accent) var(--a),transparent);text-align:center;font-variant-numeric:tabular-nums}
.board td.hi{color:var(--bg);font-weight:600}
:root[data-theme="dark"] .board td.hi:not(.hi2){color:var(--text)}
.board th{text-align:center}
svg .lbl{fill:var(--text)}
.s-grid h3{font-size:18px}
.s-grid h3 small{font-size:14px}
svg .barval{fill:var(--text);font-weight:700}
svg .barpct{fill:var(--muted);font-weight:500}
svg .grp{font-size:14px;fill:var(--text)}
.s-foot{color:var(--muted);font-size:14px;margin:18px 0 0;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.s-foot b{color:var(--text);font-weight:500}
@media screen and (max-width:760px){.s-grid{grid-template-columns:minmax(0,1fr)}.s-head h1{font-size:26px}
.s-head p{font-size:15px}svg .grp{font-size:17px}}
"""


def summary_html(series: list[Series], panels: list[dict], text: dict, pngs: list[str]) -> str:
    legend = "".join(f'<span><i class="sw{s.idx % len(PALETTE)}"></i>{esc(s.label)}</span>' for s in series)
    cards = []
    for pnl in panels:
        full = ""
        if pnl["kind"] == "scoreboard":
            svg, full = board_html(pnl), " full"
        elif pnl["kind"] == "scatter":
            svg = svg_scatter(pnl)
        elif pnl["kind"] == "bars":
            svg = svg_bars(pnl, series)
        else:
            svg = svg_chart(pnl["spec"])
        if pnl.get("caption"):
            sub = pnl["caption"]
        elif pnl["kind"] == "bars":
            sub = f"{pnl['unit']}, {panel_better(pnl)}"
        else:
            sub = f"{pnl['unit']}, {pnl['spec']['better']} is better" + (", log scale" if pnl["spec"].get("ylog") else "")
        cards.append(f'<div class="card{full}"><h3>{esc(pnl["title"])} <small>{esc(sub)}</small></h3>{svg}</div>')
    links = [THEME_BUTTON] + [f'<a href="{esc(p)}">{esc(p)}</a>' for p in pngs] + ['<a href="../report.html">Full report</a>']
    return f"""{page_head(text["title"], page_css(SUMMARY_CSS))}
<body>
<div class="wrap">
<header class="top">{logo_tags()}<nav class="actions">{"".join(links)}</nav></header>
{sample_banner_html(series)}
<div class="s-head"><h1>{esc(text["title"])}</h1><p>{esc(text["subtitle"])}</p></div>
<div class="legend s-legend">{legend}</div>
<div class="s-grid">{"".join(cards)}</div>
<p class="s-foot"><span>{esc(text["footnote"])}</span><b>{esc(text["url"])}</b></p>
</div>
</body>
</html>
"""


def _summary_layout(kind: str, n: int) -> list[tuple]:
    """Gridspec slots (row, col span) per panel: [(rows, cols, [(r0, r1, c0, c1), ...])]."""
    if kind == "wide":
        if n == 4:
            return [(2, 2, [(0, 1, 0, 1), (0, 1, 1, 2), (1, 2, 0, 1), (1, 2, 1, 2)])]
        return [(1, n, [(0, 1, c, c + 1) for c in range(n)])]
    if n == 1:
        return [(1, 1, [(0, 1, 0, 1)])]
    if n == 2:
        return [(2, 1, [(0, 1, 0, 1), (1, 2, 0, 1)])]
    if n == 3:
        return [(2, 2, [(0, 1, 0, 2), (1, 2, 0, 1), (1, 2, 1, 2)])]
    return [(3, 2, [(0, 1, 0, 2), (1, 2, 0, 1), (1, 2, 1, 2), (2, 3, 0, 2)])]


def mpl_panel_style(ax, T: dict, kind: str) -> None:
    ax.set_facecolor(T["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(T["rule"])
    ax.spines["bottom"].set_color(T["rule"])
    ax.tick_params(labelsize=11 if kind == "wide" else 12.5, length=0, pad=6)


def mpl_panel_title(ax, title: str, sub: str, T: dict, big: float) -> None:
    ax.set_title(title, loc="left", fontsize=big, fontweight="bold", color=T["text"], pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=big - 5, color=T["muted"], va="bottom")


def mpl_line_panel(ax, sp: dict, T: dict, kind: str) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter, NullLocator
    pal = T["palette"]
    for ln in sp["lines"]:
        i = ln["series"].idx % len(pal)
        pts = sorted(ln["points"])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=pal[i], linewidth=2.6,
                marker=MPL_MARKERS[MARKERS[i]], markersize=5.5, zorder=3)
    mpl_axes_x(ax, sp)
    ys = [p[1] for ln in sp["lines"] for p in ln["points"] if p[1]]
    if sp.get("ylog") and ys:
        ax.set_yscale("log")
        lo, hi = min(ys) / 1.3, max(ys) * 1.3
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator(log_ticks_125(lo, hi)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_ylim(0, y_top(ys))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
    ax.grid(True, axis="y", color=T["rule"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(sp["xlabel"], fontsize=11 if kind == "wide" else 12.5, color=T["muted"])


def mpl_scoreboard(ax, pnl: dict, T: dict, kind: str, w: float, h: float, text_px) -> None:
    """The scoreboard in an axes of w x h (1x pixels): a row per series, a shaded cell per column."""
    from matplotlib.patches import Rectangle
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    rows, cols = pnl["rows"], pnl["cols"]
    base = 11.5 if kind == "wide" else 13

    def widest(texts, size, weight="bold"):
        return max((text_px(ln, size, weight) for t in texts if t for ln in t.split("\n")), default=0)

    # Labels take at most a fifth of the width: "Model · engine" breaks at the dot first, then shrinks.
    labels = [s.label for s in rows]
    if widest(labels, base, "normal") + 30 > 0.22 * w:
        labels = [lab.replace(" · ", "\n", 1) for lab in labels]
    lsize = base
    while lsize > 7 and widest(labels, lsize, "normal") + 30 > 0.22 * w:
        lsize -= 0.5
    two = any("\n" in lab for lab in labels)
    lw = widest(labels, lsize, "normal") + 30
    cw = (w - lw) / len(cols)
    hs = base - 1
    while hs > 6.5 and widest([t for c in cols for t in c["head"]], hs) > cw - 6:
        hs -= 0.5
    line = hs / 0.72 * 1.25
    nlines = max(len(c["head"]) for c in cols)
    hh = nlines * line + 8
    cap = (38 if two else 30) if kind == "wide" else (46 if two else 36)
    rh = min(cap, (h - hh) / len(rows))
    cs = base
    while cs > 6.5 and widest([t for c in cols for t in c["texts"]], cs) > cw - 10:
        cs -= 0.5
    for j, c in enumerate(cols):
        x = lw + (j + 0.5) * cw
        for k, t in enumerate(c["head"]):
            ax.text(x, hh - 6 - (len(c["head"]) - 1 - k) * line, t, ha="center", va="bottom", fontsize=hs,
                    color=T["text"] if k == 0 else T["muted"], fontweight="bold" if k == 0 else "normal")
    for r, (s, lab) in enumerate(zip(rows, labels)):
        y = hh + r * rh
        ax.add_patch(Rectangle((0, y + rh / 2 - 6), 12, 12, color=T["palette"][s.idx % len(T["palette"])]))
        ax.text(20, y + rh / 2, lab, va="center", fontsize=lsize, color=T["text"], linespacing=1.1)
        for j, c in enumerate(cols):
            t, sh = c["texts"][r], c["shades"][r]
            x = lw + j * cw
            if t is None:
                ax.text(x + cw / 2, y + rh / 2, "–", ha="center", va="center", fontsize=cs, color=T["muted"])
                continue
            a = shade_alpha(sh)
            ax.add_patch(Rectangle((x + 1.5, y + 1.5), cw - 3, rh - 3, facecolor=T["accent"], alpha=a,
                                   edgecolor="none"))
            ax.text(x + cw / 2, y + rh / 2, t, ha="center", va="center", fontsize=cs, fontweight="bold",
                    # dark text on a light orange needs more orange than light text on a dark one
                    color=T["bg"] if a > (0.72 if T["bg"] == THEMES["dark"]["bg"] else 0.55) else T["text"])


def mpl_scatter(fig, ax, pnl: dict, T: dict, kind: str) -> None:
    """Quality against speed: a labelled point per series, labels placed clear of each other."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    pal = T["palette"]
    x0, x1, y0, y1 = scatter_domain(pnl)
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.xaxis.set_major_locator(FixedLocator(log_ticks_125(x0, x1)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, color=T["rule"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(pnl["xlabel"], fontsize=11 if kind == "wide" else 12.5, color=T["muted"])
    fsize = 9.5 if kind == "wide" else 12.5
    renderer = fig.canvas.get_renderer()
    texts, sizes = [], []
    for s, x, y in pnl["points"]:
        i = s.idx % len(pal)
        ax.scatter([x], [y], s=70 if kind == "wide" else 95, color=pal[i], marker=MPL_MARKERS[MARKERS[i]],
                   edgecolors=T["bg"], linewidths=1, zorder=4)
        t = ax.text(x, y, s.label, fontsize=fsize, color=T["text"], ha="left", va="bottom", zorder=5)
        bb = t.get_window_extent(renderer=renderer)
        texts.append(t)
        sizes.append((bb.width, bb.height))
    anchors = [tuple(ax.transData.transform((x, y))) for _, x, y in pnl["points"]]
    boxes = place_labels(anchors, sizes, tuple(ax.bbox.extents), marker=14)
    inv = ax.transData.inverted()
    for t, (ax_, ay), b in zip(texts, anchors, boxes):
        t.set_position(inv.transform((b[0], b[1])))
        cx, cy = min(max(ax_, b[0]), b[2]), min(max(ay, b[1]), b[3])
        if math.hypot(cx - ax_, cy - ay) > 26:
            (px, py), (qx, qy) = inv.transform((ax_, ay)), inv.transform((cx, cy))
            ax.plot([px, qx], [py, qy], color=T["muted"], linewidth=0.8, zorder=3)


def field_summary_png(fig, panels: list[dict], T: dict, kind: str, box: tuple, text_px, big: float) -> None:
    """The scoreboard layout in box (x0, y0, x1, y1), 1x pixels from the top left. Wide: the scoreboard
    on the left, quality vs speed over latency on the right. Tall: the three stacked."""
    wpx, hpx = SUMMARY_SIZES[kind]
    x0, y0, x1, y1 = box
    tb, xb, yl = 50, 44, 44  # title block above an axes, tick labels and label below, tick labels left

    def axes(l, t, r, b):
        return fig.add_axes((l / wpx, 1 - b / hpx, (r - l) / wpx, (b - t) / hpx))

    board = next(p for p in panels if p["kind"] == "scoreboard")
    charts = [p for p in panels if p["kind"] != "scoreboard"]
    n = len(board["rows"])
    if kind == "wide":
        bx1 = x0 + (x1 - x0) * 0.64
        brect = (x0, y0 + tb, bx1, y1)
        shares = [0.58, 0.42][:len(charts)] if len(charts) == 2 else [1.0] * len(charts)
        slots, t = [], y0
        for f in shares:
            slots.append((bx1 + 36 + yl, t + tb, x1, t + f * (y1 - y0) - xb - 10))
            t += f * (y1 - y0)
    else:
        bh = tb + 76 + n * 42
        brect = (x0, y0 + tb, x1, y0 + bh)
        rest = y1 - (y0 + bh + 30)
        shares = [0.56, 0.44][:len(charts)] if len(charts) == 2 else [1.0] * len(charts)
        slots, t = [], y0 + bh + 30
        for f in shares:
            slots.append((x0 + yl, t + tb, x1, t + f * rest - xb - 10))
            t += f * rest
    ax = axes(*brect)
    mpl_scoreboard(ax, board, T, kind, brect[2] - brect[0], brect[3] - brect[1], text_px)
    mpl_panel_title(ax, board["title"], board["caption"], T, big)
    for pnl, r in zip(charts, slots):
        ax = axes(*r)
        mpl_panel_style(ax, T, kind)
        if pnl["kind"] == "scatter":
            mpl_scatter(fig, ax, pnl, T, kind)
        else:
            mpl_line_panel(ax, pnl["spec"], T, kind)
        mpl_panel_title(ax, pnl["title"], pnl["caption"], T, big)


def summary_png(plt, series: list[Series], panels: list[dict], text: dict, path: Path,
                kind: str, theme: str) -> None:
    import textwrap
    T = mpl_theme(plt, theme)
    pal = T["palette"]
    wpx, hpx = SUMMARY_SIZES[kind]
    fig = plt.figure(figsize=(wpx / 100, hpx / 100), dpi=200)
    fig.patch.set_facecolor(T["bg"])

    def fx(x):
        return x / wpx

    def fy(y):  # y measured from the top, in 1x pixels
        return 1 - y / hpx

    pad = 44
    y = 34
    if text.get("sample"):
        from matplotlib.patches import Rectangle as _Rect
        band = 40
        fig.patches.append(_Rect((0, fy(band)), 1, band / hpx, color="#b91c1c",
                                 transform=fig.transFigure, figure=fig))
        fig.text(0.5, fy(band / 2), SAMPLE_BANNER, ha="center", va="center", fontsize=16,
                 fontweight="bold", color="#ffffff")
        y += band
    logo = T["logo"]
    if logo.exists():
        try:
            from PIL import Image
            img = Image.open(logo)
            lh = 34
            lw = lh * img.width / img.height
            ax = fig.add_axes((fx(pad), fy(y + lh), fx(lw), lh / hpx))
            ax.imshow(img)
            ax.axis("off")
        except Exception:  # Pillow missing or no WebP support: text mark instead
            fig.text(fx(pad), fy(y), "CapyCTL", fontsize=18, fontweight="bold", color=T["accent"], va="top")
    else:
        fig.text(fx(pad), fy(y), "CapyCTL", fontsize=18, fontweight="bold", color=T["accent"], va="top")
    y += 34 + 22
    # The title wraps by its rendered width, not a character count: a bold title of capitals
    # and digits ran off the right edge at a 62-character wrap. Past two lines it shrinks.
    tsize = 25 if kind == "wide" else 29
    renderer = fig.canvas.get_renderer()

    def text_px(s: str, size: float, weight: str = "bold") -> float:
        t = fig.text(0, 0, s, fontsize=size, fontweight=weight)
        w = t.get_window_extent(renderer=renderer).width / 2  # dpi 200 = 2x
        t.remove()
        return w

    def wrap_px(s: str, size: float) -> list[str]:
        out, cur = [], ""
        for word in s.split():
            cand = f"{cur} {word}".strip()
            if cur and text_px(cand, size) > wpx - 2 * pad:
                out.append(cur)
                cur = word
            else:
                cur = cand
        return out + [cur] if cur else out or [""]

    lines = wrap_px(text["title"], tsize)
    while len(lines) > 2 and tsize > 16:
        tsize -= 1
        lines = wrap_px(text["title"], tsize)
    for ln in lines:
        fig.text(fx(pad), fy(y), ln, fontsize=tsize, fontweight="bold", color=T["text"], va="top")
        y += tsize * 1.45
    y += 4
    fig.text(fx(pad), fy(y), text["subtitle"], fontsize=13.5 if kind == "wide" else 15, color=T["muted"], va="top")
    y += 30 if kind == "wide" else 34
    # A field of models has no legend: the scoreboard's rows carry the colors.
    many = bool(panels) and panels[0]["kind"] == "scoreboard"
    from matplotlib.patches import Rectangle
    lx = pad
    lsize = 13 if kind == "wide" else 15
    for s in ([] if many else series):
        i = s.idx % len(pal)
        t = fig.text(fx(lx + 22), fy(y + 1), s.label, fontsize=lsize, color=T["text"], va="top")
        width_px = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / 2  # dpi 200 = 2x
        # An entry that would run past the right margin starts the next row instead.
        if lx > pad and lx + 22 + width_px > wpx - pad:
            lx = pad
            y += 26
            t.set_position((fx(lx + 22), fy(y + 1)))
        fig.patches.append(Rectangle((fx(lx), fy(y + 15)), fx(14), 14 / hpx, color=pal[i],
                                     transform=fig.transFigure, figure=fig))
        lx += 22 + width_px + 28
    y += 14 if many else 58
    fsize = 10.5 if kind == "wide" else 12
    foot = textwrap.wrap(text["footnote"], 100 if kind == "wide" else 78)
    # room for every line of the footnote (and, when tall, the URL under it)
    foot_h = (max(58, 10 + len(foot) * fsize * 1.5) if kind == "wide"
              else max(100, 30 + (len(foot) + 1) * fsize * 1.5))
    big = 15 if kind == "wide" else 17
    if many:
        field_summary_png(fig, panels, T, kind, (pad, y, wpx - pad, hpx - foot_h - 12), text_px, big)
    grid = [] if many else panels
    top, bottom = fy(y), (foot_h + 12) / hpx
    (rows, cols, slots), = _summary_layout(kind, max(1, len(grid)))
    grid_top, grid_bottom = top - 30 / hpx, bottom + (34 if kind == "wide" else 50) / hpx
    # The wide layout's gap between panel rows is fixed in pixels (axis label, panel title and
    # unit line), so a two-line title or legend shrinks the panels, never the gap.
    hspace = 0.62 if kind == "tall" else 1.15
    if kind == "wide" and rows > 1:
        gap = 112
        axes_h = ((grid_top - grid_bottom) * hpx - gap * (rows - 1)) / rows
        hspace = gap / max(axes_h, 1)
    gs = fig.add_gridspec(rows, cols, left=fx(pad + 8), right=1 - fx(pad), top=grid_top,
                          bottom=grid_bottom, hspace=hspace, wspace=0.22)
    for pnl, (r0, r1, c0, c1) in zip(grid, slots):
        ax = fig.add_subplot(gs[r0:r1, c0:c1])
        mpl_panel_style(ax, T, kind)
        if pnl["kind"] == "bars":
            groups = pnl["groups"]
            n = len(series)
            bw = min(0.8 / n, 0.42)
            vmax = max((v for g in groups for _, v in g["bars"] if v is not None), default=1) or 1
            ax.set_xlim(-0.5, len(groups) - 0.5)
            # Each label gets its bar's slot (bw in data units, in points here) and shrinks to fit,
            # so neighbouring labels never touch; the change sits under the value.
            axes_w = ax.get_position().width * fig.get_figwidth() * 72
            slot = bw * axes_w / len(groups) * 0.82
            texts, notes = bar_labels(pnl, series)
            vsize = fit_font(texts, big + 3 if n <= 3 else big - 1, slot)
            psize = fit_font(notes, vsize - 6, slot)
            for gi, g in enumerate(groups):
                base = g["bars"][0][1]
                for k, (s, v) in enumerate(g["bars"]):
                    if v is None:
                        continue
                    x = gi + (k - (n - 1) / 2) * bw
                    ax.bar(x, v, width=bw * 0.86, color=pal[s.idx % len(pal)], zorder=2)
                    note = panel_note(pnl, series, s, v, base)
                    off = 4
                    if note:
                        ax.annotate(note, (x, v), xytext=(0, off), textcoords="offset points", ha="center",
                                    va="bottom", fontsize=psize, color=T["muted"], fontweight="bold")
                        off += psize * 1.3
                    ax.annotate(fmt(v, pnl["unit"]), (x, v), xytext=(0, off), textcoords="offset points", ha="center",
                                va="bottom", fontsize=vsize, fontweight="bold", color=T["text"])
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g["label"] for g in groups], fontsize=big - 2, color=T["text"])
            label_pts = 4 + vsize * 1.25 + (psize * 1.3 if n > 1 else 0)
            axes_pts = ax.get_position().height * fig.get_figheight() * 72
            ax.set_ylim(0, bar_top(vmax, label_pts, axes_pts))
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            sub = f"{pnl['unit']}, {panel_better(pnl)}"
        else:
            mpl_line_panel(ax, pnl["spec"], T, kind)
            sub = f"{pnl['unit']}, {pnl['spec']['better']} is better"
        mpl_panel_title(ax, pnl["title"], sub, T, big)
    fy0 = hpx - foot_h + 4
    for k, ln in enumerate(foot):
        fig.text(fx(pad), fy(fy0 + k * fsize * 1.5), ln, fontsize=fsize, color=T["muted"], va="top")
    if kind == "wide":
        fig.text(1 - fx(pad), fy(fy0), text["url"], fontsize=fsize, color=T["accent_text"], va="top",
                 ha="right", fontweight="bold")
    else:
        fig.text(fx(pad), fy(fy0 + len(foot) * fsize * 1.5 + 4), text["url"], fontsize=fsize,
                 color=T["accent_text"], va="top", fontweight="bold")
    fig.savefig(path, facecolor=T["bg"])
    plt.close(fig)


def write_summary(series: list[Series], out: Path, title: str | None, png: bool = True) -> list[Path]:
    d = out / "summary"
    d.mkdir(parents=True, exist_ok=True)
    panels = summary_panels(series)
    if not panels:
        log("summary skipped: the results have neither a context sweep nor a concurrency sweep")
        return []
    text = summary_text(series, title)
    written = []
    plt = load_matplotlib("summary/*.png") if png else None
    if plt is not None:
        for kind in ("wide", "tall"):
            for theme in ("light", "dark"):
                p = d / f"summary-{kind}{'' if theme == 'light' else '-dark'}.png"
                summary_png(plt, series, panels, text, p, kind, theme)
                written.append(p)
    names = [p.name for p in written if "-dark" not in p.name]
    (d / "summary.html").write_text(summary_html(series, panels, text, names))
    return [d / "summary.html"] + written


def default_title(series: list[Series]) -> str:
    labels = [s.label for s in series]
    if len(labels) == 1:
        return labels[0]
    return " vs ".join(labels)


def cmd_report(args) -> int:
    series = load_series(args.results, parse_relabel(args.relabel))
    title = args.title or default_title(series)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = chart_specs(series)
    pngs = [] if args.no_png else write_pngs(specs, out, title, args.theme, is_sample(series))
    shared = write_summary(series, out, args.title, png=not args.no_png) if args.summary else []
    write_csv(series, out / "data.csv")
    (out / "summary.md").write_text(summary_md(series, title))
    (out / "report.html").write_text(report_html(series, specs, title, has_zip=not args.no_zip,
                                                 has_summary=bool(shared)))
    if not args.no_zip:
        with zipfile.ZipFile(out / "report.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("report.html", "summary.md", "data.csv"):
                z.write(out / name, name)
            for p in pngs:
                z.write(p, f"charts/{p.name}")
            for p in shared:
                z.write(p, f"summary/{p.name}")
            for s in series:
                z.write(s.path, f"results/{s.path.name}")
    log(f"wrote {out / 'report.html'}, summary.md, data.csv"
        + (f", {len(pngs)} charts" if pngs else "")
        + (f", summary/ ({len(shared)} files)" if shared else "") + ("" if args.no_zip else ", report.zip"))
    return 0


# ---------------------------------------------------------------------------
# report: decisions
# ---------------------------------------------------------------------------

EMPTY = {"median": None, "min": None, "max": None, "n": 0}

DEC_LAT_TABLE = [
    ("input_tokens", "Input tokens", "tok"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
    ("per_question_p50_ms", "Per question p50", "ms"),
    ("latency_ms", "Latency", "ms"),
    ("memory_peak_gib", "Peak memory", "GiB"),
]
DEC_CACHE_TABLE = [
    ("input_tokens", "Input tokens", "tok"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
    ("latency_ms", "Latency", "ms"),
]
DEC_CC_TABLE = [
    ("requests_per_s", "Requests", "per s"),
    ("questions_per_s", "Questions", "per s"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
    ("error_rate", "Errors", "%"),
    ("memory_peak_gib", "Peak memory", "GiB"),
]
# Per set kind: (summary key, column title, unit). Results of schema 2 have classic sets only.
ACC_TABLES = {
    "classic": [("accuracy", "Accuracy", "%"), ("macro_f1", "Macro-F1", "%"), ("brier_vector", "Brier, vector", ""),
                ("brier", "Brier, binary", ""), ("nll", "NLL", ""), ("ece", "ECE", ""),
                ("latency_ms", "Latency", "ms"), ("memory_peak_gib", "Peak memory", "GiB")],
    "typed": [("accuracy", "Accuracy", "%"), ("accuracy_gold_ties", "Accuracy, gold ties", "%"),
              ("kl", "KL from gold", ""), ("brier", "Brier, soft gold", ""), ("brier_hard", "Brier, hard label", ""),
              ("ece", "ECE", ""), ("score_mae", "Score MAE", "levels"), ("coverage", "Coverage", "%"),
              ("latency_ms", "Latency", "ms"), ("memory_peak_gib", "Peak memory", "GiB")],
    "di": [("score", "Score", "%"), ("coverage", "Coverage", "%"), ("raw", "Raw", "%"), ("skill", "Skill", "%"),
           ("latency_ms", "Latency", "ms"), ("memory_peak_gib", "Peak memory", "GiB")],
}
PCT_KEYS = {"accuracy", "accuracy_gold_ties", "macro_f1", "coverage", "score", "raw", "skill", "field_accuracy",
            "case_exact_accuracy", "error_rate"}
LOWER_BETTER = {"kl", "brier", "brier_vector", "brier_hard", "nll", "ece", "ece_gold_ties", "score_mae"}
# Typed Decisions rows of the comparison: (label, key, unit); acc_<type> is the accuracy of one question type.
TYPED_ROWS = [("Accuracy", "accuracy", "%"), ("Accuracy, gold ties", "accuracy_gold_ties", "%"),
              ("Accuracy, noul", "acc_noul", "%"), ("Accuracy, choice", "acc_choice", "%"),
              ("Accuracy, score", "acc_score", "%"), ("KL from gold", "kl", ""), ("Brier, soft gold", "brier", ""),
              ("Brier, hard label", "brier_hard", ""), ("ECE", "ece", ""), ("ECE, gold ties", "ece_gold_ties", ""),
              ("Score MAE", "score_mae", "levels"), ("Coverage", "coverage", "%")]
CLASSIC_ROWS = [("Accuracy", "accuracy", "%"), ("Macro-F1", "macro_f1", "%"), ("Brier, vector", "brier_vector", ""),
                ("Brier, binary", "brier", ""), ("NLL", "nll", ""), ("ECE", "ece", "")]


def one_value(v) -> dict:
    return {"median": v, "min": v, "max": v, "n": 1 if v is not None else 0}


def dec_value(summary: dict, key: str) -> dict:
    """A value of a decisions summary; latencies in ms for display (the results hold seconds)."""
    if key in ("latency_p50_ms", "latency_p95_ms", "per_question_p50_ms", "per_question_p95_ms"):
        return one_value(ms(summary.get(key[:-3] + "_s")))
    if key == "latency_ms":
        v = summary.get("latency_s") or EMPTY
        return {"median": ms(v["median"]), "min": ms(v["min"]), "max": ms(v["max"]), "n": v["n"]}
    if key == "requests_per_s":
        v = summary.get("requests_per_s")
        if v is None and isinstance(summary.get("decisions_per_s"), dict):
            v = summary["decisions_per_s"]["median"]  # schema 2: the median over burst rounds
        return one_value(v)
    if key == "questions_per_s":
        return one_value(summary.get(key))
    if key == "error_rate":
        v = summary.get(key)
        return one_value(v * 100 if v is not None else None)
    v = summary.get(key)
    return v if isinstance(v, dict) else EMPTY


def set_kind(st: dict) -> str:
    return st.get("kind") or "classic"


def headline_key(st: dict) -> str:
    return "score" if set_kind(st) == "di" else "accuracy"


def acc_value(st: dict, key: str) -> dict:
    sm = st["summary"]
    if key == "latency_ms":
        return dec_value(sm, key)
    if key.startswith("acc_"):
        v = ((sm.get("by_type") or {}).get(key[4:]) or {}).get("accuracy")
        return one_value(v * 100 if v is not None else None)
    v = sm.get(key)
    if isinstance(v, dict):
        return v
    return one_value(v * 100 if v is not None and key in PCT_KEYS else v)


def lat_key(p: dict) -> tuple:
    return (p["questions"], p["target_state_tokens"])


def lat_label(p: dict) -> str:
    q = p["questions"]
    return f"{q} question{'s' if q > 1 else ''}, {p['state_label']}-token state"


def lat_get(s: Series, k: tuple, key: str):
    return next((dec_value(p["summary"], key)["median"] for p in s.lat_points if lat_key(p) == k), None)


def level_get(s: Series, c: int, key: str):
    return next((dec_value(lv["summary"], key)["median"] for lv in s.dec_levels if lv["concurrency"] == c), None)


def cache_get(s: Series, part: str, key: str):
    sm = (s.cache.get(part) or {}).get("summary")
    return dec_value(sm, key)["median"] if sm else None


def cache_ratio(s: Series):
    cold, rep = cache_get(s, "cold", "latency_p50_ms"), cache_get(s, "repeated", "latency_p50_ms")
    return rep / cold if cold and rep is not None else None


def acc_sets_named(series: list[Series]) -> list[dict]:
    """One set object per set name, from the first series that ran it, in run order."""
    found: dict[str, dict] = {}
    for s in series:
        for st in s.acc_sets:
            found.setdefault(st["name"], st)
    return list(found.values())


def acc_get(s: Series, name: str, key: str):
    return next((acc_value(st, key)["median"] for st in s.acc_sets if st["name"] == name), None)


def reliability_specs(series: list[Series]) -> list[dict]:
    """Reliability diagrams: per bin, the accuracy against the mean confidence, both in %."""
    specs = []
    for st in acc_sets_named(series):
        if set_kind(st) == "di":
            continue
        lines = []
        for s in series:
            sm = next((x["summary"] for x in s.acc_sets if x["name"] == st["name"]), {})
            pts = [(b["confidence"] * 100, b["accuracy"] * 100, f"confidence {b['lo']:.1f}–{b['hi']:.1f}, n = {b['n']}")
                   for b in sm.get("ece_bins") or [] if b["n"]]
            if pts:
                lines.append({"series": s, "name": s.label, "dash": False, "points": pts})
        if lines:
            specs.append({"id": f"reliability-{st['name']}", "section": "calibration", "metric": "reliability",
                          "title": f"Reliability, {st['title']}", "unit": "%", "better": "closer to the diagonal",
                          "xlog": False, "xlabel": "Confidence (%)", "xdomain": (0, 100),
                          "xticks": [0, 20, 40, 60, 80, 100], "ydomain": (0, 100), "diagonal": True, "lines": lines})
    return specs


def decision_chart_specs(series: list[Series]) -> list[dict]:
    specs = []
    pairs = (("latency_p50_ms", " p50", False), ("latency_p95_ms", " p95", True))
    for qn in sorted({p["questions"] for s in series for p in s.lat_points}):
        lines = []
        for s in series:
            for key, suffix, dash in pairs:
                pts = [(p["target_state_tokens"], dec_value(p["summary"], key)["median"], p["state_label"])
                       for p in s.lat_points if p["questions"] == qn]
                pts = [pt for pt in pts if pt[1] is not None]
                if pts:
                    lines.append({"series": s, "name": s.label + suffix, "dash": dash, "points": pts})
        specs.append({"id": f"latency-{qn}q", "section": "latency", "metric": "latency",
                      "title": f"Latency, {qn} question{'s' if qn > 1 else ''}", "unit": "ms",
                      "better": "lower", "xlog": True, "xlabel": "State tokens", "lines": lines})
    if len({lv["concurrency"] for s in series for lv in s.dec_levels}) > 1:
        for sid, title, unit, better, sub in (
                ("requests_per_s", "Requests per second", "requests/s", "higher", (("requests_per_s", "", False),)),
                ("questions_per_s", "Answered questions per second", "questions/s", "higher",
                 (("questions_per_s", "", False),)),
                ("latency", "Latency, p50 and p95", "ms", "lower", pairs)):
            lines = []
            for s in series:
                for key, suffix, dash in sub:
                    pts = [(lv["concurrency"], dec_value(lv["summary"], key)["median"], str(lv["concurrency"]))
                           for lv in s.dec_levels]
                    pts = [pt for pt in pts if pt[1] is not None]
                    if pts:
                        lines.append({"series": s, "name": s.label + suffix, "dash": dash, "points": pts})
            if lines:
                specs.append({"id": f"concurrency-{sid}", "section": "concurrency", "metric": sid, "title": title,
                              "unit": unit, "better": better, "xlog": False, "xlabel": "Concurrent streams",
                              "lines": lines})
    return specs + reliability_specs(series)


def decision_csv_rows(series: list[Series]):
    def row(s, section, point, x, key, unit, v):
        return [s.label, section, point, x, key, unit, v["median"], v["min"], v["max"], v["n"]]
    for s in series:
        for p in s.lat_points:
            for key, _, unit in DEC_LAT_TABLE:
                yield row(s, "latency", p["label"], p["target_state_tokens"], key, unit, dec_value(p["summary"], key))
        for part in ("cold", "repeated"):
            sm = (s.cache.get(part) or {}).get("summary")
            if sm:
                for key, _, unit in DEC_CACHE_TABLE:
                    yield row(s, "cache", part, s.cache["settings"]["state_tokens"], key, unit, dec_value(sm, key))
        for lv in s.dec_levels:
            for key, _, unit in DEC_CC_TABLE:
                yield row(s, "concurrency", str(lv["concurrency"]), lv["concurrency"], key, unit,
                          dec_value(lv["summary"], key))
        for st in s.acc_sets:
            n = st["summary"].get("answered")
            for key, _, unit in ACC_TABLES[set_kind(st)]:
                yield row(s, "accuracy", st["name"], n, key, unit, acc_value(st, key))
            for b in st["summary"].get("ece_bins") or []:
                for key in ("n", "confidence", "accuracy"):
                    yield row(s, "reliability", f"{st['name']} {b['lo']:.1f}-{b['hi']:.1f}", b["lo"], key, "",
                              one_value(b[key]))


def md_compare(series: list[Series], keys, get, unit: str = "", points: bool = False) -> list[list[str]]:
    """Rows [label, value per series]; every series after the first carries its change against it,
    unless the series are a field of models (`many_series`), where a change against one means nothing."""
    base = series[0]
    deltas = not many_series(series)
    rows = []
    for label, k in keys:
        base_v = get(base, k)
        row = [label]
        for s in series:
            v = get(s, k)
            c = fmt(v, unit)
            if deltas and s is not base and v is not None and base_v is not None:
                if points:
                    c += f" ({v - base_v:+.1f} pt)"
                elif base_v:
                    c += f" ({pct_change(v, base_v)})"
            row.append(c)
        rows.append(row)
    return rows


def md_metric_rows(series: list[Series], metrics, get) -> list[list[str]]:
    """Rows of different metrics: [(label, key, unit)]; % rows change in points, the rest in %."""
    rows = []
    for label, key, unit in metrics:
        if all(get(s, key) is None for s in series):
            continue
        better = "lower" if key in LOWER_BETTER else "higher"
        name = f"{label} ({unit}, {better} is better)" if unit else f"{label} ({better} is better)"
        rows += md_compare(series, [(name, key)], get, unit, points=unit == "%")
    return rows


def pinned_set_line(st: dict) -> str:
    ds = st["dataset"]
    kind = set_kind(st)
    if kind == "di":
        src = ds["source"]
        where = (f"`{src['repo']}` at `{src['commit']}`" if "repo" in src else
                 f"{src['url']} (sha256 `{src['sha256'][:16]}…`)" if "url" in src else src.get("generator", ""))
        return (f"{st['title']}: Decision Index catalog {ds['catalog_id']} ({ds['track']}), built from {where} as "
                f"the kit `{ds['kit']['repo']}` at `{ds['kit']['commit'][:12]}` builds it (rows sha256 "
                f"`{ds['sha256'][:16]}…`), {st['rows']['n']} of {ds['kept']} scored rows; licence: {ds['license']}")
    rows = (f"{st['sample']['n']} of {ds['rows']} rows, seed {st['sample']['seed']}" if kind == "classic"
            else f"{st['rows']['n']} of {ds['rows']} cases")
    lic = f"; licence: {ds['license']}" if ds.get("license") else ""
    return (f"{st['title']}: `{ds['repo']}` {ds['split']} at revision `{ds['revision']}` "
            f"(rows sha256 `{ds['sha256'][:16]}…`), {rows}{lic}")


def reliability_md(series: list[Series], st: dict) -> list[str]:
    bins = next((x["summary"].get("ece_bins") for s in series for x in s.acc_sets
                 if x["name"] == st["name"] and x["summary"].get("ece_bins")), None)
    if not bins:
        return []

    def get(s, k):
        b = next(((x["summary"].get("ece_bins") or [None] * len(bins))[k] for x in s.acc_sets
                  if x["name"] == st["name"]), None)
        return b

    rows = []
    for k, b in enumerate(bins):
        cells = []
        for s in series:
            v = get(s, k)
            cells.append("–" if not v or not v["n"] else f"{fmt(v['accuracy'] * 100, '%')} at {fmt(v['confidence'] * 100, '%')} (n {v['n']})")
        rows.append([f"{b['lo']:.1f}–{b['hi']:.1f}"] + cells)
    return ["", f"#### Reliability, {st['title']}", "",
            "Per confidence bin: accuracy at the bin's mean confidence, and the bin's count.", "",
            md_table(["Confidence"] + [s.label for s in series], rows)]


def decisions_summary_md(series: list[Series], title: str) -> str:
    labels = [s.label for s in series]
    how = ("Latency cells are the p50 or p95 of the runs at a point; percentages compare "
           f"with {series[0].label}, differences of shares are percentage points.")
    if many_series(series):
        how = ("Latency cells are the p50 or p95 of the runs at a point. The series are different models, so "
               "the tables give values only, no change against the first.")
    out = md_head(series, title, how)
    pts = sorted({lat_key(p): lat_label(p) for s in series for p in s.lat_points}.items())
    if pts:
        out += ["", "## Latency", "",
                "One request at a time, client side: request sent to the whole answer read. Per question is the "
                "request's latency over the questions it answered (amortized, not the latency of one question)."]
        for key, name, unit in (("latency_p50_ms", "Latency p50", "ms"), ("latency_p95_ms", "Latency p95", "ms"),
                                ("per_question_p50_ms", "Per answered question, p50", "ms"),
                                ("input_tokens", "Input tokens", "tok"), ("memory_peak_gib", "Peak memory", "GiB")):
            def get(s, k, key=key):
                return lat_get(s, k, key)
            if all(get(s, k) is None for s in series for k, _ in pts):
                continue
            better = "" if key == "input_tokens" else ", lower is better"
            out += ["", f"### {name} ({unit}{better})", "",
                    md_table(["Point"] + labels, md_compare(series, [(lab, k) for k, lab in pts], get))]
    cached = [s for s in series if s.cache]
    if cached:
        cs = cached[0].cache["settings"]
        out += ["", "## Cache", "",
                f"{cs['runs']} requests with a new state each (cold prefix), then {cs['runs']} repeats of one state "
                f"after its first sight; {cs['questions']} questions, {size_label(cs['state_tokens'])}-token state.", ""]
        rows = md_compare(series, [("Cold p50", ("cold", "latency_p50_ms")), ("Cold p95", ("cold", "latency_p95_ms")),
                                   ("Repeated p50", ("repeated", "latency_p50_ms")),
                                   ("Repeated p95", ("repeated", "latency_p95_ms"))],
                          lambda s, k: cache_get(s, *k))
        rows += md_compare(series, [("Repeated p50 / cold p50", None)], lambda s, k: cache_ratio(s))
        out += [md_table(["ms, lower is better"] + labels, rows)]
    levels = sorted({lv["concurrency"] for s in series for lv in s.dec_levels})
    if levels:
        cc = (series[0].decisions.get("concurrency") or {}).get("settings") or {}
        how = ("Closed loop: N streams, each sending its next request when its last one returns, "
               f"{cc.get('requests')} requests per level" if cc.get("requests") else "N requests started together")
        out += ["", "## Throughput", "",
                f"{how}; {cc.get('questions')} questions and a {size_label(cc.get('state_tokens') or 0)}-token "
                "state each, every request its own state."]
        for key, name, unit, better in (("requests_per_s", "Requests per second", "requests/s", "higher"),
                                        ("questions_per_s", "Answered questions per second", "questions/s", "higher"),
                                        ("latency_p50_ms", "Latency p50", "ms", "lower"),
                                        ("latency_p95_ms", "Latency p95", "ms", "lower"),
                                        ("error_rate", "Error rate", "%", "lower"),
                                        ("memory_peak_gib", "Peak memory", "GiB", "lower")):
            def get(s, c, key=key):
                return level_get(s, c, key)
            if all(get(s, c) is None for s in series for c in levels):
                continue
            out += ["", f"### {name} ({unit}, {better} is better)", "",
                    md_table(["Streams"] + labels, md_compare(series, [(str(c), c) for c in levels], get,
                                                              "%" if unit == "%" else ""))]
    sets = acc_sets_named(series)
    for st in [x for x in sets if set_kind(x) == "typed"]:
        out += ["", f"## {st['title']}", "",
                "Every test case in one request (state and its five questions). Accuracy: the argmax of the answer's "
                "distribution against the gold's hard label; gold ties: right when tied for the gold's top. KL from "
                "gold and Brier against the soft gold distribution, Brier also against the one-hot hard label; "
                "Brier sums over the options. ECE: top label, 10 equal-width bins. Score MAE: |expected score − gold "
                "expected score| in rubric levels.", "",
                md_table(["Metric"] + labels, md_metric_rows(series, TYPED_ROWS, lambda s, k: acc_get(s, st["name"], k)))]
        wfs = sorted({w for s in series for x in s.acc_sets if x["name"] == st["name"]
                      for w in (x["summary"].get("by_workflow") or {})})
        if wfs:
            def wf_get(s, w):
                sm = next((x["summary"] for x in s.acc_sets if x["name"] == st["name"]), {})
                v = ((sm.get("by_workflow") or {}).get(w) or {}).get("accuracy")
                return v * 100 if v is not None else None
            out += ["", f"### {st['title']} accuracy by workflow (%, higher is better)", "",
                    md_table(["Workflow"] + labels, md_compare(series, [(w, w) for w in wfs], wf_get, "%", points=True))]
        out += reliability_md(series, st)
    di = [x for x in sets if set_kind(x) == "di"]
    if di:
        out += ["", "## Decision Index components", "",
                "Each component scored as the decision-index kit scores it: accuracy of the answer's `choice` (API-Bank, "
                "GPQA Diamond) or case exact accuracy, every field of a request right (Home appliances), over the "
                "answered rows. Raw is the score times coverage (unanswered counts as wrong); skill is raw "
                "chance-corrected, (raw − chance) / (1 − chance), as on the board. These are single components, "
                "not a Decision Index."]
        for key, name in (("score", "Score"), ("coverage", "Coverage"), ("skill", "Skill, chance-corrected")):
            rows = md_compare(series, [(st["title"], st["name"]) for st in di],
                              lambda s, n, key=key: acc_get(s, n, key), "%", points=True)
            out += ["", f"### {name} (%, higher is better)", "", md_table(["Component"] + labels, rows)]
    classic = [x for x in sets if set_kind(x) == "classic"]
    if classic:
        out += ["", "## AG News and BoolQ" if len(classic) > 1 else f"## {classic[0]['title']}", "",
                "One question per request on a fixed sample. Brier, vector: mean over rows of the sum over classes of "
                "(p − y)²; Brier, binary (BoolQ): mean of (p(yes) − y)², half the two-class vector value. NLL: mean "
                f"−ln p(true class), p clipped at {PROB_EPS:g}. ECE: top label, 10 equal-width bins."]
        for st in classic:
            out += ["", f"### {st['title']}", "",
                    md_table(["Metric"] + labels,
                             md_metric_rows(series, CLASSIC_ROWS, lambda s, k, n=st["name"]: acc_get(s, n, k)))]
            out += reliability_md(series, st)
    if sets:
        out += ["", "Pinned sets:", ""] + [f"- {pinned_set_line(st)}" for st in sets]
    out += ["", f"Produced by `{TOOL_NAME} report`; settings and per-run spread are in report.html.", ""]
    return "\n".join(out)


def ok_sent_td(ok: int, sent: int) -> str:
    return f'<td class="err">{ok}/{sent}</td>' if ok < sent else f"<td>{ok}/{sent}</td>"


def set_counts(st: dict) -> tuple[int, int]:
    sm = st["summary"]
    return sm["answered"], sm.get("items", sm.get("cases", sm.get("requests")))


def set_source_tds(st: dict) -> str:
    ds = st["dataset"]
    kind = set_kind(st)
    if kind == "di":
        return (f"<td>Decision Index {ds['catalog_id']}<small>{esc(ds['track'])}</small></td>"
                f"<td>{st['rows']['n']} of {ds['kept']}<small>{ds['rows']} built</small></td>")
    rows = (f"<td>{st['sample']['n']} of {ds['rows']}<small>seed {st['sample']['seed']}</small></td>" if kind == "classic"
            else f"<td>{st['rows']['n']} of {ds['rows']}<small>cases</small></td>")
    return f"<td>{esc(ds['repo'])} {esc(ds['split'])}<small>{esc(ds['revision'][:12])}</small></td>{rows}"


def decision_series_tables(s: Series) -> str:
    out = []
    i = s.idx % len(PALETTE)

    def has(rows, key, get):
        return key != "memory_peak_gib" or any(get(r, key)["median"] is not None for r in rows)

    def table(name, first, cols, rows, last):
        head = "".join(f"<th>{esc(t)}<small>{esc(u)}</small></th>" for _, t, u in cols)
        return (f'<h3><span class="swatch sw{i}"></span>{esc(s.label)}: {esc(name)}</h3>'
                f'<div class="tablewrap"><table><thead><tr><th>{esc(first)}</th>{head}{last}'
                f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    def ok_td(sm):
        return ok_sent_td(sm["requests"] - sm["errors"], sm["requests"])

    if s.lat_points:
        get = lambda p, k: dec_value(p["summary"], k)  # noqa: E731
        cols = [c for c in DEC_LAT_TABLE if has(s.lat_points, c[0], get)]
        rows = [f"<tr><td>{esc(lat_label(p))}</td>{''.join(cell(get(p, k), u) for k, _, u in cols)}"
                f"{ok_td(p['summary'])}</tr>" for p in s.lat_points]
        out.append(table("latency", "Point", cols, rows, "<th>Runs<small>ok/sent</small></th>"))
    parts = [(name, s.cache[part]["summary"]) for part, name in (("cold", "Cold prefix"), ("repeated", "Repeated state"))
             if part in s.cache]
    if parts:
        rows = [f"<tr><td>{name}</td>{''.join(cell(dec_value(sm, k), u) for k, _, u in DEC_CACHE_TABLE)}"
                f"{ok_td(sm)}</tr>" for name, sm in parts]
        out.append(table("cache test", "Requests", DEC_CACHE_TABLE, rows, "<th>Runs<small>ok/sent</small></th>"))
    if s.dec_levels:
        get = lambda lv, k: dec_value(lv["summary"], k)  # noqa: E731
        cols = [c for c in DEC_CC_TABLE if any(get(lv, c[0])["median"] is not None for lv in s.dec_levels)]
        rows = [f"<tr><td>{lv['concurrency']}</td>{''.join(cell(get(lv, k), u) for k, _, u in cols)}"
                f"{ok_td(lv['summary'])}</tr>" for lv in s.dec_levels]
        out.append(table("throughput", "Streams", cols, rows, "<th>Requests<small>ok/sent</small></th>"))
    for kind, name in (("typed", "Typed Decisions"), ("di", "Decision Index components"), ("classic", "accuracy")):
        sets = [st for st in s.acc_sets if set_kind(st) == kind]
        if not sets:
            continue
        cols = [c for c in ACC_TABLES[kind] if c[0] in ("latency_ms", "memory_peak_gib") and has(sets, c[0], acc_value)
                or c[0] not in ("latency_ms", "memory_peak_gib")]
        head_extra = ("<th>Source<small>catalog, track</small></th><th>Rows<small>scored</small></th>" if kind == "di"
                      else "<th>Dataset<small>revision</small></th><th>Rows<small>scored</small></th>")
        rows = [f"<tr><td>{esc(st['title'])}</td>{set_source_tds(st)}"
                f"{''.join(cell(acc_value(st, k), u) for k, _, u in cols)}{ok_sent_td(*set_counts(st))}</tr>"
                for st in sets]
        cols_html = "".join(f"<th>{esc(t)}<small>{esc(u)}</small></th>" for _, t, u in cols)
        out.append(f'<h3><span class="swatch sw{i}"></span>{esc(s.label)}: {esc(name)}</h3>'
                   f'<div class="tablewrap"><table><thead><tr><th>Set</th>{head_extra}{cols_html}'
                   f'<th>Answered<small>ok/sent</small></th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')
    return "".join(out)


def compare_td(series: list[Series], s: Series, v, base, unit: str, extra=()) -> str:
    """A comparison cell: the value, with any extras and its change against the first series below it."""
    sub = [x for x in extra if x]
    if s is not series[0] and v is not None and base is not None and not many_series(series):
        sub.append(f"{v - base:+.1f} pt" if unit == "%" else pct_change(v, base))
    sub = [x for x in sub if x]
    return f"<td>{fmt(v, unit)}" + (f"<small>{esc(', '.join(sub))}</small>" if sub else "") + "</td>"


def compare_table(series: list[Series], first: str, rows: list[tuple]) -> str:
    """rows: (label html, small text, getter(s) -> value, unit, extra(s) -> [str])."""
    head = "".join(f'<th><span class="swatch sw{s.idx % len(PALETTE)}"></span>{esc(s.label)}</th>' for s in series)
    body = []
    for label, small, get, unit, extra in rows:
        base = get(series[0])
        tds = "".join(compare_td(series, s, get(s), base, unit, extra(s) if extra else ()) for s in series)
        body.append(f"<tr><td>{esc(label)}" + (f"<small>{esc(small)}</small>" if small else "") + f"</td>{tds}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr><th>{esc(first)}</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def cache_section_html(series: list[Series]) -> str:
    cached = [s for s in series if s.cache]
    if not cached:
        return ""
    cs = cached[0].cache["settings"]
    rows = [(f"{name} {pct}", "", (lambda s, part=part, key=key: cache_get(s, part, key)), "", None)
            for part, name in (("cold", "Cold prefix"), ("repeated", "Repeated state"))
            for pct, key in (("p50", "latency_p50_ms"), ("p95", "latency_p95_ms"))]
    rows.append(("Repeated p50 / cold p50", "below 1: the engine reuses work for a state it has seen",
                 cache_ratio, "", None))
    lede = (f"{cs['runs']} requests one at a time, each with a new state (cold prefix), then the same request "
            f"{cs['runs']} times after one untimed first sight (repeated state); {cs['questions']} questions, "
            f"{esc(size_label(cs['state_tokens']))}-token state. Milliseconds, lower is better.")
    return f'<h2 id="cache">Cache</h2><p class="lede">{lede}</p>' + compare_table(series, "Latency, ms", rows)


def quality_section_html(series: list[Series]) -> str:
    sets = acc_sets_named(series)
    if not sets:
        return ""
    out = ['<h2 id="accuracy">Quality</h2>']
    for st in [x for x in sets if set_kind(x) == "typed"]:
        rows = [(label, "", (lambda s, k=key: acc_get(s, st["name"], k)), unit, None)
                for label, key, unit in TYPED_ROWS if any(acc_get(s, st["name"], key) is not None for s in series)]
        out.append(f'<h3>{esc(st["title"])}</h3><p class="lede">Every test case of '
                   f'<code>{esc(st["dataset"]["repo"])}</code> in one request: its state and all five questions. '
                   "Accuracy is against the gold's hard label; KL and the soft Brier against the gold distribution "
                   "(the mean of three teacher samples); lower is better for KL, Brier, ECE and score MAE.</p>"
                   + compare_table(series, "Metric", rows))
    di = [x for x in sets if set_kind(x) == "di"]
    if di:
        def extra(s, n):
            cov, skill = acc_get(s, n, "coverage"), acc_get(s, n, "skill")
            return [f"coverage {fmt(cov, '%')}" if cov is not None and cov < 100 else "",
                    f"skill {fmt(skill, '%')}" if skill is not None else ""]
        rows = [(st["title"], f"{st['summary']['metric']}, Decision Index {st['dataset']['catalog_id']}",
                 (lambda s, n=st["name"]: acc_get(s, n, "score")), "%", (lambda s, n=st["name"]: extra(s, n)))
                for st in di]
        out.append('<h3>Decision Index components</h3><p class="lede">Rebuilt byte for byte as the '
                   f'<a href="{DI_KIT["url"]}">decision-index kit</a> builds them and scored as it scores them, over '
                   "the answered rows. Skill is the score times coverage, chance-corrected as on the board. Single "
                   "components, not a Decision Index.</p>" + compare_table(series, "Component", rows))
    classic = [x for x in sets if set_kind(x) == "classic"]
    if classic:
        def extra_c(s, n):
            bits = []
            for key, name in (("macro_f1", "macro-F1"), ("brier_vector", "Brier"), ("brier", "binary Brier"),
                              ("nll", "NLL"), ("ece", "ECE")):
                v = acc_get(s, n, key)
                if v is not None:
                    bits.append(f"{name} {fmt(v, '%' if key == 'macro_f1' else '')}")
            return bits
        rows = [(st["title"], f"{st['dataset']['repo']} {st['dataset']['split']}",
                 (lambda s, n=st["name"]: acc_get(s, n, "accuracy")), "%", (lambda s, n=st["name"]: extra_c(s, n)))
                for st in classic]
        out.append('<h3>Accuracy and calibration</h3><p class="lede">Share of answered rows right, on a fixed sample '
                   "of each pinned set, one question per request; a noul counts as yes at 0.5 or more. Brier is the "
                   "vector Brier (sum over classes), binary Brier the BoolQ (p(yes) − truth)²; NLL and ECE as in the "
                   "method.</p>" + compare_table(series, "Set", rows))
    return "".join(out)


def decision_meta_rows(d: dict) -> list[tuple]:
    dec = d.get("decisions") or {}
    rows = []
    lat = (dec.get("latency") or {}).get("settings")
    if lat:
        rows.append(("Latency sweep", f"questions {','.join(map(str, lat['question_counts']))}, states "
                                      f"{', '.join(size_label(t) for t in lat['state_targets'])} tokens, "
                                      f"{lat['runs']} runs each, warm-up {lat['warmup']}"))
    cal = dec.get("calibration")
    if cal:
        rows.append(("State sizing", f"{cal['per_sentence_tokens']:.1f} tokens per sentence ({cal['method']})"))
    cs = (dec.get("cache") or {}).get("settings")
    if cs:
        rows.append(("Cache test", f"{cs['runs']} cold + {cs['runs']} repeated, {cs['questions']} questions, "
                                   f"{size_label(cs['state_tokens'])}-token state"))
    cc = (dec.get("concurrency") or {}).get("settings")
    if cc:
        per = f"{cc['requests']} requests closed-loop" if cc.get("requests") else f"{cc.get('rounds')} burst rounds"
        rows.append(("Throughput", f"streams {','.join(map(str, cc['levels']))}, {per}, "
                                   f"{cc['questions']} questions, {size_label(cc['state_tokens'])}-token state"))
    for st in (dec.get("accuracy") or {}).get("sets") or []:
        ds = st["dataset"]
        if set_kind(st) == "di":
            rows.append((st["title"], f"Decision Index {ds['catalog_id']}, kit {ds['kit']['commit'][:12]}, "
                                      f"{st['rows']['n']} of {ds['kept']} rows"))
        elif set_kind(st) == "typed":
            rows.append((st["title"], f"{ds['repo']} {ds['split']} at {ds['revision'][:12]}, "
                                      f"{st['rows']['n']} of {ds['rows']} cases"))
        else:
            rows.append((st["title"], f"{ds['repo']} {ds['split']} at {ds['revision'][:12]}, "
                                      f"{st['sample']['n']} of {ds['rows']} rows, seed {st['sample']['seed']}"))
    return rows


def decision_method_items() -> list[str]:
    return [
        "Every request is one <code>POST /v1/systemone</code> to the CapyCTL inference endpoint, with the "
        "deployment name as <code>model</code>. A decision generates no tokens: there is no time to first "
        "token and no tok/s.",
        "<b>Latency</b>: client side, from the request written to the whole answer read, one request at a time. "
        "p50 and p95 are linear-interpolation percentiles over the runs at a point (100 by default, so p95 rests on "
        "about five requests). <b>Per question</b> is a request's latency over the questions it answered: amortized "
        "cost, not the latency of a single question.",
        "<b>State</b>: deterministic filler text with a seed of its own for every request of the run, warm-ups "
        "included, so no request reuses another's prefix cache (the cache test's repeated block excepted). Its "
        "length is sized from two calibration requests: the growth of <code>usage.input_tokens</code> per filler "
        "sentence (4 characters per token when the engine reports no usage). <b>Input tokens</b> are "
        "<code>usage.input_tokens</code> per request: state, questions and the engine's template, so the same text "
        "can count differently on different models.",
        "<b>Questions</b>: the first N of a fixed set of five (choice, noul, score, choice, noul). The results "
        "file's <code>decisions.manifest</code> holds them, their options and the serialization.",
        "<b>Cache</b>: cold-prefix requests (each a new state) against one state repeated after its first sight.",
        "<b>Throughput</b>: closed loop at a fixed number of streams; each stream sends its next request as soon as "
        "its last one returns, every request its own state. Requests and answered questions per second are the "
        "answered requests (questions) over the time from the first sent to the last finished; errors are the share "
        "of requests that failed.",
        "<b>Typed Decisions</b> (LocalLLaMA/typed-decisions, test, 400 cases): each case's state and five questions "
        "in one request, as the dataset card asks. Accuracy: the answer's most probable option against the gold "
        "label (also reported counting a prediction tied for the gold's top as right, the public third-party "
        "convention). <b>KL</b>(gold ‖ prediction) with predictions clipped at 1e-9, and <b>Brier</b> as the sum "
        "over options of (p − gold)², against the soft gold; hard-label Brier against the one-hot label. "
        "<b>Score MAE</b>: |expected score − gold expected score| in levels.",
        "<b>Decision Index components</b> (API-Bank, GPQA Diamond, Home appliances): rebuilt from pinned sources "
        "exactly as the decision-index kit builds them (byte-identical to its normalized rows), with its "
        "exclusions and subsets, sent unchanged, and scored on the answer's <code>choice</code>: accuracy, or case "
        "exact accuracy (every field right). Raw is score × coverage; skill = (raw − chance) / (1 − chance), "
        "clipped to [0, 1].",
        "<b>AG News and BoolQ</b>: a fixed sample of each pinned set, one question per request. Accuracy and "
        "macro-F1 over the answered rows; a noul counts as yes at 0.5 or more. <b>Brier, vector</b>: mean of "
        "Σₖ(pₖ − yₖ)²; <b>Brier, binary</b>: mean of (p(yes) − y)², half the two-class vector value. <b>NLL</b>: "
        "mean −ln p(true class), p clipped at 1e-9.",
        "<b>ECE</b>: top label (the most probable option and its probability), 10 equal-width confidence bins, the "
        "count-weighted mean of |accuracy − mean confidence|. The reliability charts plot each bin's accuracy at "
        "its mean confidence; the bin counts are in summary.md and data.csv.",
        "Rows come from the Hugging Face datasets-server (checked against the sha256 pinned with the revision) or "
        "are rebuilt from pinned sources (checked against the kit's sha256). Results keep ids, hashes and answers, "
        "never the rows' text.",
        "<b>Peak memory</b>: the highest value the memory command printed while the request, level or set ran.",
        "Tables show the median with the min–max range across runs below it.",
    ]


def decision_headline(series: list[Series]) -> dict | None:
    """Bars of the headline: requests/s at the fewest and most streams, else latency p50."""
    levels = None
    for s in series:
        lv = {l["concurrency"] for l in s.dec_levels if dec_value(l["summary"], "requests_per_s")["median"] is not None}
        levels = lv if levels is None else levels & lv
    if levels:
        groups = [{"label": f"{c} stream{'s' if c > 1 else ''}",
                   "bars": [(s, level_get(s, c, "requests_per_s")) for s in series]}
                  for c in dict.fromkeys([min(levels), max(levels)])]
        return {"title": "Requests per second", "unit": "requests/s", "better": "higher", "groups": groups}
    common = None
    for s in series:
        keys = {lat_key(p) for p in s.lat_points if p["summary"].get("latency_p50_s") is not None}
        common = keys if common is None else common & keys
    if common:
        groups = [{"label": f"{q} question{'s' if q > 1 else ''}, {size_label(t)} tok",
                   "bars": [(s, lat_get(s, (q, t), "latency_p50_ms")) for s in series]}
                  for q, t in dict.fromkeys([min(common), max(common)])]
        return {"title": "Latency p50", "unit": "ms", "better": "lower", "groups": groups}
    return None


def quality_bars(series: list[Series], sets: list[dict], title: str) -> dict:
    """Summary bars of each set's headline number; group labels short enough for a half-width panel."""
    return {"kind": "bars", "title": title, "unit": "%", "better": "higher", "delta": "pt",
            "groups": [{"label": ACCURACY_SETS.get(st["name"], {}).get("short", st["title"]),
                        "bars": [(s, acc_get(s, st["name"], headline_key(st))) for s in series]} for st in sets]}


def many_series(series: list[Series]) -> bool:
    """Decision results compared as a field of models: more than three series, or series of different
    models. Their summary is a scoreboard instead of bars, and no table gives changes against the first
    series (a change against one model means nothing in a field of them)."""
    return bool(series) and is_decisions(series) and (
        len(series) > 3 or len({s.data.get("model") for s in series}) > 1)


def nearest_point(series: list[Series], questions: int, tokens: int) -> tuple | None:
    """The latency point (questions, state tokens) of the results nearest the one asked for."""
    keys = {lat_key(p) for s in series for p in s.lat_points}
    if not keys:
        return None
    return min(keys, key=lambda k: (abs(k[0] - questions), abs(math.log(k[1] / tokens))))


def peak_memory(s: Series):
    """The highest --memory-cmd sample of the whole run, GiB."""
    return max((g for _, g in ((s.data.get("memory") or {}).get("samples") or [])), default=None)


def rank_shades(texts: list, better: str) -> list:
    """1 for the best value of a column down to 0 for the worst, by the values as displayed (so values
    that print the same tie); None where there is no value."""
    def num(t):
        return float(re.sub(r"[^0-9.\-]", "", t))
    vals = sorted({num(t) for t in texts if t is not None}, reverse=better == "higher")
    return [None if t is None else (1.0 if len(vals) == 1 else 1 - vals.index(num(t)) / (len(vals) - 1))
            for t in texts]


def shade_alpha(shade: float) -> float:
    return 0.08 + 0.8 * shade


def scoreboard_panel(series: list[Series]) -> dict | None:
    cols = []

    def add(head, unit, better, values):
        if any(v is not None for v in values):
            # the header carries the unit: a share prints as 72.4, not 72.4%
            texts = [None if v is None else (f"{v:.1f}" if unit == "%" else fmt(v, unit)) for v in values]
            cols.append({"head": head, "unit": unit, "better": better, "values": values, "texts": texts,
                         "shades": rank_shades(texts, better)})

    # Headers are three short lines (what, metric, unit or point) so ten columns fit a half page.
    for name, key, head in (("typed_decisions", "accuracy", ("Typed", "accuracy", "%")),
                            ("api_bank", "score", ("API-Bank", "score", "%")),
                            ("gpqa_diamond", "score", ("GPQA", "score", "%")),
                            ("home_appliances", "score", ("Home", "score", "%")),
                            ("ag_news", "accuracy", ("AG News", "accuracy", "%")),
                            ("boolq", "accuracy", ("BoolQ", "accuracy", "%")),
                            ("typed_decisions", "ece", ("Typed", "ECE", "↓"))):
        unit = "" if key == "ece" else "%"
        add(head, unit, "lower" if key in LOWER_BETTER else "higher", [acc_get(s, name, key) for s in series])
    k = nearest_point(series, 1, 2000)
    if k:
        add(("Latency", f"{k[0]}q {size_label(k[1])}", "ms ↓"), "", "lower",
            [lat_get(s, k, "latency_p50_ms") for s in series])
    cc = next(((s.decisions.get("concurrency") or {}).get("settings") for s in series if s.dec_levels), None)
    if cc:
        add(("Questions/s", "1 stream", f"{cc['questions']}q {size_label(cc['state_tokens'])}"), "", "higher",
            [level_get(s, 1, "questions_per_s") for s in series])
    add(("Memory", "peak", "GiB ↓"), "", "lower", [peak_memory(s) for s in series])
    if not cols:
        return None
    return {"kind": "scoreboard", "title": "Scoreboard", "unit": "%",
            "caption": "Typed = Typed Decisions; each column shaded by rank, darker is better; ↓ lower is better",
            "rows": series, "cols": cols}


def quality_speed_panel(series: list[Series]) -> dict | None:
    k = nearest_point(series, 3, 2000)
    if not k:
        return None
    pts = [(s, lat_get(s, k, "per_question_p50_ms"), acc_get(s, "typed_decisions", "accuracy")) for s in series]
    pts = [p for p in pts if p[1] and p[2] is not None]
    if not pts:
        return None
    return {"kind": "scatter", "title": "Quality vs speed", "unit": "Typed Decisions accuracy, %",
            "caption": "Typed Decisions accuracy %; up-left is better",
            "xlabel": f"ms per answered question ({k[0]}q {size_label(k[1])}, p50, log)",
            "points": pts}


def field_panels(series: list[Series]) -> list[dict]:
    """The summary of a field of models: a scoreboard, quality against speed, latency against state."""
    panels = [p for p in (scoreboard_panel(series), quality_speed_panel(series)) if p]
    sp = {s["id"]: s for s in decision_chart_specs(series)}.get(f"latency-{(nearest_point(series, 1, 2000) or (1,))[0]}q")
    lines = [ln for ln in (sp or {}).get("lines", []) if not ln["dash"]]
    if lines:
        title = sp["title"].replace("Latency,", "Latency p50,")
        panels.append({"kind": "line", "title": title, "unit": "ms",
                       "caption": "ms, lower is better, log scale",
                       "spec": dict(sp, title=title, lines=lines, ylog=True)})
    return panels


def place_labels(anchors: list, sizes: list, bounds: tuple, marker: float = 6.0) -> list:
    """A box (x0, y0, x1, y1), y up, for each point's label: the candidate around the point (right,
    left, above, below, the diagonals, at growing distances) that overlaps the labels placed so far,
    the points and the bounds least, the most crowded points first."""
    bx0, by0, bx1, by1 = bounds
    points = [(x - marker, y - marker, x + marker, y + marker) for x, y in anchors]

    def overlap(a, b):
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        return w * h if w > 0 and h > 0 else 0.0

    def crowd(i):
        x, y = anchors[i]
        return -sum(1 for (u, v) in anchors if abs(u - x) < 8 * marker and abs(v - y) < 4 * marker)

    def leader(i, box):
        """Points along the line from point i to its label, when the label sits far enough to need one."""
        x, y = anchors[i]
        cx, cy = min(max(x, box[0]), box[2]), min(max(y, box[1]), box[3])
        if math.hypot(cx - x, cy - y) <= 2 * marker:
            return []
        return [(x + (cx - x) * k / 24, y + (cy - y) * k / 24) for k in range(4, 24)]

    def inside(p, b):
        return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]

    def crosses(s, t):
        """Whether two leader lines (their samples, first to last) cross."""
        if not s or not t:
            return False
        (a, b), (c, d) = (s[0], s[-1]), (t[0], t[-1])

        def side(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return side(a, b, c) * side(a, b, d) < 0 and side(c, d, a) * side(c, d, b) < 0

    placed: list = [None] * len(anchors)
    leaders: list = [[] for _ in anchors]

    def best_box(i):
        x, y = anchors[i]
        w, h = sizes[i]
        others = [b for j, b in enumerate(placed) if b and j != i]
        other_lines = [p for j, ln in enumerate(leaders) if j != i for p in ln]
        other_leaders = [ln for j, ln in enumerate(leaders) if j != i and ln]
        best = None
        for r in (marker * 1.5, marker * 2.6, marker * 4.2, marker * 6.5, marker * 9.5, marker * 13):
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
                cx, cy = x + dx * r, y + dy * r
                x0 = cx if dx > 0 else cx - w if dx < 0 else cx - w / 2
                y0 = cy if dy > 0 else cy - h if dy < 0 else cy - h / 2
                box = (x0, y0, x0 + w, y0 + h)
                cost = 50 * sum(overlap(box, b) for b in others) + 50 * sum(overlap(box, p) for p in points)
                out = max(0, bx0 - box[0]) + max(0, box[2] - bx1) + max(0, by0 - box[1]) + max(0, box[3] - by1)
                # nearer is better: a label far from its point needs a leader line
                cost += 40 * out * (w + h) + r * h + (2 if dy else 0)
                # a leader line must not cross another label, nor this label another's leader
                own = leader(i, box)
                crossings = sum(inside(p, b) for p in own for b in others)
                crossings += sum(inside(p, box) for p in other_lines)
                cost += crossings * w * h / 4 + sum(crosses(own, ln) for ln in other_leaders) * w * h
                if best is None or cost < best[0]:
                    best = (cost, box)
        return best[1]

    # Greedy, the most crowded points first, then a few passes that move each label to its best place
    # given all the others, so the result does not hang on the order.
    order = sorted(range(len(anchors)), key=lambda i: (crowd(i), anchors[i]))
    for _ in range(4):
        for i in order:
            placed[i] = best_box(i)
            leaders[i] = leader(i, placed[i])
    return placed


def board_html(pnl: dict) -> str:
    cols = pnl["cols"]
    head = "".join(f"<th>{esc(c['head'][0])}<small>{esc(' '.join(c['head'][1:]))}</small></th>" for c in cols)
    body = []
    for r, s in enumerate(pnl["rows"]):
        tds = []
        for c in cols:
            t, sh = c["texts"][r], c["shades"][r]
            if t is None:
                tds.append("<td>–</td>")
                continue
            a = shade_alpha(sh)
            cls = "b" + (" hi" if a > 0.55 else "") + (" hi2" if a > 0.72 else "")
            tds.append(f'<td class="{cls}" style="--a:{a * 100:.0f}%">{esc(t)}</td>')
        body.append(f'<tr><td><span class="swatch sw{s.idx % len(PALETTE)}"></span>{esc(s.label)}</td>{"".join(tds)}</tr>')
    return (f'<div class="tablewrap"><table class="board"><thead><tr><th>Model</th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def log_ticks_125(lo: float, hi: float) -> list[float]:
    """1, 2, 5 x 10^n ticks in [lo, hi]; decades only when that is too many."""
    out = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        out += [m * 10.0 ** e for m in (1, 2, 5) if lo <= m * 10.0 ** e <= hi]
    return out if len(out) <= 7 else [t for t in out if f"{t:g}".lstrip("0.").startswith("1")]


def scatter_domain(pnl: dict) -> tuple:
    xs = [p[1] for p in pnl["points"]]
    ys = [p[2] for p in pnl["points"]]
    x0, x1 = min(xs) / 1.5, max(xs) * 1.5
    y0, y1 = max(0.0, min(ys) - 6), min(100.0, max(ys) + 6)
    if y1 - y0 < 10:
        y0, y1 = max(0.0, y0 - 5), min(100.0, y1 + 5)
    return x0, x1, y0, y1


def svg_scatter(pnl: dict) -> str:
    W, H = 460, 300
    ml, mr, mt, mb = 54, 14, 12, 46
    pw, ph = W - ml - mr, H - mt - mb
    x0, x1, y0, y1 = scatter_domain(pnl)
    yt = [t for t in nice_ticks(y0, y1) if y0 <= t <= y1] or [y0, y1]

    def X(v):
        return ml + (math.log10(v) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * pw

    def Y(v):
        return mt + ph - (v - y0) / ((y1 - y0) or 1) * ph

    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(pnl["title"])}" '
           'xmlns="http://www.w3.org/2000/svg">']
    for t in yt:
        out.append(f'<line class="grid" x1="{ml}" x2="{W - mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in log_ticks_125(x0, x1):
        out.append(f'<line class="axis" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt + ph}" y2="{mt + ph + 4}"/>'
                   f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 18}" text-anchor="middle">{t:g}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{W - mr}" y1="{mt + ph}" y2="{mt + ph}"/>'
               f'<text class="axlabel" x="{ml + pw / 2:.1f}" y="{H - 6}" text-anchor="middle">'
               f'{esc(pnl["xlabel"])}</text><text class="axlabel" transform="translate(13 {mt + ph / 2:.1f}) '
               f'rotate(-90)" text-anchor="middle">%</text>')
    fs = 11
    anchors = [(X(x), H - Y(y)) for _, x, y in pnl["points"]]
    sizes = [(text_width_em(s.label) * fs * 0.86, fs * 1.2) for s, _, _ in pnl["points"]]
    boxes = place_labels(anchors, sizes, (ml, H - mt - ph, W - mr, H - mt), marker=5)
    for (s, x, y), (ax_, ay), b in zip(pnl["points"], anchors, boxes):
        i = s.idx % len(PALETTE)
        cx, cy = min(max(ax_, b[0]), b[2]), min(max(ay, b[1]), b[3])
        if math.hypot(cx - ax_, cy - ay) > 12:
            out.append(f'<line class="axis" x1="{ax_:.1f}" y1="{H - ay:.1f}" x2="{cx:.1f}" y2="{H - cy:.1f}"/>')
        out.append(marker_svg(MARKERS[i], X(x), Y(y), f"mk s{i}", f"{s.label}: {fmt(y, '%')}, {fmt(x)} ms"))
        out.append(f'<text class="tick lbl" x="{b[0]:.1f}" y="{H - b[1] - 3:.1f}">{esc(s.label)}</text>')
    out.append("</svg>")
    return "".join(out)


def decision_summary_panels(series: list[Series]) -> list[dict]:
    """Quality first when the results have it (accuracy, Decision Index components), then speed; a
    field of models (`many_series`) gets the scoreboard layout instead."""
    if many_series(series):
        return field_panels(series)
    specs = {sp["id"]: sp for sp in decision_chart_specs(series)}
    panels = []
    sets = acc_sets_named(series)
    acc = [st for st in sets if set_kind(st) != "di"]
    di = [st for st in sets if set_kind(st) == "di"]
    if acc:
        panels.append(quality_bars(series, acc, "Accuracy"))
    if di:
        panels.append(quality_bars(series, di, "Decision Index components"))
    if not panels:
        head = decision_headline(series)
        if head:
            panels.append(dict(head, kind="bars"))
    qs = [p["questions"] for s in series for p in s.lat_points]
    if qs:
        qmax = max(qs)
        sp = specs.get(f"latency-{qmax}q")
        lines = [ln for ln in (sp or {}).get("lines", []) if not ln["dash"]]
        if lines:
            title = f"Latency p50, {qmax} question{'s' if qmax > 1 else ''}"
            panels.append({"kind": "line", "title": title, "unit": "ms",
                           "spec": dict(sp, title=title, lines=lines)})
    sp = specs.get("concurrency-questions_per_s") or specs.get("concurrency-requests_per_s")
    if sp and sp["lines"]:
        title = "Questions/s vs streams" if sp["metric"] == "questions_per_s" else "Requests/s vs streams"
        panels.append({"kind": "line", "title": title, "unit": sp["unit"], "spec": dict(sp, title=title)})
    return panels[:4]


def decision_notes(first: dict) -> list[str]:
    dec = first.get("decisions") or {}
    notes = []
    lat = (dec.get("latency") or {}).get("settings")
    if lat:
        t = lat["state_targets"]
        notes.append(f"states {size_label(min(t))}–{size_label(max(t))} tokens, "
                     f"{','.join(map(str, lat['question_counts']))} questions, {lat['runs']} runs per point")
    cc = (dec.get("concurrency") or {}).get("settings")
    if cc:
        lv = cc["levels"]
        streams = f"{lv[0]} stream{'s' if lv[0] > 1 else ''}" if len(set(lv)) == 1 else f"{min(lv)}–{max(lv)} streams"
        per = f"{cc['requests']} requests closed-loop" if cc.get("requests") else f"{cc.get('rounds')} rounds"
        notes.append(f"{streams}, {cc['questions']} questions, {size_label(cc['state_tokens'])}-token state, {per}")
    acc = (dec.get("accuracy") or {}).get("sets") or []
    classic = [st for st in acc if set_kind(st) == "classic"]
    if classic:
        ns = {st["sample"]["n"] for st in classic}
        names = " and ".join(st["title"] for st in classic)
        rows = f"{ns.pop()} rows each of {names}" if len(ns) == 1 else \
            ", ".join(f"{st['sample']['n']} rows of {st['title']}" for st in classic)
        notes.append(f"accuracy on {rows} (seed {classic[0]['sample']['seed']})")
    for st in acc:
        if set_kind(st) == "typed":
            notes.append(f"{st['title']} {st['rows']['n']} cases")
    di = [st for st in acc if set_kind(st) == "di"]
    if di:
        notes.append("components (not an index) " + ", ".join(f"{st['title']} {st['rows']['n']}" for st in di))
    notes.append("latency client side")
    return notes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="capyctl_bench.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="measure an endpoint and write a results JSON file")
    r.add_argument("--decisions", action="store_true",
                   help="measure a decision model through POST /v1/systemone instead of chat completions")
    r.add_argument("--endpoint", default="http://127.0.0.1:8443/v1", help="OpenAI-compatible base URL")
    r.add_argument("--api-key-file", help="CapyCTL credentials file (api_key line); else $CAPYCTL_API_KEY")
    r.add_argument("--model", required=True, help="model name the endpoint serves (the deployment name)")
    r.add_argument("--label", required=True, help='series name in reports, e.g. "TensorFold 0.6.2"')
    r.add_argument("--out", default="results.json", help="results file to write")
    r.add_argument("--context-sweep", type=parse_sweep, help="prompt sizes, e.g. 0.5k,1k,2k,4k (k = 1000 tokens)")
    r.add_argument("--runs", type=int,
                   help="requests per context or latency point (default 3; 100 with --decisions)")
    r.add_argument("--max-tokens", type=int, default=128, help="max_tokens for the context sweep (default 128)")
    r.add_argument("--concurrency", type=parse_levels,
                   help="stream counts, e.g. 1-8 or 1,2,4,8 (with --decisions the default is 1,2,4,8)")
    r.add_argument("--rounds", type=int, help="rounds per concurrency level of a chat run (default 5)")
    r.add_argument("--concurrency-max-tokens", type=int, default=512,
                   help="max_tokens for concurrency rounds (default 512)")
    r.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompt set for concurrency (default prompts.json)")
    r.add_argument("--warmup", type=int,
                   help="unrecorded warm-up requests or rounds (default 1; with --decisions 5, before every "
                        "latency point, the cache test and each throughput level)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--extra-body", help='JSON object merged into every request, e.g. \'{"ignore_eos": true}\'')
    r.add_argument("--memory-cmd", help="shell command printing used memory in GiB; sampled during the run")
    r.add_argument("--memory-interval", type=float, default=0.5, help="seconds between memory samples")
    r.add_argument("--record-chunks", action="store_true", help="keep every token chunk's arrival time")
    r.add_argument("--meta", action="append", metavar="KEY=VALUE",
                   help="run metadata, e.g. capyctl=0.2.0 engine=vllm-0.30.0 gpu=GB10 (repeatable)")
    r.add_argument("--timeout", type=float, default=900.0, help="socket timeout per request, seconds")
    d = r.add_argument_group("decisions (with --decisions)")
    d.add_argument("--question-counts", type=parse_question_counts,
                   help="questions per latency request, 1 to 5 (default 1,3,5)")
    d.add_argument("--state-sweep", type=parse_sweep,
                   help="state lengths of the latency sweep, k = 1000 tokens (default 0.2k,2k,8k)")
    d.add_argument("--concurrency-questions", type=int,
                   help="questions per throughput and cache-test request (default 3)")
    d.add_argument("--concurrency-state", type=parse_size,
                   help="state length of throughput and cache-test requests (default 2k)")
    d.add_argument("--cache-runs", type=int,
                   help="requests of each half of the cache test, cold states then one repeated state "
                        "(default 100; 0 skips it)")
    d.add_argument("--throughput-requests", type=int,
                   help="requests per closed-loop throughput level (default 200; 0 skips throughput)")
    d.add_argument("--accuracy", action="store_true", help="also measure quality on the built-in pinned sets")
    d.add_argument("--accuracy-sets", type=parse_sets,
                   help=f"sets or suites to run: {', '.join(ACCURACY_SETS)}, or {', '.join(SUITES)} "
                        "(default all)")
    d.add_argument("--accuracy-samples", type=int, help="rows sampled from AG News and BoolQ (default 500)")
    d.add_argument("--accuracy-seed", type=int, help="seed of the AG News and BoolQ sample (default 0)")
    d.add_argument("--accuracy-limit", type=int,
                   help="score only the first N rows of Typed Decisions and the Decision Index sets, for a "
                        "smoke test (default 0: all; the results say partial)")
    d.add_argument("--datasets-url", help=f"datasets-server base URL (default {DATASETS_URL})")
    d.add_argument("--dataset-cache", help=f"directory of downloaded and built sets (default {DATASET_CACHE})")
    d.add_argument("--plan", action="store_true",
                   help="print the requests and an estimated time of this run, then exit without sending any")
    d.add_argument("--plan-overhead-ms", type=float, default=60.0,
                   help="--plan: assumed fixed time per request, ms (default 60)")
    d.add_argument("--plan-prefill-tps", type=float, default=2500.0,
                   help="--plan: assumed input tokens processed per second (default 2500)")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="render results files into one report")
    rp.add_argument("results", nargs="+", help="results JSON files; the first is the baseline")
    rp.add_argument("--out", required=True, help="output directory")
    rp.add_argument("--title", help="report title (default: the labels)")
    rp.add_argument("--relabel", action="append", metavar="FILE_LABEL=NEW_LABEL",
                    help="show the series labelled FILE_LABEL in its results file as NEW_LABEL (repeatable)")
    rp.add_argument("--summary", action="store_true",
                    help="also write summary/: a one-page shareable summary (HTML, wide and tall PNGs, light and dark)")
    rp.add_argument("--theme", choices=("light", "dark"), default="light", help="theme of charts/*.png (default light)")
    rp.add_argument("--no-png", action="store_true", help="skip every PNG")
    rp.add_argument("--no-zip", action="store_true", help="skip report.zip")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
