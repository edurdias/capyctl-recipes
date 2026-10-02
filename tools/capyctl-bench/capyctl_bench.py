#!/usr/bin/env python3
"""capyctl-bench: a standard benchmark and report for CapyCTL recipes.

`run` measures an OpenAI-compatible endpoint (CapyCTL's inference endpoint)
with streaming chat completions and writes one results JSON file.
`report` overlays any number of results files on the same charts and tables.

Standard library only. The PNG charts of `report` need matplotlib; run it with
`uv run --with matplotlib python3 capyctl_bench.py report ...`. Without
matplotlib the report still writes report.html, summary.md and data.csv.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import html
import http.client
import json
import math
import os
import re
import ssl
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path

TOOL_NAME = "capyctl-bench"
TOOL_VERSION = "0.1.0"
SCHEMA_VERSION = 1
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

    def headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
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
        rec["error"] = "no content or reasoning tokens in the stream"
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
        raise argparse.ArgumentTypeError("empty --context-sweep")
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
            if rec["error"] or not rec["prompt_tokens"]:
                raise SystemExit(f"calibration request failed: {rec['error'] or 'no usage.prompt_tokens'}")
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
        levels = []
        for c in a.concurrency:
            for w in range(a.warmup):
                self.round_(c, prompts, 0, a.concurrency_max_tokens)
            rounds = []
            for r in range(a.rounds):
                rnd = self.round_(c, prompts, r * c, a.concurrency_max_tokens)
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


def cmd_run(args) -> int:
    if not args.context_sweep and not args.concurrency:
        raise SystemExit("give --context-sweep, --concurrency or both")
    key = read_api_key(args.api_key_file)
    ep = Endpoint(args.endpoint, key, timeout=args.timeout)
    sampler = NoSampler()
    if args.memory_cmd:
        sampler = MemorySampler(args.memory_cmd, args.memory_interval)
        if sampler.sample_once() is None:
            raise SystemExit(f"--memory-cmd failed: {sampler.last_error}")
        sampler.start()
    runner = Runner(args, ep, sampler)
    result = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION,
                 "python": ".".join(map(str, sys.version_info[:3]))},
        "label": args.label,
        "model": args.model,
        "endpoint": ep.redacted(),
        "started_at": utc_now(),
        "finished_at": None,
        "meta": parse_meta(args.meta),
        "settings": {"temperature": args.temperature, "extra_body": runner.extra_body,
                     "memory_command": bool(args.memory_cmd),
                     "memory_interval_s": args.memory_interval if args.memory_cmd else None,
                     "record_chunks": args.record_chunks},
        "context_sweep": None,
        "concurrency": None,
        "memory": None,
    }
    try:
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


def is_sample(series: list[Series]) -> bool:
    return any((s.data.get("meta") or {}).get("sample") in (True, "true") for s in series)


def load_series(paths) -> list[Series]:
    series = []
    seen: dict[str, int] = {}
    for i, p in enumerate(paths):
        p = Path(p)
        data = json.loads(p.read_text())
        v = data.get("schema_version")
        if v != SCHEMA_VERSION:
            raise SystemExit(f"{p}: schema_version {v!r}, this tool reads {SCHEMA_VERSION}")
        label = data.get("label") or p.stem
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({seen[label]})"
        series.append(Series(i, p, data, label))
    return series


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
    for key, title, unit, better in CONCURRENCY_METRICS:
        if not has_metric_cc(series, key):
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
    if spec["xlog"]:
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
        ymax = max(ys) * 1.08 if max(ys) > 0 else 1
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
    if spec["xlog"]:
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
        ax.set_ylim(0, y_top(p[1] for ln in spec["lines"] for p in ln["points"]))
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


def summary_md(series: list[Series], title: str) -> str:
    base = series[0]
    out = [f"# {title}", ""]
    if is_sample(series):
        out += [f"> **{SAMPLE_BANNER}.** These numbers come from the test fake server.", ""]
    out.append(f"Measured through CapyCTL with {TOOL_NAME} {TOOL_VERSION}. Each cell is the median "
               f"of the runs at that point; percentages compare with {base.label}.")
    out.append("")
    out.append(md_table(["Series", "Model", "Measured", "Meta"], [
        [s.label, f"`{s.data.get('model', '')}`", (s.data.get("started_at") or "")[:10],
         ", ".join(f"{k}={v}" for k, v in (s.data.get("meta") or {}).items()) or "–"]
        for s in series]))
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
    rows.append(("Temperature", st.get("temperature")))
    if st.get("extra_body"):
        rows.append(("Extra body", json.dumps(st["extra_body"])))
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
              ("concurrency settings", ("concurrency", "settings"))]
    for name, path in checks:
        if len({key(s, path) for s in series}) > 1:
            notes.append(name)
    return notes


def method_html(series: list[Series]) -> str:
    items = [
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
            extra = " Dashed lines are p95." if sp["metric"] == "ttft" else ""
            cards.append(f'<div class="card"><h3>{esc(sp["title"])} <small>{esc(sp["unit"])}, '
                         f'{esc(sp["better"])} is better.{extra}</small></h3>{svg}</div>')
        if not cards:
            return ""
        return (f'<h2 id="{name}">{esc(heading)}</h2><p class="lede">{intro}</p>'
                f'<div class="legend">{legend}</div><div class="grid2">{"".join(cards)}</div>')

    ctx = section("context", "Context sweep",
                  "One request at a time, prompt length growing. x axis is log scale.")
    cc = section("concurrency", "Concurrency", "N streams started together from a fixed prompt set.")
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
{ctx}
{cc}
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
    """Bars of the headline panel: generation tok/s at ~1k and at the largest common context.

    Without a context sweep: aggregate tok/s at 1 stream and at the most streams all series ran.
    """
    def ctx_val(s, lab):
        return next((ctx_value(p, "decode_tps")["median"] for p in s.context_points if p["label"] == lab), None)

    common = None
    for s in series:
        labs = {p["label"]: p["target_prompt_tokens"] for p in s.context_points
                if ctx_value(p, "decode_tps")["median"] is not None}
        common = labs if common is None else {k: v for k, v in common.items() if k in labs}
    groups = []
    if common:
        short = min(common, key=lambda k: abs(math.log(common[k] / 1000)))
        long_ = max(common, key=lambda k: common[k])
        for lab in dict.fromkeys([short, long_]):
            groups.append({"label": f"{lab} prompt", "bars": [(s, ctx_val(s, lab)) for s in series]})
        return "Generation, one stream", groups
    levels = None
    for s in series:
        lv = {l["concurrency"] for l in s.levels if cc_value(l, "aggregate_tps")["median"] is not None}
        levels = lv if levels is None else levels & lv
    if levels:
        for c in dict.fromkeys([min(levels), max(levels)]):
            groups.append({"label": f"{c} stream{'s' if c > 1 else ''}",
                           "bars": [(s, next(cc_value(l, "aggregate_tps")["median"] for l in s.levels
                                             if l["concurrency"] == c)) for s in series]})
        return "Generation, aggregate", groups
    return "", []


def summary_panels(series: list[Series]) -> list[dict]:
    """2 to 4 panels, chosen from what the results contain."""
    specs = {sp["id"]: sp for sp in chart_specs(series)}
    panels = []
    title, groups = headline_groups(series)
    if groups:
        panels.append({"kind": "bars", "title": title, "unit": "tok/s", "groups": groups})
    for sid, short in (("context-prompt_tps", "Prompt processing"), ("context-ttft_s", "Time to first token"),
                       ("concurrency-aggregate_tps", "Throughput vs streams")):
        sp = specs.get(sid)
        if not sp or not sp["lines"]:
            continue
        if sp["section"] == "concurrency" and len({p[0] for ln in sp["lines"] for p in ln["points"]}) < 2:
            continue  # one stream count: no line against streams to draw
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
    notes = []
    cs = (first.get("context_sweep") or {}).get("settings")
    if cs:
        # The points' own labels: an adapted run's 512 ... 131072 reads 0.5k ... 128k.
        pts = sorted((first.get("context_sweep") or {}).get("points") or [], key=lambda p: p["target_prompt_tokens"])
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


def bar_note(series: list[Series], s: Series, v, base) -> str:
    """% vs the first series on N-series bars; nothing for one series or the baseline."""
    if len(series) < 2 or s is series[0] or v is None or not base:
        return ""
    return pct_change(v, base)


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
                       f'<title>{esc(s.label)}: {fmt(v)} {esc(panel["unit"])}</title></rect>')
            note = bar_note(series, s, v, base)
            ty = y - (24 if note else 8)
            out.append(f'<text class="barval" x="{x + w / 2:.1f}" y="{ty:.1f}" text-anchor="middle">{fmt(v)}</text>')
            if note:
                out.append(f'<text class="barpct" x="{x + w / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle">{note}</text>')
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
.s-grid h3{font-size:18px}
.s-grid h3 small{font-size:14px}
svg .barval{fill:var(--text);font-size:20px;font-weight:700}
svg .barpct{fill:var(--muted);font-size:14px;font-weight:500}
svg .grp{font-size:14px;fill:var(--text)}
.s-foot{color:var(--muted);font-size:14px;margin:18px 0 0;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
.s-foot b{color:var(--text);font-weight:500}
@media screen and (max-width:760px){.s-grid{grid-template-columns:minmax(0,1fr)}.s-head h1{font-size:26px}
.s-head p{font-size:15px}svg .barval{font-size:24px}svg .barpct{font-size:17px}svg .grp{font-size:17px}}
"""


def summary_html(series: list[Series], panels: list[dict], text: dict, pngs: list[str]) -> str:
    legend = "".join(f'<span><i class="sw{s.idx % len(PALETTE)}"></i>{esc(s.label)}</span>' for s in series)
    cards = []
    for pnl in panels:
        if pnl["kind"] == "bars":
            svg = svg_bars(pnl, series)
            better = "higher is better"
        else:
            svg = svg_chart(pnl["spec"])
            better = f"{pnl['spec']['better']} is better" + (", log scale" if pnl["spec"].get("ylog") else "")
        cards.append(f'<div class="card"><h3>{esc(pnl["title"])} <small>{esc(pnl["unit"])}, {better}</small></h3>{svg}</div>')
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
    tsize = 25 if kind == "wide" else 29
    width = 62 if kind == "wide" else 40
    lines = textwrap.wrap(text["title"], width) or [""]
    for ln in lines:
        fig.text(fx(pad), fy(y), ln, fontsize=tsize, fontweight="bold", color=T["text"], va="top")
        y += tsize * 1.45
    y += 4
    fig.text(fx(pad), fy(y), text["subtitle"], fontsize=13.5 if kind == "wide" else 15, color=T["muted"], va="top")
    y += 30 if kind == "wide" else 34
    # legend: colored squares with labels
    from matplotlib.patches import Rectangle
    lx = pad
    lsize = 13 if kind == "wide" else 15
    for s in series:
        i = s.idx % len(pal)
        fig.patches.append(Rectangle((fx(lx), fy(y + 15)), fx(14), 14 / hpx, color=pal[i],
                                     transform=fig.transFigure, figure=fig))
        t = fig.text(fx(lx + 22), fy(y + 1), s.label, fontsize=lsize, color=T["text"], va="top")
        width_px = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / 2  # dpi 200 = 2x
        lx += 22 + width_px + 28
        if lx > wpx - 200:
            lx = pad
            y += 26
    y += 58
    foot_h = 58 if kind == "wide" else 100
    top, bottom = fy(y), (foot_h + 12) / hpx
    (rows, cols, slots), = _summary_layout(kind, len(panels))
    gs = fig.add_gridspec(rows, cols, left=fx(pad + 8), right=1 - fx(pad), top=top - 30 / hpx,
                          bottom=bottom + (34 if kind == "wide" else 50) / hpx, hspace=0.62 if kind == "tall" else 1.15, wspace=0.22)
    big = 15 if kind == "wide" else 17
    for pnl, (r0, r1, c0, c1) in zip(panels, slots):
        ax = fig.add_subplot(gs[r0:r1, c0:c1])
        ax.set_facecolor(T["bg"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(T["rule"])
        ax.spines["bottom"].set_color(T["rule"])
        ax.tick_params(labelsize=11 if kind == "wide" else 12.5, length=0, pad=6)
        if pnl["kind"] == "bars":
            groups = pnl["groups"]
            n = len(series)
            bw = min(0.8 / n, 0.42)
            vmax = max((v for g in groups for _, v in g["bars"] if v is not None), default=1) or 1
            for gi, g in enumerate(groups):
                base = g["bars"][0][1]
                for k, (s, v) in enumerate(g["bars"]):
                    if v is None:
                        continue
                    x = gi + (k - (n - 1) / 2) * bw
                    ax.bar(x, v, width=bw * 0.86, color=pal[s.idx % len(pal)], zorder=2)
                    note = bar_note(series, s, v, base)
                    vsize = big + 3 if n <= 3 else big - 1
                    off = 4
                    if note:
                        ax.annotate(note, (x, v), xytext=(0, off), textcoords="offset points", ha="center",
                                    va="bottom", fontsize=vsize - 6, color=T["muted"], fontweight="bold")
                        off += (vsize - 6) * 1.3
                    ax.annotate(fmt(v), (x, v), xytext=(0, off), textcoords="offset points", ha="center",
                                va="bottom", fontsize=vsize, fontweight="bold", color=T["text"])
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g["label"] for g in groups], fontsize=big - 2, color=T["text"])
            ax.set_ylim(0, vmax * (1.55 if len(series) > 1 else 1.3))
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        else:
            sp = pnl["spec"]
            for ln in sp["lines"]:
                i = ln["series"].idx % len(pal)
                pts = sorted(ln["points"])
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=pal[i], linewidth=2.6,
                        marker=MPL_MARKERS[MARKERS[i]], markersize=5.5, zorder=3)
            mpl_axes_x(ax, sp)
            if sp.get("ylog"):
                from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(0, y_top(p[1] for ln in sp["lines"] for p in ln["points"]))
                from matplotlib.ticker import FuncFormatter, MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"))
            ax.grid(True, axis="y", color=T["rule"], linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_xlabel(sp["xlabel"], fontsize=11 if kind == "wide" else 12.5, color=T["muted"])
        better = "higher is better" if pnl["kind"] == "bars" else f"{pnl['spec']['better']} is better"
        ax.set_title(f"{pnl['title']}", loc="left", fontsize=big, fontweight="bold", color=T["text"], pad=24)
        ax.text(0, 1.02, f"{pnl['unit']}, {better}", transform=ax.transAxes, fontsize=big - 5,
                color=T["muted"], va="bottom")
    fsize = 10.5 if kind == "wide" else 12
    foot = textwrap.wrap(text["footnote"], 100 if kind == "wide" else 78)
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
    series = load_series(args.results)
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
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="capyctl_bench.py", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="measure an endpoint and write a results JSON file")
    r.add_argument("--endpoint", default="http://127.0.0.1:8443/v1", help="OpenAI-compatible base URL")
    r.add_argument("--api-key-file", help="CapyCTL credentials file (api_key line); else $CAPYCTL_API_KEY")
    r.add_argument("--model", required=True, help="model name the endpoint serves (the deployment name)")
    r.add_argument("--label", required=True, help='series name in reports, e.g. "TensorFold 0.6.2"')
    r.add_argument("--out", default="results.json", help="results file to write")
    r.add_argument("--context-sweep", type=parse_sweep, help="prompt sizes, e.g. 0.5k,1k,2k,4k (k = 1000 tokens)")
    r.add_argument("--runs", type=int, default=3, help="requests per context point (default 3)")
    r.add_argument("--max-tokens", type=int, default=128, help="max_tokens for the context sweep (default 128)")
    r.add_argument("--concurrency", type=parse_levels, help="stream counts, e.g. 1-8 or 1,2,4,8")
    r.add_argument("--rounds", type=int, default=5, help="rounds per concurrency level (default 5)")
    r.add_argument("--concurrency-max-tokens", type=int, default=512,
                   help="max_tokens for concurrency rounds (default 512)")
    r.add_argument("--prompts", default=str(DEFAULT_PROMPTS), help="prompt set for concurrency (default prompts.json)")
    r.add_argument("--warmup", type=int, default=1, help="unrecorded warm-up requests or rounds (default 1)")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--extra-body", help='JSON object merged into every request, e.g. \'{"ignore_eos": true}\'')
    r.add_argument("--memory-cmd", help="shell command printing used memory in GiB; sampled during the run")
    r.add_argument("--memory-interval", type=float, default=0.5, help="seconds between memory samples")
    r.add_argument("--record-chunks", action="store_true", help="keep every token chunk's arrival time")
    r.add_argument("--meta", action="append", metavar="KEY=VALUE",
                   help="run metadata, e.g. capyctl=0.2.0 engine=vllm-0.30.0 gpu=GB10 (repeatable)")
    r.add_argument("--timeout", type=float, default=900.0, help="socket timeout per request, seconds")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="render results files into one report")
    rp.add_argument("results", nargs="+", help="results JSON files; the first is the baseline")
    rp.add_argument("--out", required=True, help="output directory")
    rp.add_argument("--title", help="report title (default: the labels)")
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
