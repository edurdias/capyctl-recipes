#!/usr/bin/env python3
"""Measure the published 4096-context recipe using shared capyctl-bench timings.

No engine launches, retries, authentication, or background traffic. Output contains
prompts, generated deltas, numeric usage, timings, and label-free metric snapshots;
it never contains the endpoint, response metadata, or HTTP error bodies.
"""
import argparse
import hashlib
import http.client
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlsplit


MAX_TOKENS = 512
CONTEXT_LIMIT = 4096
PRIME_PROMPT = "List the first 5 prime numbers and explain in 2 sentences why each one is prime."
METRIC_LINE = re.compile(
    r'^(vllm:spec_decode_num_[A-Za-z0-9_]+)(\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$'
)
POSITION_LABEL = re.compile(r'(?:^|[,{])\s*(?:position|pos|token_position)="(\d+)"')


def numeric_usage(value):
    """Keep usage counters/details, never arbitrary string-valued metadata."""
    if isinstance(value, dict):
        return {key: numeric_usage(item) for key, item in value.items()
                if isinstance(item, dict) or type(item) in (int, float)}
    return value


class CapturedResponse:
    def __init__(self, response):
        self.response = response
        self.deltas = []
        self.usage = None
        self.http_status = response.status
        self.first_choices_monotonic = None

    def __getattr__(self, name):
        return getattr(self.response, name)

    def readline(self, *args):
        line = self.response.readline(*args)
        if line.strip().startswith(b"data:"):
            try:
                obj = json.loads(line.strip()[5:].strip())
            except (ValueError, UnicodeError):
                return line
            if isinstance(obj, dict):
                if obj.get("choices") and self.first_choices_monotonic is None:
                    self.first_choices_monotonic = time.perf_counter()
                if isinstance(obj.get("usage"), dict):
                    self.usage = numeric_usage(obj["usage"])
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta")
                    if isinstance(delta, dict):
                        # These are generated output, not envelope/model identifiers.
                        self.deltas.append({key: value for key, value in delta.items()
                                            if key in ("content", "reasoning", "reasoning_content", "tool_calls")})
        return line


class CapturedConnection:
    def __init__(self, connection, endpoint):
        self.connection = connection
        self.endpoint = endpoint

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def getresponse(self):
        response = CapturedResponse(self.connection.getresponse())
        self.endpoint.response = response
        return response


class CapturedEndpoint:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.response = None

    def __getattr__(self, name):
        return getattr(self.endpoint, name)

    def connect(self):
        return CapturedConnection(self.endpoint.connect(), self)


def install_output_capture(cb):
    """Wrap this private module instance; Runner's timing and barriers stay intact."""
    original = cb.stream_chat

    def stream_chat(endpoint, payload, record_chunks=False):
        captured = CapturedEndpoint(endpoint)
        rec = original(captured, payload, record_chunks)
        completed = time.perf_counter()
        response = captured.response
        rec.pop("extras", None)
        rec["http_status"] = response.http_status if response else None
        rec["usage"] = response.usage if response else None
        rec["output_deltas"] = response.deltas if response else []
        rec["output"] = "".join(cb._delta_text(delta) for delta in rec["output_deltas"])
        first_choices = response.first_choices_monotonic if response else None
        rec["publication_decode_s"] = completed - first_choices if first_choices is not None else None
        rec["publication_net_decode_tps"] = (rec["completion_tokens"] / rec["publication_decode_s"]
                                            if rec["publication_decode_s"] else None)
        rec["messages"] = payload["messages"]
        rec["max_tokens"] = payload["max_tokens"]
        rec["temperature"] = payload.get("temperature")
        if rec.get("error"):
            # The shared timer includes raw HTTP bodies / exception messages.
            # Keep the failure explicit without persisting remote identifiers.
            if rec["http_status"] not in (None, 200):
                rec["error"] = f"HTTP {rec['http_status']}"
            else:
                rec["error"] = "stream request failed or ended incompletely"
        return rec

    cb.stream_chat = stream_chat


def metrics_snapshot(endpoint):
    """GET native /metrics once; labels exist only in transient reset detection."""
    public = {"sampled_at": time.time(), "status": "unmeasured", "counters": {}, "error": None}
    series = {}
    connection = endpoint.connect()
    try:
        connection.request("GET", "/metrics", headers=endpoint.headers("text/plain"))
        response = connection.getresponse()
        if response.status != 200:
            public["error"] = f"HTTP {response.status}"
            return public, series
        text = response.read().decode("utf-8")
        for line in text.splitlines():
            match = METRIC_LINE.fullmatch(line.strip())
            if not match:
                continue
            name, labels, raw_value = match.groups()
            if name.endswith("_created"):
                continue
            value = float(raw_value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid spec_decode counter")
            position = POSITION_LABEL.search(labels or "")
            key = name + (f"_position_{int(position[1])}" if position else "")
            identity = (key, labels or "")
            if identity in series:
                raise ValueError("duplicate spec_decode series")
            series[identity] = value
            public["counters"][key] = public["counters"].get(key, 0) + value
        if public["counters"]:
            public["status"] = "measured"
        else:
            public["error"] = "spec_decode counters absent"
    except (OSError, http.client.HTTPException, UnicodeError, ValueError) as exc:
        public["counters"] = {}
        series = {}
        public["error"] = type(exc).__name__ + " while reading spec_decode counters"
    finally:
        connection.close()
    return public, series


def acceptance_span(before, after):
    first, first_series = before
    last, last_series = after
    result = {"before": first, "after": last, "status": "unmeasured", "deltas": {},
              "counter_resets": [], "series_changed": [], "errors": [],
              "mean_accepted_length": None, "token_acceptance": None}
    if first["status"] != "measured" or last["status"] != "measured":
        result["errors"] = [snapshot["error"] for snapshot in (first, last) if snapshot["error"]]
        return result
    for key in sorted(first["counters"].keys() | last["counters"].keys()):
        before_keys = {identity for identity in first_series if identity[0] == key}
        after_keys = {identity for identity in last_series if identity[0] == key}
        reset = any(last_series[identity] < first_series[identity]
                    for identity in before_keys & after_keys)
        changed = before_keys != after_keys
        if reset:
            result["counter_resets"].append(key)
        if changed:
            result["series_changed"].append(key)
        result["deltas"][key] = (None if reset or changed else
                                    last["counters"][key] - first["counters"][key])
    result["status"] = "partial" if result["counter_resets"] or result["series_changed"] else "measured"

    def counter(stem):
        # Per-position series must not be double-counted into global counters.
        names = [name for name in ("vllm:spec_decode_" + stem, "vllm:spec_decode_" + stem + "_total")
                 if name in result["deltas"]]
        return result["deltas"][names[0]] if len(names) == 1 else None

    drafts = counter("num_drafts")
    drafted = counter("num_draft_tokens")
    accepted = counter("num_accepted_tokens")
    if accepted is not None and drafts is not None and drafts > 0:
        result["mean_accepted_length"] = 1 + accepted / drafts
    if accepted is not None and drafted is not None and drafted > 0:
        result["token_acceptance"] = accepted / drafted
    if result["mean_accepted_length"] is None or result["token_acceptance"] is None:
        result["errors"].append("required counters missing, reset, changed, ambiguous, or denominator zero")
    return result


def validate_requests(requests):
    for rec in requests:
        usage = rec.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if rec.get("error"):
            raise ValueError("request failed; captured evidence saved")
        if (type(prompt) is not int or prompt <= 0 or type(completion) is not int or completion <= 0
                or rec.get("usage_missing")):
            raise ValueError("missing or invalid reported token usage; captured evidence saved")
        if prompt + MAX_TOKENS > CONTEXT_LIMIT:
            raise ValueError("reported prompt tokens + 512 exceed 4096; captured evidence saved")
        if completion > MAX_TOKENS:
            raise ValueError("reported completion exceeds 512-token cap; captured evidence saved")
        decode = rec.get("decode_tps")
        if not isinstance(decode, (int, float)) or not math.isfinite(decode) or decode <= 0:
            raise ValueError("per-request decode rate is unmeasured; captured evidence saved")
        rec["context_verified"] = True


def summarize_round(rnd, concurrency):
    requests = rnd["requests"]
    validate_requests(requests)
    wall = max(rec["sent_at"] + rec["total_s"] for rec in requests) - min(rec["sent_at"] for rec in requests)
    if wall <= 0:
        raise ValueError("nonpositive request wall span; captured evidence saved")
    rnd["wall_s"] = wall
    rnd["completion_tokens"] = sum(rec["usage"]["completion_tokens"] for rec in requests)
    rnd["aggregate_tps"] = rnd["completion_tokens"] / wall
    rnd["mean_decode_tps"] = statistics.mean(rec["decode_tps"] for rec in requests)
    rnd["published_mean_times_concurrency_tps"] = rnd["mean_decode_tps"] * concurrency


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def run(args, cb, prompts, prompt_digest):
    endpoint = cb.Endpoint(args.endpoint, timeout=1800)
    runner = cb.Runner(argparse.Namespace(model=args.model, temperature=0.0, extra_body=None,
                                          record_chunks=True), endpoint, None)
    result = {
        "status": "running", "prompt_sha256": prompt_digest, "prompts": prompts,
        "settings": {"concurrency": [1, 2, 4, 8], "recorded_rounds": 3,
                     "unrecorded_warmup_rounds_per_level": 1, "max_tokens": MAX_TOKENS,
                     "temperature": 0, "context_limit": CONTEXT_LIMIT, "prime_control_runs": 5},
        "definitions": {
            "decode_tps": "shared capyctl-bench: (reported completion_tokens - 1) / (last text delta time - first text delta time); content and reasoning included",
            "decode_tps_level": "median per-request decode_tps over all recorded requests at this concurrency",
            "aggregate_tps_round": "sum(reported completion_tokens) / (max(sent_at + total_s) - min(sent_at)) across all requests in one round",
            "aggregate_tps_level": "median of the three recorded round aggregate_tps values",
            "published_mean_times_concurrency_tps_round": "arithmetic mean of per-request decode_tps in this round times concurrency; NOT tokens/wall",
            "published_mean_times_concurrency_tps_level": "arithmetic mean of per-request decode_tps across all recorded requests at this level times concurrency; NOT tokens/wall",
            "prime_decode_stddev_tps": "population standard deviation (ddof=0) of the five serial per-request decode_tps values",
            "mean_accepted_length": "1 + delta(num_accepted_tokens) / delta(num_drafts); null for unavailable or zero denominator",
            "token_acceptance": "delta(num_accepted_tokens) / delta(num_draft_tokens); null for unavailable or zero denominator",
            "metrics": "native root /metrics snapshots after level warmup and after recorded rounds; short-control snapshots bracket only its five serial requests; series summed without labels, numeric position retained; reset/series-change deltas null",
            "prompt_selection": "same rolling modulo offset as measure.py, advanced by concurrency after warmup and every recorded round, across levels",
            "usage": "reported streaming usage only; missing usage fails rather than estimating tokens",
        },
        "concurrency": [],
    }
    save(args.out, result)
    offset = 0
    try:
        for concurrency in (1, 2, 4, 8):
            warmup = runner.round_(concurrency, prompts, offset, MAX_TOKENS)
            offset += concurrency
            try:
                validate_requests(warmup["requests"])
            except ValueError:
                result["failed_warmup"] = {"streams": concurrency, "requests": warmup["requests"]}
                raise
            level = {"streams": concurrency, "status": "running", "rounds": []}
            result["concurrency"].append(level)
            before = metrics_snapshot(endpoint)
            try:
                for index in range(3):
                    rnd = runner.round_(concurrency, prompts, offset, MAX_TOKENS)
                    offset += concurrency
                    rnd["round"] = index + 1
                    level["rounds"].append(rnd)
                    summarize_round(rnd, concurrency)
                requests = [rec for rnd in level["rounds"] for rec in rnd["requests"]]
                level["decode_tps"] = statistics.median(rec["decode_tps"] for rec in requests)
                level["aggregate_tps"] = statistics.median(rnd["aggregate_tps"] for rnd in level["rounds"])
                level["mean_decode_tps"] = statistics.mean(rec["decode_tps"] for rec in requests)
                level["published_mean_times_concurrency_tps"] = level["mean_decode_tps"] * concurrency
                level["status"] = "complete"
            finally:
                level["acceptance"] = acceptance_span(before, metrics_snapshot(endpoint))
                save(args.out, result)
        control = {"prompt": PRIME_PROMPT, "status": "running", "requests": []}
        result["prime_control"] = control
        before = metrics_snapshot(endpoint)
        try:
            messages = [{"role": "user", "content": PRIME_PROMPT}]
            for index in range(5):
                rec = runner.one(messages, MAX_TOKENS)
                rec["run"] = index + 1
                control["requests"].append(rec)
                validate_requests([rec])
            rates = [rec["decode_tps"] for rec in control["requests"]]
            control["mean_decode_tps"] = statistics.mean(rates)
            control["stddev_decode_tps"] = statistics.pstdev(rates)
            control["status"] = "complete"
        finally:
            control["acceptance"] = acceptance_span(before, metrics_snapshot(endpoint))
            save(args.out, result)
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        # Only our fixed validation text is safe to persist verbatim.
        result["error"] = str(exc) if type(exc) is ValueError else type(exc).__name__ + " during benchmark"
        save(args.out, result)
        raise SystemExit("published benchmark failed; partial evidence saved") from None
    save(args.out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--endpoint", required=True, help="OpenAI URL ending in /v1; never saved")
    parser.add_argument("--model", required=True, help="Serving model identifier; never saved")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.path.endswith("/v1")
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        parser.error("--endpoint must be an HTTP(S) URL ending in /v1 without credentials, query, or fragment")
    benchmark = args.recipes / "tools/capyctl-bench/capyctl_bench.py"
    spec = importlib.util.spec_from_file_location("capyctl_bench", benchmark)
    cb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cb
    spec.loader.exec_module(cb)
    install_output_capture(cb)
    prompt_file = args.recipes / "tools/capyctl-bench/prompts.json"
    prompts = cb.load_prompts(prompt_file)
    run(args, cb, prompts, hashlib.sha256(prompt_file.read_bytes()).hexdigest())
    print(json.dumps({"status": "complete", "out": str(args.out)}))


if __name__ == "__main__":
    main()
