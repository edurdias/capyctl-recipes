#!/usr/bin/env python3
"""Measure three serial decode requests each at exactly 64,000 and 110,000 input tokens.

Uses only the local CPU tokenizer. The caller selects and labels engine depth.
No warmups, retries, quality evaluations, or engine configuration changes are made.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import time
import urllib.parse
import uuid


def load_benchmark():
    path = Path(__file__).resolve().parents[2] / "tools/capyctl-bench/capyctl_bench.py"
    spec = importlib.util.spec_from_file_location("capyctl_bench", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare(tokenizer, cb, target, seed):
    def count(messages):
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids["input_ids"] if hasattr(ids, "keys") else ids)

    low, high = 0, 4096
    while count(cb.context_messages(high, seed)) < target:
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if count(cb.context_messages(middle, seed)) <= target:
            low = middle
        else:
            high = middle
    messages = cb.context_messages(low, seed)
    remaining = target - count(messages)
    document, request = messages[0]["content"].rsplit("\n\n", 1)
    messages[0]["content"] = document + " a" * remaining + "\n\n" + request
    actual = count(messages)
    if actual != target:
        raise ValueError("Exact chat-template padding failed")
    return messages, actual


class CapturingResponse:
    """Capture generated text without retaining server metadata or headers."""
    def __init__(self, response, output):
        self.response = response
        self.output = output
        self.status = response.status

    def read(self, *args):
        # The shared timer reads error bodies. Do not retain or expose them.
        self.response.read(*args)
        return b""

    def readline(self):
        line = self.response.readline()
        if line.strip().startswith(b"data:"):
            try:
                obj = json.loads(line.strip()[5:].strip())
            except ValueError:
                return line
            if isinstance(obj, dict):
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    fields = {k: delta[k] for k in ("content", "reasoning_content", "reasoning")
                              if isinstance(delta.get(k), str)}
                    if fields:
                        self.output.append(fields)
        return line


class CapturingConnection:
    def __init__(self, connection, output):
        self.connection = connection
        self.output = output

    def request(self, *args, **kwargs):
        return self.connection.request(*args, **kwargs)

    def getresponse(self):
        return CapturingResponse(self.connection.getresponse(), self.output)

    def close(self):
        self.connection.close()


def summarize(rows):
    valid = [row for row in rows if row["passed"]]
    metrics = ("prompt_tokens", "completion_tokens", "ttft_s", "decode_tps", "total_s")
    return {
        "completed_requests": len(rows),
        "passed_requests": len(valid),
        "passed": len(rows) == 3 and len(valid) == 3,
        "median_scope": "Passed measurements only; failures are excluded, not replaced",
        "medians": {name: statistics.median(row["timing"][name] for row in valid)
                    if valid else None for name in metrics},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="Root URL or URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.endpoint)
    if url.scheme not in ("http", "https") or not url.hostname:
        parser.error("Endpoint must be an HTTP or HTTPS URL")
    endpoint = args.endpoint.rstrip("/")
    if not url.path.rstrip("/").endswith("/v1"):
        endpoint += "/v1"
    cb = load_benchmark()
    key = cb.read_api_key(str(args.api_key_file) if args.api_key_file else None)
    ep = cb.Endpoint(endpoint, key, timeout=1800)
    output = []
    connect = ep.connect
    ep.connect = lambda: CapturingConnection(connect(), output)
    runner = cb.Runner(argparse.Namespace(model=args.model, temperature=0.0,
                                         extra_body=None, record_chunks=True), ep, None)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    result = {
        "method": "Shared streaming timer; serial requests; local chat-template token counts",
        "concurrency": 1,
        "requests_per_target": 3,
        "max_output_tokens": 512,
        "decode_tps_definition": "(completion_tokens - 1) / (last text chunk time - first text chunk time)",
        "pass_conditions": ["No request or preparation error", "Reported prompt tokens equal target",
                            "Reported completion usage is present", "Positive decode throughput and output tokens",
                            "TTFT and total time are present"],
        "output_policy": "Generated text deltas only; server metadata and error bodies omitted; endpoint and key redacted",
        "targets": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def clean(value):
        if isinstance(value, str):
            for secret in (key, endpoint, args.endpoint, url.hostname):
                if secret:
                    value = value.replace(secret, "[redacted]")
            value = re.sub(r"https?://[^\s\"<>]+", "[redacted URL]", value)
            value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[redacted address]", value)
            return value
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {name: clean(item) for name, item in value.items()}
        return value

    def save():
        args.out.write_text(json.dumps(clean(result), ensure_ascii=False, indent=2) + "\n")

    save()
    for target in (64000, 110000):
        group = {"target_prompt_tokens": target, "requests": [], "summary": summarize([])}
        result["targets"].append(group)
        for index in range(3):
            # A fresh UUID also appears near the beginning of the document through
            # context_messages' integer seed, even across separate invocations.
            seed = uuid.uuid4().int
            row = {"request": index + 1, "seed": str(seed), "expected_prompt_tokens": target,
                   "input": None, "local_prompt_tokens": None, "timing": None,
                   "output_deltas": [], "errors": [], "passed": False}
            output.clear()
            started = time.perf_counter()
            try:
                messages, actual = prepare(tokenizer, cb, target, seed)
                row["local_prompt_tokens"] = actual
                row["input"] = runner.payload(messages, 512)
                timing = cb.stream_chat(ep, row["input"], record_chunks=True)
                # Do not store arbitrary server extras or error strings: those can
                # contain downstream platform names, addresses, or credentials.
                error = timing.pop("error", None)
                timing.pop("extras", None)
                if error:
                    status = re.match(r"HTTP (\d+)", error)
                    row["errors"].append("HTTP " + status.group(1) if status else
                                         "Streaming request failed; transport or stream incomplete")
                row["timing"] = timing
                row["output_deltas"] = list(output)
                if timing.get("prompt_tokens") != target:
                    row["errors"].append("Reported prompt token count does not equal target")
                if timing.get("usage_missing"):
                    row["errors"].append("Completion token usage is missing; chunk counts are not token counts")
                if not timing.get("decode_tps") or timing.get("completion_tokens", 0) <= 1:
                    row["errors"].append("Decode throughput is unavailable")
                if timing.get("ttft_s") is None or timing.get("total_s") is None:
                    row["errors"].append("Required timing is missing")
                row["passed"] = not row["errors"]
            except Exception as exc:
                row["output_deltas"] = list(output)
                row["errors"].append("Probe failed: " + type(exc).__name__)
                row["elapsed_until_failure_s"] = time.perf_counter() - started
            group["requests"].append(row)
            group["summary"] = summarize(group["requests"])
            save()
    passed = all(group["summary"]["passed"] for group in result["targets"])
    print(json.dumps({"passed": passed, "recorded_requests": 6}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
