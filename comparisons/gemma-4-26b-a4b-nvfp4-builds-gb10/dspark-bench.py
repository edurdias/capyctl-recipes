#!/usr/bin/env python3
"""Reproduce the pinned external DSpark single-probe and concurrency workloads."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time
from urllib.parse import urlsplit


SOURCE_COMMIT = "4cdf39e97328887f591af0a6113499d86b3cd338"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_inputs(source_repo):
    raw = (source_repo / "scripts/bench.py").read_bytes()
    tree = ast.parse(raw)
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "PROBES"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("source must define exactly one literal PROBES mapping")
    probes = ast.literal_eval(assignments[0].value)
    if not isinstance(probes, dict) or set(probes) != {"code", "essay", "chat"}:
        raise ValueError("unexpected source probe names")
    for name, cap in (("code", 700), ("essay", 700), ("chat", 300)):
        probe = probes[name]
        if (not isinstance(probe, dict) or not isinstance(probe.get("prompt"), str)
                or not probe["prompt"] or probe.get("max_tokens") != cap):
            raise ValueError("unexpected source probe definition")
    warmups = {node.args[1].value for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "stream_chat" and len(node.args) >= 3
               and isinstance(node.args[1], ast.Constant)
               and isinstance(node.args[1].value, str)
               and isinstance(node.args[2], ast.Constant) and node.args[2].value == 32}
    if len(warmups) != 1:
        raise ValueError("source must contain one unambiguous 32-token warmup prompt")
    return probes, warmups.pop(), hashlib.sha256(raw).hexdigest()


def validate(rec):
    usage = rec.get("usage") or {}
    if rec.get("error") or rec.get("http_status") != 200:
        raise ValueError("request failed; captured evidence saved")
    if (rec.get("usage_missing") or any(type(usage.get(key)) is not int or usage[key] <= 0
                                      for key in ("prompt_tokens", "completion_tokens"))):
        raise ValueError("positive reported token usage required; captured evidence saved")
    if usage["completion_tokens"] > rec["max_tokens"]:
        raise ValueError("reported completion exceeds request cap; captured evidence saved")
    for key in ("decode_tps", "publication_net_decode_tps"):
        value = rec.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("positive measured decode rates required; captured evidence saved")


def run(args, cb, helper, probes, warmup, digest):
    endpoint = cb.Endpoint(args.endpoint, timeout=900)
    settings = {"temperature": 0.7, "top_p": 0.8, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
    result = {
        "status": "running",
        "source": {"expected_commit": SOURCE_COMMIT, "file": "scripts/bench.py",
                   "sha256": digest, "prompts": probes, "warmup_prompt": warmup},
        "settings": {**settings, "probe_order": ["code", "essay", "chat"],
                     "repeats_per_probe": 3, "initial_warmups": 3,
                     "warmup_max_tokens": 32, "ladder": [1, 2, 4, 8]},
        "published_reference": {
            "single_probe_tps": {"DSpark": {"code": 159.18, "essay": 65.86, "chat": 93.26},
                                 "MTP": {"code": 98.18, "essay": 68.26, "chat": 80.91}},
            "c8_wall_aggregate_tps": 293.2,
            "ladder_provenance_limitation": "README calls the ladder DSpark, but the committed ladder branch launches assistant depth 1; drafter provenance remains unresolved.",
            "source_vllm_version": "0.26.1rc1.dev1177+ga9a17e709",
            "comparison_vllm_version": "0.32.0",
            "version_limitation": "Source and existing comparison use different vLLM versions; these are reference labels, not discovered server metadata.",
            "measurement_labels": "Actual drafter and speculative depth are supplied by the parent launcher, not inferred here."},
        "definitions": {
            "decode_tps": "shared timer: (reported completion_tokens - 1) / (last text time - first text time)",
            "publication_net_decode_tps": "reported completion_tokens / elapsed first SSE choices to after stream close",
            "publication_median_decode_tps": "median of per-request publication rates rounded to two decimal places before median",
            "shared_median_decode_tps": "median of unrounded shared decode rates",
            "publication_mean_decode_tps": "mean of publication rates rounded to two decimals before averaging",
            "aggregate_tps": "sum reported completion_tokens / whole batch perf_counter wall time; no barrier; evidence serialized after the timed batch",
            "acceptance": "label-free cumulative native speculative counter deltas bracketing each recorded probe or batch"},
        "warmups": [], "probes": {}, "ladder_warmups": [],
        "ladder_standalone": {"requests": []}, "ladder": [],
    }

    def save():
        helper.save(args.out, result)

    def request(prompt, cap):
        payload = {**settings, "model": args.model,
                   "messages": [{"role": "user", "content": prompt}], "max_tokens": cap}
        rec = cb.stream_chat(endpoint, payload, record_chunks=True)
        # Never retain a rate derived from the shared timer's missing-usage fallback.
        if rec.get("usage_missing"):
            for key in ("decode_tps", "publication_net_decode_tps", "tpot_ms"):
                rec[key] = None
        rate = rec.get("publication_net_decode_tps")
        rec["publication_rounded_decode_tps"] = round(rate, 2) if rate is not None else None
        rec["request_settings"] = settings
        return rec

    def serial(target, prompt, cap, count):
        for index in range(count):
            rec = request(prompt, cap)
            rec["run"] = index + 1
            target.append(rec)
            save()
            validate(rec)

    def bracket(section, action):
        before = helper.metrics_snapshot(endpoint)
        try:
            action()
        finally:
            section["acceptance"] = helper.acceptance_span(before, helper.metrics_snapshot(endpoint))
            save()

    save()
    try:
        serial(result["warmups"], warmup, 32, 3)
        for name in ("code", "essay", "chat"):
            probe = probes[name]
            section = {"requests": [], "status": "running"}
            result["probes"][name] = section
            bracket(section, lambda: serial(section["requests"], probe["prompt"], probe["max_tokens"], 3))
            section["shared_median_decode_tps"] = statistics.median(r["decode_tps"] for r in section["requests"])
            section["publication_median_decode_tps"] = statistics.median(r["publication_rounded_decode_tps"] for r in section["requests"])
            section["status"] = "complete"
            save()
        serial(result["ladder_warmups"], warmup, 32, 1)
        standalone = result["ladder_standalone"]
        bracket(standalone, lambda: serial(standalone["requests"], probes["essay"]["prompt"], 700, 1))
        for streams in (1, 2, 4, 8):
            section = {"streams": streams, "status": "running", "requests": []}
            result["ladder"].append(section)

            def batch():
                # Match source placement: executor created before starting the wall timer.
                with ThreadPoolExecutor(max_workers=streams) as executor:
                    start = time.perf_counter()
                    futures = {executor.submit(request, probes["essay"]["prompt"], 700): i + 1
                               for i in range(streams)}
                    for future in as_completed(futures):
                        rec = future.result()
                        rec["run"] = futures[future]
                        section["requests"].append(rec)
                section["wall_s"] = time.perf_counter() - start
                for rec in section["requests"]:
                    validate(rec)
                records = section["requests"]
                section["aggregate_tps"] = sum(r["usage"]["completion_tokens"] for r in records) / section["wall_s"]
                section["shared_mean_decode_tps"] = statistics.mean(r["decode_tps"] for r in records)
                section["publication_mean_decode_tps"] = statistics.mean(r["publication_rounded_decode_tps"] for r in records)
                section["publication_rounded_mean_decode_tps"] = round(section["publication_mean_decode_tps"], 1)
                section["publication_rounded_aggregate_tps"] = round(section["aggregate_tps"], 1)
                section["status"] = "complete"

            bracket(section, batch)
        result["status"] = "complete"
    except Exception:
        result["status"] = "failed"
        result["error"] = "benchmark failed; inspect saved request evidence"
        save()
        raise SystemExit("DSpark benchmark failed; partial evidence saved") from None
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    url = urlsplit(args.endpoint)
    if (url.scheme not in ("http", "https") or not url.hostname or not url.path.endswith("/v1")
            or url.query or url.fragment or url.username or url.password):
        parser.error("--endpoint must be HTTP(S), end in /v1, and omit credentials, query and fragment")
    try:
        probes, warmup, digest = source_inputs(args.source_repo)
        cb = load_module("dspark_shared_timer", args.recipes / "tools/capyctl-bench/capyctl_bench.py")
        helper = load_module("dspark_published_helpers", Path(__file__).with_name("published-bench.py"))
    except Exception:
        raise SystemExit("cannot load benchmark helpers or exact literal source inputs") from None
    helper.install_output_capture(cb)
    run(args, cb, helper, probes, warmup, digest)
    print(json.dumps({"status": "complete"}))


if __name__ == "__main__":
    main()
