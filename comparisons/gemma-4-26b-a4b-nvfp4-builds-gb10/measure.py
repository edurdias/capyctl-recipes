#!/usr/bin/env python3
"""Reuse capyctl-bench streaming/timing, plus greedy and tool-call guards.

Run inside an existing environment. The endpoint may be CapyCTL or a direct engine.
No engine credentials or authorization headers are written to result files.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import urllib.request


def post(endpoint, path, body, key):
    req = urllib.request.Request(endpoint.rstrip('/') + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.load(response)


def capture_guard(endpoint, model, key, prompts):
    def one(prompt):
        data = post(endpoint, "/chat/completions", {
            "model": model, "messages": prompt["messages"], "temperature": 0,
            "max_tokens": 256, "seed": 42}, key)
        choice = data["choices"][0]
        message = choice["message"]
        return {"id": prompt["id"], "content": message.get("content"),
                "reasoning": message.get("reasoning_content", message.get("reasoning")),
                "finish_reason": choice.get("finish_reason")}
    captures = {}
    for concurrency in (1, 8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            captures[str(concurrency)] = list(pool.map(one, prompts))
    tools = []
    for city in ("Paris", "Tokyo", "Lisbon"):
        data = post(endpoint, "/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": f"Call get_weather for {city} in celsius. Do not answer without calling the tool."}],
            "temperature": 0, "max_tokens": 512,
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Look up weather for a city.",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city", "unit"]}}}],
            "tool_choice": "auto"}, key)
        choice = data["choices"][0]
        calls = choice["message"].get("tool_calls", [])
        ok = False
        if len(calls) == 1:
            function = calls[0].get("function", {})
            try:
                arguments = json.loads(function.get("arguments", ""))
                ok = function.get("name") == "get_weather" and arguments == {"city": city, "unit": "celsius"}
            except (ValueError, TypeError):
                pass
        tools.append({"city": city, "passed": ok, "tool_calls": calls, "finish_reason": choice.get("finish_reason")})
    return {"greedy": captures, "tools": tools, "tools_passed": all(x["passed"] for x in tools)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="OpenAI URL ending in /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--recipes", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline", type=Path, help="measure.json for same checkpoint, engine and non-draft flags")
    parser.add_argument("--quality-verdict", type=Path, help="same-config native accuracy verdict, validated by run-host.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--guard-only", action="store_true")
    parser.add_argument("--public-mtp-screen", action="store_true", help="owner-authorized public-build screening, not native-quality approval")
    parser.add_argument("--context-prompts", type=Path, help="make-context.py output, prepared before engine launch")
    args = parser.parse_args()
    if not args.guard_only and not args.context_prompts:
        parser.error("--context-prompts is required for exact 24k measurement")
    benchmark = args.recipes / "tools/capyctl-bench/capyctl_bench.py"
    spec = importlib.util.spec_from_file_location("capyctl_bench", benchmark)
    cb = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cb
    spec.loader.exec_module(cb)
    key = cb.read_api_key(str(args.api_key_file) if args.api_key_file else None)
    prompt_file = args.recipes / "tools/capyctl-bench/prompts.json"
    prompts = cb.load_prompts(prompt_file)
    result = {"method": "capyctl-bench timings; Qwen comparison aggregate=sum(tokens)/round wall time",
              "prompt_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
              "temperature": 0, "guard_max_tokens": 256}
    guard = capture_guard(args.endpoint, args.model, key, prompts)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        if baseline["prompt_sha256"] != result["prompt_sha256"]:
            raise SystemExit("guard baseline uses different prompts")
        guard["matches_baseline"] = guard["greedy"] == baseline["guard"]["greedy"]
        guard["mismatches"] = [{"concurrency": c, "prompt": actual["id"]}
            for c, rows in guard["greedy"].items()
            for actual, expected in zip(rows, baseline["guard"]["greedy"][c]) if actual != expected]
    else:
        guard["matches_baseline"] = None
    result["guard"] = guard
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    quality = json.loads(args.quality_verdict.read_text()) if args.quality_verdict else None
    result["quality"] = quality
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if not guard["tools_passed"]:
        raise SystemExit("three-tool correctness guard failed; refusing speed qualification")
    if guard["matches_baseline"] is False and not args.guard_only and not args.public_mtp_screen and not (quality and quality.get("approved")):
        raise SystemExit("greedy numerical variation recorded; native quality approval required before speed qualification")
    if not args.guard_only:
        bench_args = argparse.Namespace(model=args.model, temperature=0.0, extra_body=None, record_chunks=False)
        runner = cb.Runner(bench_args, cb.Endpoint(args.endpoint, key, timeout=1800), None)
        result["concurrency"] = []
        offset = 0
        for c in map(int, args.concurrency.split(',')):
            runner.round_(c, prompts, offset, 512)  # unrecorded warm-up
            offset += c
            rounds = []
            for _ in range(args.rounds):
                rounds.append(runner.round_(c, prompts, offset, 512))
                offset += c
            if any(q.get("error") for r in rounds for q in r["requests"]):
                raise SystemExit("speed request failed; see engine log")
            aggregates = []
            for r in rounds:
                reqs = r["requests"]
                span = max(q["sent_at"] + q["total_s"] for q in reqs) - min(q["sent_at"] for q in reqs)
                aggregates.append(sum(q["completion_tokens"] for q in reqs) / span)
            result["concurrency"].append({"streams": c, "aggregate_tps": statistics.median(aggregates),
                "decode_tps": statistics.median(q["decode_tps"] for r in rounds for q in r["requests"]), "rounds": rounds})
            args.out.write_text(json.dumps(result, indent=2) + "\n")
        contexts = json.loads(args.context_prompts.read_text())
        result["ttft24k"] = {"target_prompt_tokens": 24000, "levels": []}
        for c in (1, 4):
            rounds = []
            for r in range(args.rounds):
                rounds.append(runner.round_(c, contexts["levels"][str(c)][r], 0, 128))
            requests = [q for r in rounds for q in r["requests"]]
            valid = all(not q.get("error") and q.get("prompt_tokens") == 24000 for q in requests)
            result["ttft24k"]["levels"].append({"streams": c, "valid": valid,
                "ttft_s": statistics.median(q["ttft_s"] for q in requests) if valid else None,
                "actual_prompt_tokens": [q.get("prompt_tokens") for q in requests], "rounds": rounds})
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            if not valid:
                raise SystemExit("24k request failed or actual prompt length is not exactly 24000; evidence saved")
    print(json.dumps({"out": str(args.out), "guard_passed": guard["tools_passed"] and guard["matches_baseline"] is not False}))


if __name__ == "__main__":
    main()
