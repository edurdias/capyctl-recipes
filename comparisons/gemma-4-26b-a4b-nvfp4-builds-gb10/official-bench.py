#!/usr/bin/env python3
"""Run the deterministic historical reconstruction of the official coding workload."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from urllib.parse import urlsplit


MAX_TOKENS = 8192
WARMUPS = 3
MEASURED = 10


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("benchmark module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate(rec):
    usage = rec.get("usage")
    if not isinstance(usage, dict) or not positive(usage.get("completion_tokens")) or rec.get("usage_missing"):
        raise ValueError("positive reported completion usage required")
    if rec.get("http_status") != 200 or rec.get("error"):
        raise ValueError("HTTP or stream failure")
    if not positive(rec.get("decode_tps")) or not positive(rec.get("decode_s")):
        raise ValueError("positive shared-timer decode rate required")
    if usage["completion_tokens"] > MAX_TOKENS:
        raise ValueError("reported completion usage exceeds output token cap")


def run(args):
    result = {
        "status": "running",
        "source_published_decode_tokens_per_second": 78.10,
        "reproduction": {
            "classification": "deterministic historical reconstruction, not an exact public-row claim",
            "limitation": "Exact public row selection and sampling method were not published.",
            "harness_difference": "stdlib shared capyctl-bench streaming timer, not AIPerf 0.13.0",
            "source_sampling": "unspecified",
        },
        "settings": {
            "warmup_requests": WARMUPS, "measured_requests": MEASURED,
            "concurrency": 1, "requests_per_row": 1, "turn_selection": "first user turn only",
            "max_tokens": MAX_TOKENS, "stream": True, "stream_options": {"include_usage": True},
            "sampling": "native model defaults; temperature, top_p, seed and chat_template_kwargs omitted",
        },
        "definitions": {
            "decode_tps": "(reported completion_tokens - 1) / (last text delta time - first text delta time), shared capyctl-bench timer",
            "decode_tokens_per_second_mean": "arithmetic mean of the ten measured per-request decode_tps values; warmups excluded",
            "decode_tokens_per_second_stddev": "population standard deviation (ddof=0) of the ten measured rates",
            "usage": "reported streaming usage only; no chunk-count speed estimates",
            "acceptance": "label-free cumulative speculative counter deltas bracketing only the ten measured requests",
        },
        "requests": [],
    }
    save(args.out, result)
    try:
        raw = args.dataset.read_bytes()
        dataset = json.loads(raw)
        result["dataset_selection"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "stored_dataset": dataset,
            "warmup_stored_positions": list(range(WARMUPS)),
            "measured_stored_positions": list(range(WARMUPS, WARMUPS + MEASURED)),
        }
        save(args.out, result)
        rows = dataset.get("rows") if isinstance(dataset, dict) else None
        if not isinstance(rows, list) or len(rows) != WARMUPS + MEASURED:
            raise ValueError("dataset must contain exactly thirteen stored rows")
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("row_index")) is not int
                    or not isinstance(row.get("src_id"), str) or row.get("category") != "coding"
                    or not isinstance(row.get("turns"), list) or not row["turns"]
                    or not isinstance(row["turns"][0], str) or not row["turns"][0]):
                raise ValueError("each stored coding row requires row_index, src_id and first user turn")
        cb = load_module("official_capyctl_bench", args.recipes / "tools/capyctl-bench/capyctl_bench.py")
        helpers = load_module("official_published_bench", Path(__file__).resolve().with_name("published-bench.py"))
        helpers.install_output_capture(cb)
        endpoint = cb.Endpoint(args.endpoint)

        def request(position):
            row = rows[position]
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": row["turns"][0]}],
                "max_tokens": MAX_TOKENS, "stream": True,
                "stream_options": {"include_usage": True},
            }
            rec = cb.stream_chat(endpoint, payload, record_chunks=True)
            rec.update({"stored_position": position, "row_index": row["row_index"],
                        "src_id": row["src_id"], "phase": "warmup" if position < WARMUPS else "measured"})
            # Shared timer has a chunk-count fallback: never persist its estimated rates as evidence.
            if rec.get("usage_missing"):
                rec["completion_tokens"] = None
                for key in ("decode_tps", "tpot_ms", "publication_net_decode_tps"):
                    rec[key] = None
            result["requests"].append(rec)
            save(args.out, result)
            validate(rec)

        for position in range(WARMUPS):
            request(position)
        before = helpers.metrics_snapshot(endpoint)
        result["acceptance"] = {"status": "running", "before": before[0]}
        save(args.out, result)
        try:
            for position in range(WARMUPS, WARMUPS + MEASURED):
                request(position)
        finally:
            result["acceptance"] = helpers.acceptance_span(before, helpers.metrics_snapshot(endpoint))
            save(args.out, result)
        rates = [rec["decode_tps"] for rec in result["requests"] if rec["phase"] == "measured"]
        if len(rates) != MEASURED:
            raise ValueError("all ten measured rows are required")
        result.update({
            "decode_tokens_per_second_mean": statistics.mean(rates),
            "decode_tokens_per_second_stddev": statistics.pstdev(rates),
            "decode_tokens_per_second_min": min(rates),
            "decode_tokens_per_second_max": max(rates),
            "decode_tokens_per_second_per_request": rates,
            "status": "complete",
        })
    except BaseException as exc:
        result["status"] = "failed"
        # Never persist exception messages: network/import/file errors can expose private paths or hosts.
        result["error"] = type(exc).__name__ + " during official benchmark"
        save(args.out, result)
        raise SystemExit("official benchmark failed; partial evidence saved") from None
    save(args.out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipes", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--endpoint", required=True, help="OpenAI HTTP(S) URL ending in /v1; never saved")
    parser.add_argument("--model", required=True, help="Serving identifier; never saved")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.endpoint)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.path.endswith("/v1")
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        parser.error("--endpoint must be HTTP(S), end in /v1, and contain no credentials, query or fragment")
    run(args)
    print(json.dumps({"status": "complete"}))


if __name__ == "__main__":
    main()
