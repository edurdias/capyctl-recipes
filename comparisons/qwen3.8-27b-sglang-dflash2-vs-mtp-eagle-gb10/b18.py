#!/usr/bin/env python3
"""Checks and figures for the DFlash2 against MTP/EAGLE comparison (run.sh, remote.sh).

Standard library only, plus capyctl-bench (its directory is --bench, default the
capyctl-bench beside this repository's tools/). Keys are read from the
CapyCTL credentials file and never printed or written.

    b18.py toolcall  --cred F --endpoint URL --model M --out F
    b18.py reasoning --cred F --endpoint URL --model M --out F
    b18.py stall     --cred F --endpoint URL --model M --out F [--streams 4]
                     [--prompt-tokens 80000] [--stream-max-tokens 3072] [--seed N]
    b18.py chat      --cred F --endpoint URL --model M   one short request: `HTTP seconds ok|wrong`
    b18.py facts     LOG [LOG...] --out F         engine-resolved values from an engine log
    b18.py load      --cred F --mport N --name M --out F   CapyCTL's load read for one deployment
    b18.py get       FILE dotted.path             one value of a JSON file (status, inspect)
    b18.py merge     RAW OUT                      pool the interleaved passes per recipe
    b18.py readme    RUN_DIR README               write the comparison README from a run
"""
import argparse
import http.client
import json
import os
import re
import statistics
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_BENCH = HERE.parent.parent / "tools" / "capyctl-bench"


def bench_module(path):
    sys.path.insert(0, str(Path(path or os.environ.get("CAPYCTL_BENCH_DIR") or DEFAULT_BENCH)))
    import capyctl_bench  # noqa: E402
    return capyctl_bench


def credential(path, name):
    """One `name: value` line of the standalone credentials file; never printed."""
    for line in Path(path).read_text().splitlines():
        m = re.match(rf"^\s*{name}\s*:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    raise SystemExit(f"no {name} line in the credentials file")


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")
    print(json.dumps(obj.get("summary", obj))[:600])


# ---------------------------------------------------------------------------
# Tool call and reasoning split
# ---------------------------------------------------------------------------

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
MARKUP = ("<tool_call>", "</tool_call>", "<function=", "<think>", "</think>")


def post(a, body):
    u = a.endpoint.rstrip("/")
    host, port = re.match(r"^http://([^/:]+):(\d+)", u).groups()
    conn = http.client.HTTPConnection(host, int(port), timeout=1800)
    headers = {"Authorization": "Bearer " + credential(a.cred, "api_key"),
               "Content-Type": "application/json"}
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=headers)
    return conn, conn.getresponse(), t0


def cmd_toolcall(a):
    body = {"model": a.model, "temperature": 0, "max_tokens": 4096, "tools": TOOLS,
            "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."}]}
    conn, resp, t0 = post(a, body)
    raw = resp.read()
    conn.close()
    out = {"http_status": resp.status, "total_s": round(time.perf_counter() - t0, 3)}
    try:
        d = json.loads(raw)
    except ValueError:
        d = {}
    choice = (d.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = msg.get("tool_calls") or []
    names = [((c.get("function") or {}).get("name")) for c in calls]
    args = []
    for c in calls:
        try:
            args.append(json.loads((c.get("function") or {}).get("arguments") or "{}"))
        except ValueError:
            args.append(None)
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    out.update({
        "finish_reason": choice.get("finish_reason"),
        "tool_calls": [{"name": n, "arguments": g} for n, g in zip(names, args)],
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "markup_in_content": [m for m in MARKUP if m in content],
        "engine_metrics": sorted(((d.get("capyctl") or {}).get("metrics") or {}).keys()),
    })
    ok = (resp.status == 200 and "get_weather" in names
          and any(isinstance(g, dict) and "lisbon" in str(g.get("city", "")).lower() for g in args)
          and not out["markup_in_content"])
    out["summary"] = {"pass": ok, "detail": (
        f"HTTP {resp.status}, finish {out['finish_reason']}, calls {names}, "
        f"reasoning {len(reasoning)} chars, markup in content {out['markup_in_content'] or 'none'}")}
    write(a.out, out)


def cmd_chat(a):
    """One short non-streaming request; prints `HTTP seconds ok|wrong` (a wake is timed by it)."""
    body = {"model": a.model, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "user", "content": "What is 17+25? Answer with only the number."}]}
    conn, resp, t0 = post(a, body)
    raw = resp.read()
    conn.close()
    took = time.perf_counter() - t0
    try:
        msg = (json.loads(raw).get("choices") or [{}])[0].get("message") or {}
    except (ValueError, AttributeError):
        msg = {}
    print(f"{resp.status} {took:.1f} {'ok' if '42' in (msg.get('content') or '') else 'wrong'}")


def cmd_get(a):
    """The value at a dotted path (`[n]` indexes a list), searched from every
    object in the file, so the path may start at any depth; empty when absent."""
    def follow(d, parts):
        for part in parts:
            m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
            if not m or not isinstance(d, dict) or m.group(1) not in d:
                return None
            d = d[m.group(1)]
            if m.group(2) is not None:
                if not isinstance(d, list) or int(m.group(2)) >= len(d):
                    return None
                d = d[int(m.group(2))]
        return d

    def walk(d):
        yield d
        children = d.values() if isinstance(d, dict) else d if isinstance(d, list) else ()
        for v in children:
            yield from walk(v)

    data = load_json(a.file)
    for node in walk(data) if data is not None else ():
        v = follow(node, a.path.split("."))
        if v is not None:
            print(v if not isinstance(v, (dict, list)) else json.dumps(v))
            return



def cmd_reasoning(a):
    body = {"model": a.model, "temperature": 0, "max_tokens": 4096, "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "What is 17 * 23? Think it through, then give the number."}]}
    conn, resp, t0 = post(a, body)
    out = {"http_status": resp.status, "reasoning_chunks": 0, "content_chunks": 0, "done": False,
           "usage": False, "metrics_comment": False, "finish_reason": None, "first_token_s": None}
    reasoning, content = [], []
    for line in resp:
        line = line.strip()
        if line.startswith(b": x-capyctl-metrics"):
            out["metrics_comment"] = True
            continue
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            out["done"] = True
            break
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if obj.get("usage"):
            out["usage"] = True
        for ch in obj.get("choices") or []:
            d = ch.get("delta") or {}
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                out["reasoning_chunks"] += 1
                reasoning.append(r)
            if d.get("content"):
                out["content_chunks"] += 1
                content.append(d["content"])
            if (r or d.get("content")) and out["first_token_s"] is None:
                out["first_token_s"] = round(time.perf_counter() - t0, 3)
            if ch.get("finish_reason"):
                out["finish_reason"] = ch["finish_reason"]
    conn.close()
    text = "".join(content)
    out["total_s"] = round(time.perf_counter() - t0, 3)
    out["markup_in_content"] = [m for m in MARKUP if m in text]
    out["answer_ok"] = "391" in text
    out["content_tail"] = text[-120:]
    ok = (resp.status == 200 and out["done"] and out["reasoning_chunks"] > 0
          and out["content_chunks"] > 0 and not out["markup_in_content"] and out["answer_ok"])
    out["summary"] = {"pass": ok, "detail": (
        f"{out['reasoning_chunks']} reasoning chunks, {out['content_chunks']} content chunks, "
        f"markup in content {out['markup_in_content'] or 'none'}, answer {'391' if out['answer_ok'] else 'wrong'}, "
        f"usage {'yes' if out['usage'] else 'no'}, metrics comment {'yes' if out['metrics_comment'] else 'no'}")}
    write(a.out, out)


# ---------------------------------------------------------------------------
# Long-prompt stall
# ---------------------------------------------------------------------------

def cmd_stall(a):
    """While `streams` requests decode, send one cold long prompt and measure the
    longest gap between token chunks on the running streams."""
    cb = bench_module(a.bench)
    ep = cb.Endpoint(a.endpoint.rstrip("/") + "/v1", credential(a.cred, "api_key"), timeout=1800)

    def payload(messages, max_tokens, extra=None):
        body = {"model": a.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        body.update(extra or {})
        return body

    # Calibration on short probes (their own seeds), so the measured prompt is cold.
    probes = []
    for n in (20, 400):
        rec = cb.stream_chat(ep, payload(cb.context_messages(n, 700000 + a.seed * 10 + n), 1))
        if not rec["prompt_tokens"]:
            raise SystemExit(f"calibration failed: {rec['error']}")
        probes.append((n, rec["prompt_tokens"]))
    (n1, t1), (n2, t2) = probes
    per = (t2 - t1) / (n2 - n1)
    sentences = max(1, round((a.prompt_tokens - (t1 - per * n1)) / per))

    prompts = cb.load_prompts(Path(cb.__file__).parent / "prompts.json")
    results = [None] * a.streams

    def stream(i):
        p = prompts[i % len(prompts)]
        body = payload(p["messages"], a.stream_max_tokens, {"ignore_eos": True})
        results[i] = cb.stream_chat(ep, body, record_chunks=True)

    threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(a.streams)]
    t_streams = time.time()
    for t in threads:
        t.start()
    time.sleep(a.lead_s)
    long_rec = {}

    def long_request():
        long_rec.update(cb.stream_chat(ep, payload(cb.context_messages(sentences, 600000 + a.seed), 64),
                                       record_chunks=True))

    t_send = time.time()
    lt = threading.Thread(target=long_request, daemon=True)
    lt.start()
    lt.join()
    for t in threads:
        t.join()
    if not long_rec.get("ttft_s"):
        raise SystemExit(f"the long request failed: {long_rec.get('error')}")
    t_first = t_send + long_rec["ttft_s"]
    per_stream = []
    for i, rec in enumerate(results):
        times = [rec["sent_at"] + s for s in rec.get("chunk_times_s") or []]
        gaps = [(b - a_, a_, b) for a_, b in zip(times, times[1:])]
        # A gap counts for the prefill window when any part of it falls inside
        # [send, first token of the long prompt].
        during = [g for g, s, e in gaps if e > t_send and s < t_first]
        before = [g for g, s, e in gaps if e <= t_send and s >= t_send - a.lead_s + 5]
        per_stream.append({
            "stream": i, "error": rec["error"], "completion_tokens": rec["completion_tokens"],
            "chunks": len(times),
            "decoding_at_send": bool(times) and times[0] < t_send,
            "alive_after_prefill": bool(times) and times[-1] > t_first,
            "max_gap_prefill_s": round(max(during), 3) if during else None,
            "max_gap_before_s": round(max(before), 3) if before else None,
            "p50_gap_before_s": round(statistics.median(before), 4) if before else None,
            "chunks_during_prefill": sum(1 for t in times if t_send <= t <= t_first),
        })
    valid = [s for s in per_stream if s["decoding_at_send"] and s["alive_after_prefill"] and not s["error"]]
    gaps = [s["max_gap_prefill_s"] for s in valid if s["max_gap_prefill_s"] is not None]
    base = [s["max_gap_before_s"] for s in valid if s["max_gap_before_s"] is not None]
    out = {
        "model": a.model, "streams": a.streams, "stream_max_tokens": a.stream_max_tokens,
        "lead_s": a.lead_s, "target_prompt_tokens": a.prompt_tokens, "sentences": sentences,
        "calibration": {"per_sentence_tokens": per, "probes": probes},
        "long": {k: long_rec.get(k) for k in ("error", "prompt_tokens", "completion_tokens", "ttft_s",
                                               "total_s", "prompt_tps")},
        "per_stream": per_stream,
        "summary": {
            "valid_streams": len(valid),
            "long_prompt_tokens": long_rec.get("prompt_tokens"),
            "long_ttft_s": round(long_rec["ttft_s"], 2),
            "max_gap_prefill_s": max(gaps) if gaps else None,
            "max_gap_before_s": max(base) if base else None,
            "streams_started_s_after": round(t_send - t_streams, 1),
        },
    }
    write(a.out, out)


# ---------------------------------------------------------------------------
# Engine-resolved values from an engine log (redacted by CapyCTL as written).
# Only the named values are extracted; no log line is copied.
# ---------------------------------------------------------------------------

FACT_PATTERNS = {
    "max_total_num_tokens": r"max_total_num_tokens=(\d+)",
    "chunked_prefill_size": r"\bchunked_prefill_size=(-?\d+)",
    "max_prefill_tokens": r"\bmax_prefill_tokens=(\d+)",
    "max_running_requests": r"\bmax_running_requests=(\d+)",
    "context_len": r"\bcontext_len=(\d+)",
    "available_gpu_mem_gb": r"available_gpu_mem=([\d.]+)",
    "mem_fraction_static": r"\bmem_fraction_static=([\d.]+)",
    "kv_cache_dtype": r"\bkv_cache_dtype='?([A-Za-z0-9_]+)",
    "mamba_ssm_dtype": r"\bmamba_ssm_dtype='?([A-Za-z0-9_]+)",
    "mamba_radix_cache_strategy": r"\bmamba_radix_cache_strategy='?([A-Za-z0-9_]+)",
    "max_mamba_cache_size": r"\bmax_mamba_cache_size[=:]\s*(\d+)",
    "enable_mixed_chunk": r"\benable_mixed_chunk=(True|False)",
    "enable_linear_replayssm_spec": r"\benable_linear_replayssm_spec=(True|False)",
    "speculative_algorithm": r"\bspeculative_algorithm='?([A-Za-z0-9_]+)",
    "speculative_num_steps": r"\bspeculative_num_steps=(\d+)",
    "speculative_num_draft_tokens": r"\bspeculative_num_draft_tokens=(\d+)",
    "cuda_graph_max_bs": r"\bcuda_graph_max_bs=(\d+)",
    "tool_call_parser": r"\btool_call_parser='?([A-Za-z0-9_]+)",
    "reasoning_parser": r"\breasoning_parser='?([A-Za-z0-9_]+)",
    "kv_tokens_allocated": r"KV Cache is allocated\. #tokens: (\d+)",
    "load_weight_mem_gb": r"Load weight end\..*?mem usage=([\d.]+) GB",
    "cuda_graph_capture_s": r"Capture cuda graph end\. Time elapsed: ([\d.]+) s",
}
ACCEPT = re.compile(r"accept len: ([\d.]+)")
RUNNING = re.compile(r"#running-req: (\d+)")


def cmd_facts(a):
    facts, accept, running, oom = {}, [], [], 0
    for path in a.logs:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            for key, pat in FACT_PATTERNS.items():
                m = re.search(pat, line)
                if m:
                    facts[key] = m.group(1)
            m = ACCEPT.search(line)
            if m:
                accept.append(float(m.group(1)))
            m = RUNNING.search(line)
            if m:
                running.append(int(m.group(1)))
            if "out of memory" in line.lower():
                oom += 1
    if accept:
        facts["accept_len"] = {"n": len(accept), "mean": round(statistics.mean(accept), 3),
                               "median": round(statistics.median(accept), 3),
                               "min": min(accept), "max": max(accept)}
    if running:
        facts["running_req_max"] = max(running)
    facts["out_of_memory_lines"] = oom
    write(a.out, {"logs": [Path(p).name for p in a.logs], "summary": facts})


def cmd_load(a):
    token = credential(a.cred, "admin_token")
    conn = http.client.HTTPConnection("127.0.0.1", a.mport, timeout=10)
    conn.request("GET", "/management/v1/metrics/load", headers={"Authorization": "Bearer " + token})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    try:
        d = json.loads(raw)
    except ValueError:
        d = {}
    mine = [x for x in d.get("deployments") or [] if x.get("name") == a.name]
    write(a.out, {"http_status": resp.status, "summary": mine[0] if mine else None})


# ---------------------------------------------------------------------------
# Pool passes and write the README
# ---------------------------------------------------------------------------

SIDES = {"a": "dflash2", "b": "mtp-eagle"}


def cmd_merge(a):
    cb = bench_module(a.bench)
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for side, name in SIDES.items():
        passes = [json.loads(p.read_text()) for p in sorted(raw.glob(f"conc-{side}-pass*.json"))]
        ctx_path = raw / f"ctx-{side}.json"
        ctx = json.loads(ctx_path.read_text()) if ctx_path.exists() else None
        if not passes and not ctx:
            print(f"{name}: no results")
            continue
        merged = json.loads(json.dumps(passes[0] if passes else ctx))
        if passes:
            for i, lv in enumerate(merged["concurrency"]["levels"]):
                rounds = []
                for p, data in enumerate(passes, start=1):
                    for rnd in data["concurrency"]["levels"][i]["rounds"]:
                        rounds.append(dict(rnd, round=len(rounds), **{"pass": p}))
                lv["rounds"] = rounds
                cb.summarize_level(lv)
            merged["concurrency"]["settings"]["rounds"] = sum(d["concurrency"]["settings"]["rounds"] for d in passes)
            merged["started_at"], merged["finished_at"] = passes[0]["started_at"], passes[-1]["finished_at"]
            merged["meta"]["passes"] = f"{len(passes)}, interleaved DFlash2, MTP/EAGLE; each a fresh warm start"
            if all(d.get("memory") for d in passes):
                merged["memory"] = {"unit": "GiB", "interval_s": passes[0]["memory"]["interval_s"],
                                    "errors": sum(d["memory"]["errors"] for d in passes),
                                    "samples": [s for d in passes for s in d["memory"]["samples"]]}
        else:
            merged["concurrency"] = None
        merged["context_sweep"] = ctx["context_sweep"] if ctx else None
        (out / f"{name}.json").write_text(json.dumps(merged, indent=1) + "\n")
        print(f"{name}: {len(passes)} passes, context sweep {'yes' if ctx else 'no'}")


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def figures(run):
    """figures.tsv: side, key, value (remote.sh writes one line per figure)."""
    f = {}
    path = Path(run) / "raw" / "figures.tsv"
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                f.setdefault(parts[0], {}).setdefault(parts[1], []).append(parts[2])
    return f


def num(v, fmt="{:.1f}"):
    try:
        return fmt.format(float(v))
    except (TypeError, ValueError):
        return "n/a"


def pct(b, a_):
    try:
        return f"{(float(b) / float(a_) - 1) * 100:+.0f}%"
    except (TypeError, ValueError, ZeroDivisionError):
        return ""


def first(f, side, key):
    v = (f.get(side) or {}).get(key)
    return v[-1] if v else None


def span(f, side, key):
    v = [float(x) for x in (f.get(side) or {}).get(key) or [] if x not in ("", "n/a")]
    if not v:
        return "n/a"
    return f"{min(v):.0f} s" if len(v) == 1 or min(v) == max(v) else f"{min(v):.0f} to {max(v):.0f} s"


def gib(v):
    try:
        return f"{int(v) / 2**30:.1f} GiB"
    except (TypeError, ValueError):
        return "n/a"


def cmd_readme(a):
    run = Path(a.run)
    raw = run / "raw"
    res = run / "results"
    f = figures(run)
    settings = load_json(run / "settings.json") or {}
    versions = {}
    vpath = raw / "versions.txt"
    if vpath.exists():
        for line in vpath.read_text().splitlines():
            k, _, v = line.partition("=")
            versions[k] = v
    series = {s: load_json(res / f"{n}.json") for s, n in SIDES.items()}
    facts = {k: (load_json(raw / f"facts-{k}.json") or {}).get("summary") or {}
             for k in ("cold-a", "cold-b", "life-a", "life-b", "ctx-a", "ctx-b", "stall-bn")}
    conc_facts = {s: [(load_json(p) or {}).get("summary") or {} for p in sorted(raw.glob(f"facts-conc-{s}-pass*.json"))]
                  for s in SIDES}

    def fact(side, key):
        for k in (f"life-{side}", f"cold-{side}", f"ctx-{side}"):
            v = facts.get(k, {}).get(key)
            if v is not None:
                return v
        return None

    def accept(side):
        vals = [c.get("accept_len", {}).get("mean") for c in conc_facts[side] if c.get("accept_len")]
        vals = [v for v in vals if v is not None]
        return f"{statistics.mean(vals):.2f}" if vals else "not logged"

    def check(name, side):
        d = load_json(raw / f"{name}-{side}.json") or {}
        s = d.get("summary") or {}
        return ("pass" if s.get("pass") else "FAIL") + (f": {s.get('detail')}" if s.get("detail") else "")

    def conc_rows():
        rows = []
        la = ((series["a"] or {}).get("concurrency") or {}).get("levels") or []
        lb = ((series["b"] or {}).get("concurrency") or {}).get("levels") or []
        by = {lv["concurrency"]: lv["summary"] for lv in lb}
        for lv in la:
            c, sa, sb = lv["concurrency"], lv["summary"], by.get(lv["concurrency"]) or {}
            ag_a, ag_b = (sa.get("aggregate_tps") or {}).get("median"), (sb.get("aggregate_tps") or {}).get("median")
            ps_a, ps_b = (sa.get("per_stream_tps") or {}).get("median"), (sb.get("per_stream_tps") or {}).get("median")
            rows.append(f"| {c} | {num(ag_a)} | {num(ag_b)} ({pct(ag_b, ag_a)}) | {num(ps_a)} | {num(ps_b)} | "
                        f"{num(sa.get('ttft_p50_s'), '{:.2f}')} s | {num(sb.get('ttft_p50_s'), '{:.2f}')} s | "
                        f"{sa.get('errors', 'n/a')} / {sb.get('errors', 'n/a')} |")
        return "\n".join(rows) or "| no concurrency results | | | | | | | |"

    def ctx_rows():
        rows = []
        pa = ((series["a"] or {}).get("context_sweep") or {}).get("points") or []
        pb = {p["label"]: p["summary"] for p in ((series["b"] or {}).get("context_sweep") or {}).get("points") or []}
        for p in pa:
            sa, sb = p["summary"], pb.get(p["label"]) or {}
            g = lambda s, k, fm="{:.1f}": num((s.get(k) or {}).get("median"), fm)  # noqa: E731
            rows.append(f"| {p['label']} | {g(sa, 'ttft_s', '{:.2f}')} s | {g(sb, 'ttft_s', '{:.2f}')} s | "
                        f"{g(sa, 'decode_tps')} | {g(sb, 'decode_tps')} | {g(sa, 'prompt_tps', '{:.0f}')} | "
                        f"{g(sb, 'prompt_tps', '{:.0f}')} |")
        return "\n".join(rows) or "| no context results | | | | | | |"

    def stall_cell(variant, key, fm="{:.1f}"):
        vals = [((load_json(p) or {}).get("summary") or {}).get(key) for p in sorted(raw.glob(f"stall-{variant}-t*.json"))]
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n/a"
        return " / ".join(fm.format(v) for v in vals)

    b_model = settings.get("b_model_ref", "RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead@009632fef96dd349150baa780c984e62e70e91fe")
    text = f"""# SGLang 0.5.21, Qwen3.8-27B: DFlash2 drafter against the MTP/EAGLE recipe, one GB10

Two SGLang 0.5.21 recipes for Qwen3.8-27B on the same machine, through CapyCTL,
measured with capyctl-bench and interleaved pass by pass:

- **DFlash2** (this repository's catalog recipe, [sglang/qwen3.8-27b-nvfp4-gb10](../../sglang/qwen3.8-27b-nvfp4-gb10/)):
  `nvidia/Qwen3.8-27B-NVFP4`, the DFlash2 drafter with 8 draft tokens, FP8 KV cache
  of 16 GiB, 8 running requests, a 65 GiB memory request.
- **MTP/EAGLE**: `{b_model.split('@')[0]}`, an NVFP4 checkpoint with a BF16 LM head,
  drafting with its own MTP layer (EAGLE, 3 steps, top-1, 4 draft tokens) and SGLang's
  ReplaySSM speculative verify; a 110-slot state cache for 22 running requests, mixed
  chunked prefill, a 262144-token context and a 79 GiB memory request with a 29 GiB FP8 KV pool.

![Summary](bench/summary/summary-wide.png)

The full report is [bench/report.html](bench/report.html): download it and open
it locally (GitHub shows the HTML source). [bench/summary.md](bench/summary.md)
has every table, [bench/data.csv](bench/data.csv) every point, and
[bench/results/](bench/results/) the capyctl-bench results files with every
request. [results/](results/) holds the lifecycle, check and stall-test records of the run.

## Setup

| | |
|---|---|
| Hardware | 1x NVIDIA {versions.get('gpu', 'GB10')}, unified memory, aarch64 |
| System | NVIDIA driver {versions.get('driver', 'n/a')}, CUDA {versions.get('cuda', 'n/a')}, kernel {versions.get('kernel', 'n/a')} |
| CapyCTL | `main` at `{settings.get('capyctl_commit', 'n/a')[:7]}` (prints `{versions.get('capyctl_version', 'n/a')}`), release build, `capyctl start standalone` |
| SGLang | {versions.get('sglang', 'n/a')} (torch {versions.get('torch', 'n/a')}, flashinfer-python {versions.get('flashinfer', 'n/a')}, sglang-kernel {versions.get('sglang_kernel', 'n/a')}, transformers {versions.get('transformers', 'n/a')}, torch-memory-saver {versions.get('torch_memory_saver', 'n/a')}), Python {versions.get('python', 'n/a')} |
| DFlash2 model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at `482ca0f3832238542f8f5295dde86b5f22711d80` |
| Drafter | [`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) at `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` |
| MTP/EAGLE model | [`{b_model.split('@')[0]}`](https://huggingface.co/{b_model.split('@')[0]}) at `{b_model.split('@')[1]}`, its own MTP layer, no separate drafter |
| Benchmark | capyctl-bench from this repository at `{settings.get('bench_commit', 'n/a')[:7]}` |
| Measured | {settings.get('date', 'n/a')} |

CapyCTL ran standalone with its own state directory, the managed memory limit
raised to {settings.get('managed_limit', '100GiB')} and the parked limit to {settings.get('parked_limit', '45GiB')}:

```bash
capyctl start standalone --listen 127.0.0.1:{settings.get('port', 'PORT')} \\
  --set host.resource_policy.memory.system.managed_limit={settings.get('managed_limit', '100GiB')} \\
  --set host.resource_policy.memory.system.parked_limit={settings.get('parked_limit', '45GiB')}
```

Each recipe ran under its own profile of the same SGLang venv, so that each
had its own, initially empty, kernel caches (`SGLANG_CACHE_DIR` and the
FlashInfer, Triton, TVM-FFI, Inductor and CUDA cache variables in the profile's
`env`); the DFlash2 profile approves the drafter path:

```bash
capyctl engine add ~/sglang-0.5.21-venv --name b18a \\
  --approve-option=--speculative-draft-model-path --approve-path ~/drafters --env SGLANG_CACHE_DIR=<dir a> ...
capyctl engine add ~/sglang-0.5.21-venv --name b18b --env SGLANG_CACHE_DIR=<dir b> ...
```

## Deployments

- [deployment-dflash2.yaml](deployment-dflash2.yaml): the catalog recipe, short-prompt runs
  (`context_length: 32768`); [deployment-dflash2-ctx.yaml](deployment-dflash2-ctx.yaml) the same with
  `context_length: 262144`, for the 128k point and the stall test.
- [deployment-mtp-eagle.yaml](deployment-mtp-eagle.yaml): the MTP/EAGLE recipe, every run;
  [deployment-mtp-eagle-nomix.yaml](deployment-mtp-eagle-nomix.yaml) the same without
  `--enable-mixed-chunk`, for the stall test only.

Only one deployment ran at a time.

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| Speculative decoding | `DFLASH`, separate drafter, 8 draft tokens, drafter unquantized | `EAGLE` on the checkpoint's MTP layer, 3 steps, top-1, 4 draft tokens, `--enable-linear-replayssm-spec` |
| Running requests | `max_concurrent_requests: 8`; engine `max_running_requests={fact('a', 'max_running_requests') or 'n/a'}` | `max_concurrent_requests: 22`, `--max-mamba-cache-size 110`; engine `max_running_requests={fact('b', 'max_running_requests') or 'n/a'}` |
| KV cache | `fp8_e4m3`, 16 GiB; engine {fact('a', 'max_total_num_tokens') or 'n/a'} tokens | `fp8_e4m3` (the checkpoint's calibrated default, stated), 29 GiB; engine {fact('b', 'max_total_num_tokens') or 'n/a'} tokens |
| Chunked prefill, max prefill | {fact('a', 'chunked_prefill_size') or 'n/a'}, {fact('a', 'max_prefill_tokens') or 'n/a'} | {fact('b', 'chunked_prefill_size') or 'n/a'}, {fact('b', 'max_prefill_tokens') or 'n/a'}; mixed chunks {fact('b', 'enable_mixed_chunk') or 'n/a'} |
| State cache | sized by CapyCTL for 8 requests | {fact('b', 'max_mamba_cache_size') or '110'} slots, `mamba_ssm_dtype` {fact('b', 'mamba_ssm_dtype') or 'n/a'}, radix strategy {fact('b', 'mamba_radix_cache_strategy') or 'n/a'} |
| Memory request, rendered `mem_fraction_static` | 65 GiB, {fact('a', 'mem_fraction_static') or 'n/a'} | 79 GiB, {fact('b', 'mem_fraction_static') or 'n/a'} |
| Parsers | `qwen3_coder`, `qwen3` (CapyCTL's choice for the family) | `qwen3_coder`, `qwen3` (declared) |
| Residency | CapyCTL default (`deep`, weights kept resident while parked) | the same |

## Method

- Every request went through the CapyCTL inference endpoint: OpenAI streaming
  chat completions, `temperature: 0`, thinking on (the model's default).
- **Short prompts**: {settings.get('concurrency', '1,4,8,16,22').replace(',', ', ')} streams, `max_tokens: 512`, capyctl-bench's
  `prompts.json`, {settings.get('rounds', '5')} measured rounds after one warm-up round per stream count, in
  {settings.get('passes', '2')} passes interleaved DFlash2, MTP/EAGLE, DFlash2, MTP/EAGLE; each pass a fresh warm
  start. The DFlash2 recipe runs 8 requests at once, so at 16 and 22 streams the rest wait.
- **Context**: one stream, {settings.get('sweep', '2k,32k,128k').replace(',', ', ')} prompt tokens, {settings.get('runs', '3')} runs each, `max_tokens: 128`.
- **Long-prompt stall**: {settings.get('stall_streams', '4')} streams decode (`max_tokens: {settings.get('stall_max_tokens', '3072')}`, `ignore_eos`), then one cold
  prompt of about {settings.get('stall_tokens', '80000')} tokens arrives; the figure is the longest gap between token
  chunks on the running streams while that prompt is prefilled, {settings.get('stall_trials', '2')} trials per variant.
- **Lifecycle**: a first start with empty kernel caches (cold), warm starts after `stop`,
  the ready footprint (drop in `MemAvailable` from before the start to Ready),
  one park and wake (a request wakes it) with CapyCTL's parked charge, and stop.
- Engine figures come from the engine's own log, which CapyCTL keeps redacted
  (`<state dir>/logs/<deployment>/<launch>.log`), and CapyCTL's load read.

## Results

### Lifecycle and memory

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| First start, empty kernel caches | {span(f, 'a', 'cold_s')} | {span(f, 'b', 'cold_s')} |
| `MAX_JOBS` CapyCTL derived for that start | {first(f, 'a', 'max_jobs') or 'n/a'} | {first(f, 'b', 'max_jobs') or 'n/a'} |
| Kernel caches after the first start | {first(f, 'a', 'jit_cache') or 'n/a'} | {first(f, 'b', 'jit_cache') or 'n/a'} |
| Warm start (`start` after `stop`) | {span(f, 'a', 'warm_s')} | {span(f, 'b', 'warm_s')} |
| Ready footprint (`MemAvailable` drop) | {num(first(f, 'a', 'ready_gib'))} GiB | {num(first(f, 'b', 'ready_gib'))} GiB |
| CapyCTL Ready charge | {gib(first(f, 'a', 'ready_charge'))} | {gib(first(f, 'b', 'ready_charge'))} |
| Park, parked charge, parked footprint | {num(first(f, 'a', 'park_s'))} s, {gib(first(f, 'a', 'parked_charge'))}, {num(first(f, 'a', 'parked_gib'))} GiB | {num(first(f, 'b', 'park_s'))} s, {gib(first(f, 'b', 'parked_charge'))}, {num(first(f, 'b', 'parked_gib'))} GiB |
| Wake by request, to the answer | {num(first(f, 'a', 'wake_s'))} s | {num(first(f, 'b', 'wake_s'))} s |
| Engine KV tokens / running requests | {fact('a', 'max_total_num_tokens') or 'n/a'} / {fact('a', 'max_running_requests') or 'n/a'} | {fact('b', 'max_total_num_tokens') or 'n/a'} / {fact('b', 'max_running_requests') or 'n/a'} |
| CapyCTL load read `max_running` | {first(f, 'a', 'load_max_running') or 'n/a'} | {first(f, 'b', 'load_max_running') or 'n/a'} |
| Mean accept length (engine log, short-prompt passes) | {accept('a')} | {accept('b')} |

### Short prompts

| Streams | DFlash2 aggregate, tok/s | MTP/EAGLE aggregate | DFlash2 per stream p50 | MTP/EAGLE per stream p50 | DFlash2 TTFT p50 | MTP/EAGLE TTFT p50 | Errors |
|---|---|---|---|---|---|---|---|
{conc_rows()}

### Context

| Prompt | DFlash2 TTFT | MTP/EAGLE TTFT | DFlash2 decode, tok/s | MTP/EAGLE decode | DFlash2 prompt, tok/s | MTP/EAGLE prompt |
|---|---|---|---|---|---|---|
{ctx_rows()}

### Long-prompt stall

Longest gap between token chunks on the running streams while the cold prompt
is prefilled, per trial (the gap before it arrives in brackets):

| | DFlash2 | MTP/EAGLE | MTP/EAGLE without mixed chunks |
|---|---|---|---|
| Longest gap, s | {stall_cell('a', 'max_gap_prefill_s')} | {stall_cell('b', 'max_gap_prefill_s')} | {stall_cell('bn', 'max_gap_prefill_s')} |
| Before the prompt, s | ({stall_cell('a', 'max_gap_before_s', '{:.2f}')}) | ({stall_cell('b', 'max_gap_before_s', '{:.2f}')}) | ({stall_cell('bn', 'max_gap_before_s', '{:.2f}')}) |
| Long prompt: tokens, time to first token | {stall_cell('a', 'long_prompt_tokens', '{:.0f}')}, {stall_cell('a', 'long_ttft_s')} s | {stall_cell('b', 'long_prompt_tokens', '{:.0f}')}, {stall_cell('b', 'long_ttft_s')} s | {stall_cell('bn', 'long_prompt_tokens', '{:.0f}')}, {stall_cell('bn', 'long_ttft_s')} s |

### Checks

| | DFlash2 | MTP/EAGLE |
|---|---|---|
| Tool call (`get_weather`, structured `tool_calls`) | {check('toolcall', 'a')} | {check('toolcall', 'b')} |
| Reasoning stream (trace in `reasoning_content`, none in `content`) | {check('reasoning', 'a')} | {check('reasoning', 'b')} |

## Caveats

- One machine, {settings.get('passes', '2')} passes for the short prompts and one for the context sweep.
  Thinking was on, so most of each 512-token reply is reasoning.
- The two recipes differ in checkpoint (the MTP/EAGLE checkpoint keeps a BF16 LM head
  and carries the MTP layer), drafter, KV pool, running requests and context: this
  compares recipes, not one setting.
- CapyCTL counts the state of `--max-mamba-cache-size 110` with one intermediate
  state per draft token per running request, which SGLang does not keep under
  `--enable-linear-replayssm-spec`; the static pool it renders is therefore the
  whole 79 GiB request.
- The kernel caches started empty for each recipe's first start; later starts reused them.
"""
    Path(a.readme).write_text(text)
    print(f"wrote {a.readme}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", help="capyctl-bench directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("toolcall", "reasoning", "stall", "chat"):
        s = sub.add_parser(name)
        s.add_argument("--cred", required=True)
        s.add_argument("--endpoint", required=True, help="CapyCTL inference base, http://127.0.0.1:PORT")
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=name != "chat")
        if name == "stall":
            s.add_argument("--streams", type=int, default=4)
            s.add_argument("--prompt-tokens", type=int, default=80000)
            s.add_argument("--stream-max-tokens", type=int, default=3072)
            s.add_argument("--lead-s", type=float, default=20.0, help="seconds the streams decode first")
            s.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("facts")
    s.add_argument("logs", nargs="+")
    s.add_argument("--out", required=True)
    s = sub.add_parser("load")
    s.add_argument("--cred", required=True)
    s.add_argument("--mport", type=int, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("get")
    s.add_argument("file")
    s.add_argument("path")
    s = sub.add_parser("merge")
    s.add_argument("raw")
    s.add_argument("out")
    s = sub.add_parser("readme")
    s.add_argument("run", help="the run's local output directory (raw/, results/, settings.json)")
    s.add_argument("readme")
    a = p.parse_args()
    {"toolcall": cmd_toolcall, "reasoning": cmd_reasoning, "stall": cmd_stall, "chat": cmd_chat,
     "facts": cmd_facts, "load": cmd_load, "get": cmd_get, "merge": cmd_merge,
     "readme": cmd_readme}[a.cmd](a)


if __name__ == "__main__":
    main()
