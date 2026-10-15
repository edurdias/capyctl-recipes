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
    b18.py serverargs LOG [LOG...] --out F        SGLang's resolved server arguments (no credentials)
    b18.py public    --name M --out F             CapyCTL's public launch settings and engine env names
    b18.py effective --cred F --mport N --id D --out F   CapyCTL's effective configuration (charges)
    b18.py variants  RES OUT.md [--names L]       the variant table (run.sh --variants)
    b18.py agree     --cred F --endpoint URL --model M --prompts P --out F [--max-tokens 256] [--limit N]
    b18.py gsm8k     --cred F --endpoint URL --model M --data D --out F [--max-tokens N] [--limit N]
    b18.py tools     --cred F --endpoint URL --model M --cases C --out F
    b18.py faster    RES NAME NAME                the variant with the higher C1 per-stream decode
    b18.py experiment RES OUT.md --names L        variant tables, stall test, accuracy (run.sh --experiment)
    b18.py sweetspot OUT.md --run HOST=DIR [--run HOST=DIR] [--ref CMP]   the sweet-spot report
    b18.py merge     RAW OUT                      pool the interleaved passes per recipe
    b18.py readme    RUN_DIR README               rewrite the README's run 1 section (between its markers) from a run
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


def find_path(data, path):
    """The value at a dotted path (`[n]` indexes a list), searched from every
    object in `data`, so the path may start at any depth; None when absent."""
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

    for node in walk(data) if data is not None else ():
        v = follow(node, path.split("."))
        if v is not None:
            return v
    return None


def cmd_get(a):
    """One value of a JSON file at a dotted path (see find_path); empty when absent."""
    v = find_path(load_json(a.file), a.path)
    if v is not None:
        print(v if not isinstance(v, (dict, list)) else json.dumps(v))


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

def _kv(name, value):
    """`name=value` (SGLang's summary lines) or `'name': value` (its
    `server_args={...}` dict), the value optionally quoted."""
    return rf"['\"]?\b{name}['\"]?\s*(?:=|:)\s*['\"]?({value})"


FACT_PATTERNS = {
    "max_total_num_tokens": r"max_total_num_tokens=(\d+)",
    "chunked_prefill_size": _kv("chunked_prefill_size", r"-?\d+"),
    "max_prefill_tokens": _kv("max_prefill_tokens", r"\d+"),
    "max_running_requests": _kv("max_running_requests", r"\d+"),
    "context_len": r"\bcontext_len=(\d+)",
    "available_gpu_mem_gb": r"available_gpu_mem=([\d.]+)",
    "mem_fraction_static": _kv("mem_fraction_static", r"[\d.]+"),
    "kv_cache_dtype": _kv("kv_cache_dtype", r"[A-Za-z0-9_]+"),
    "mamba_ssm_dtype": _kv("mamba_ssm_dtype", r"[A-Za-z0-9_]+"),
    "mamba_radix_cache_strategy": _kv("mamba_radix_cache_strategy", r"[A-Za-z0-9_]+"),
    "max_mamba_cache_size": _kv("max_mamba_cache_size", r"\d+"),
    "enable_mixed_chunk": _kv("enable_mixed_chunk", "True|False|true|false"),
    "enable_memory_saver": _kv("enable_memory_saver", "True|False|true|false"),
    "enable_linear_replayssm_spec": _kv("enable_linear_replayssm_spec", "True|False|true|false"),
    "speculative_algorithm": _kv("speculative_algorithm", r"[A-Za-z0-9_]+"),
    "speculative_num_steps": _kv("speculative_num_steps", r"\d+"),
    "speculative_num_draft_tokens": _kv("speculative_num_draft_tokens", r"\d+"),
    "cuda_graph_max_bs": _kv("cuda_graph_max_bs", r"\d+"),
    "tool_call_parser": _kv("tool_call_parser", r"[A-Za-z0-9_]+"),
    "reasoning_parser": _kv("reasoning_parser", r"[A-Za-z0-9_]+"),
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
# What a launch was rendered with: SGLang's resolved server arguments (its
# `server_args={...}` log line), CapyCTL's public launch settings (the
# secret-free JSON on the entry's command line) and a few environment names
# of the engine processes. Any name that looks like a credential is dropped.
# ---------------------------------------------------------------------------

SENSITIVE = re.compile(r"(?:^|_)(?:api_key|key|token|secret|password|passwd|credentials?|auth|authorization)(?:$|_)",
                       re.I)
ENV_KEEP = re.compile(r"^(MAX_JOBS|FLASHINFER_NVCC_THREADS|CUDA_HOME|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|"
                      r"PYTORCH_CUDA_ALLOC_CONF|LD_PRELOAD|TMS_.*|SGLANG_.*|FLASHINFER_.*|TRITON_.*|"
                      r"TORCHINDUCTOR_.*|TVM_FFI_.*|CUDA_CACHE_PATH)$")


def flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        name = f"{prefix}{k}"
        if SENSITIVE.search(str(k)):
            continue
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        else:
            out[name] = v
    return out


def cmd_serverargs(a):
    import ast
    found = None
    for path in a.logs:
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            i = line.find("server_args={")
            if i < 0:
                continue
            body = line[i + len("server_args="):].rstrip()
            try:
                found = ast.literal_eval(body)
            except (ValueError, SyntaxError):
                # Not a plain literal: keep the scalar pairs.
                found = {k: v for k, v in re.findall(
                    r"'(\w+)': ('[^']*'|-?[\d.]+(?:e-?\d+)?|True|False|None)", body)}
    flat = {k: (v if isinstance(v, (str, int, float, bool)) or v is None else repr(v))
            for k, v in flatten(found).items()} if found else {}
    write(a.out, {"logs": [Path(p).name for p in a.logs], "found": bool(found), "server_args": flat,
                  "summary": {"found": bool(found), "fields": len(flat)}})


def _proc_cmdlines():
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            yield proc, (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue


def cmd_public(a):
    """The public settings CapyCTL handed SGLang's entry for one deployment,
    and the selected environment of every process of that engine."""
    public, pids = None, []
    for proc, argv in _proc_cmdlines():
        if b"--public-settings-json" not in argv:
            continue
        try:
            doc = json.loads(argv[argv.index(b"--public-settings-json") + 1])
        except (ValueError, IndexError):
            continue
        if doc.get("served_name") == a.name or public is None:
            public, pids = doc, [proc.name]
    env = {}
    if pids:
        # The entry and every process of its group (scheduler, detokenizer).
        def pgid(pid):
            try:
                return os.getpgid(int(pid))
            except OSError:
                return None

        group = pgid(pids[0])
        if group is not None:
            pids = [p.name for p, _ in _proc_cmdlines() if pgid(p.name) == group]
        for pid in pids:
            try:
                raw = Path(f"/proc/{pid}/environ").read_bytes()
            except OSError:
                continue
            for item in raw.split(b"\0"):
                name, _, value = item.decode(errors="replace").partition("=")
                if ENV_KEEP.match(name) and not SENSITIVE.search(name):
                    if name == "LD_PRELOAD":
                        value = ":".join(Path(v).name for v in value.split(":") if v)
                    env.setdefault(name, set()).add(value)
    settings = (public or {}).get("settings") or {}
    write(a.out, {"served_name": (public or {}).get("served_name"), "settings": flatten(settings),
                  "processes": len(pids), "env": {k: sorted(v) for k, v in sorted(env.items())},
                  "summary": {"found": public is not None, "processes": len(pids)}})


VARIANT_FIELDS = re.compile(r"mem_fraction|memory_saver|cpu_backup|kv_cache_dtype|max_total_tokens|"
                            r"max_running|mamba|chunked_prefill|mixed_chunk|cuda_graph|speculative|"
                            r"attention_backend|attn_.*backend|replayssm|overlap|torch_compile|"
                            r"sampling_backend|weight_restore|static_bytes|margin_bytes")


def cmd_variants(a):
    """variants.md and variants.json from the variant runs in RES."""
    res = Path(a.res)
    names = a.names.split(",")
    rows, data = [], {}
    for v in names:
        bench = load_json(res / f"var-{v}.json") or {}
        levels = {lv["concurrency"]: lv["summary"] for lv in (bench.get("concurrency") or {}).get("levels") or []}
        facts = (load_json(res / f"facts-var-{v}.json") or {}).get("summary") or {}
        sargs = (load_json(res / f"serverargs-var-{v}.json") or {}).get("server_args") or {}
        pub = load_json(res / f"public-var-{v}.json") or {}
        data[v] = {"levels": {c: {"aggregate_tps": (s.get("aggregate_tps") or {}).get("median"),
                                  "per_stream_tps": (s.get("per_stream_tps") or {}).get("median"),
                                  "ttft_p50_s": s.get("ttft_p50_s"), "errors": s.get("errors")}
                              for c, s in levels.items()},
                   "accept_len": (facts.get("accept_len") or {}).get("mean"),
                   "engine_kv": facts.get("max_total_num_tokens", "n/a"),
                   "engine_running": facts.get("max_running_requests", "n/a"),
                   "server_args": {k: x for k, x in sargs.items() if VARIANT_FIELDS.search(k)},
                   "capyctl": {k: x for k, x in (pub.get("settings") or {}).items()
                               if VARIANT_FIELDS.search(k) or k in ("memory_saver", "cuda_graphs")},
                   "saver_preload": any("torch_memory_saver" in x for x in (pub.get("env") or {}).get("LD_PRELOAD", [])),
                   "public_all": pub.get("settings") or {},
                   "all_server_args": sargs}
    levels = sorted({c for v in names for c in data[v]["levels"]}, key=int)
    head = "| Variant | " + " | ".join(f"C{c} aggregate / per stream, tok/s" for c in levels) + \
        " | TTFT p50 C1 | Accept length | Memory saver | Saver preload | `mem_fraction_static` | KV dtype" \
        " | KV tokens passed / engine, running | CUDA graph decode bs |"
    rows.append(head)
    rows.append("|" + "---|" * (head.count("|") - 1))
    for v in names:
        d = data[v]
        sa, cp = d["all_server_args"], d["capyctl"]
        graph = next((sa[k] for k in ("cuda_graph_config.decode.max_bs", "cuda_graph_max_bs",
                                      "cuda_graph_config.decode.bs") if k in sa), "n/a")
        cells = [f"{num(d['levels'].get(c, {}).get('aggregate_tps'))} / {num(d['levels'].get(c, {}).get('per_stream_tps'))}"
                 for c in levels]
        rows.append(f"| {v} | " + " | ".join(cells) +
                    f" | {num(d['levels'].get(levels[0], {}).get('ttft_p50_s'), '{:.2f}')} s"
                    f" | {num(d['accept_len'], '{:.2f}')} | {sa.get('enable_memory_saver', cp.get('memory_saver', 'n/a'))}"
                    f" | {'yes' if d['saver_preload'] else 'no'} | {sa.get('mem_fraction_static', 'n/a')}"
                    f" | {sa.get('kv_cache_dtype', cp.get('kv_cache_dtype', 'n/a'))}"
                    f" | {cp.get('max_total_tokens', 'n/a' if v == 'raw' else 'none')} / {d['engine_kv']} / {d['engine_running']}"
                    f" | {graph} |")
    # Every resolved server argument whose value differs between the variants.
    keys = sorted(set().union(*(data[v]["all_server_args"] for v in names)))
    diff = [k for k in keys if len({json.dumps(data[v]["all_server_args"].get(k), sort_keys=True) for v in names}) > 1]
    # CapyCTL's own launch settings exist for the CapyCTL variants only.
    cnames = [v for v in names if v != "raw"]
    ckeys = sorted(set().union(*(data[v]["capyctl"] for v in cnames))) if cnames else []
    cdiff = [k for k in ckeys if len({json.dumps(data[v]["capyctl"].get(k)) for v in cnames}) > 1]
    # Sizing: what CapyCTL rendered, what SGLang resolved, what the machine lost.
    figs = {}
    fpath = res / "figures.tsv"
    if fpath.exists():
        for line in fpath.read_text().splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                figs.setdefault(parts[0], {})[parts[1]] = parts[2]
    sizing = ["| Variant | Memory request | Static pool (CapyCTL) | `--max-total-tokens` rendered | `mem_fraction_static` "
              "| Engine `max_total_num_tokens` | Engine `max_running_requests` | Ready footprint (`MemAvailable` drop) "
              "| CapyCTL Ready charge | Start |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        d, eff = data[v], load_json(res / f"effective-var-{v}.json")
        cp = d["public_all"]
        request = cp.get("memory.request_bytes")
        ready = find_path(eff, "resources.ready.allocations[0].bytes")
        f = figs.get(v, {})
        d["sizing"] = {"request_bytes": request, "static_bytes": cp.get("memory.static_bytes"),
                       "max_total_tokens_rendered": cp.get("max_total_tokens"),
                       "mem_fraction_static": d["all_server_args"].get("mem_fraction_static"),
                       "engine_max_total_num_tokens": d["engine_kv"], "engine_max_running_requests": d["engine_running"],
                       "ready_footprint_gib": f.get("ready_gib"), "capyctl_ready_charge_bytes": ready,
                       "start_s": f.get("start_s")}
        sizing.append(f"| {v} | {gib(request) if request else ('none (raw)' if v == 'raw' else 'n/a')}"
                      f" | {gib(cp.get('memory.static_bytes'))} | {cp.get('max_total_tokens', 'n/a')}"
                      f" | {d['all_server_args'].get('mem_fraction_static', 'n/a')} | {d['engine_kv']} | {d['engine_running']}"
                      f" | {num(f.get('ready_gib'))} GiB | {gib(ready)} | {num(f.get('start_s'), '{:.0f}')} s |")
    r79 = figs.get("refusal79", {})
    lines = ["# MTP/EAGLE recipe: SGLang launched directly (raw) and through CapyCTL, residency and KV dtype variants", "",
             "The CapyCTL variants ask for `memory.request: 84GiB` (the committed recipe's 79 GiB is",
             "refused; see the check below).", "",
             "\n".join(rows), "", "## Sizing", "", "\n".join(sizing), "",
             "## The committed 79 GiB file", "",
             f"- exit code {r79.get('rc', 'n/a')}, request named: {r79.get('named', 'n/a')}",
             f"- message: {r79.get('message', 'n/a')}", "",
             "## CapyCTL launch settings that differ between the CapyCTL variants", ""]
    lines += [f"- `{k}`: " + ", ".join(f"{v} `{data[v]['capyctl'].get(k)}`" for v in cnames) for k in cdiff] or ["- none"]
    lines += ["", "## SGLang resolved server arguments that differ", ""]
    lines += [f"- `{k}`: " + ", ".join(f"{v} `{data[v]['all_server_args'].get(k)}`" for v in names) for k in diff] or \
        ["- none (or no `server_args=` line was found in the engine logs)"]
    Path(a.out).write_text("\n".join(lines) + "\n")
    for v in names:
        data[v].pop("all_server_args")
    Path(a.out).with_suffix(".json").write_text(json.dumps({"variants": data, "server_args_differ": diff,
                                                            "capyctl_differ": cdiff, "refusal79": r79}, indent=1) + "\n")
    print("\n".join(rows + [""] + sizing))


def cmd_effective(a):
    """CapyCTL's effective configuration of one deployment (its charges by phase),
    from the management API with the admin token; credential-named keys dropped."""
    token = credential(a.cred, "admin_token")
    conn = http.client.HTTPConnection("127.0.0.1", a.mport, timeout=10)
    conn.request("GET", f"/management/v1/deployments/{a.id}/effective-config",
                 headers={"Authorization": "Bearer " + token})
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()

    def scrub(o):
        if isinstance(o, dict):
            return {k: scrub(v) for k, v in o.items() if not SENSITIVE.search(str(k))}
        if isinstance(o, list):
            return [scrub(v) for v in o]
        return o
    try:
        d = scrub(json.loads(raw))
    except ValueError:
        d = None
    d = {"http_status": resp.status, "effective_config": d}
    d["summary"] = {"http_status": resp.status,
                    "ready_bytes": find_path(d, "resources.ready.allocations[0].bytes"),
                    "cold_bytes": find_path(d, "resources.cold.allocations[0].bytes")}
    write(a.out, d)


# ---------------------------------------------------------------------------
# Accuracy: greedy token agreement, GSM8K, tool calls. Non-streaming chat
# completions through CapyCTL, `concurrency` at a time, temperature 0.
# ---------------------------------------------------------------------------

def chat_once(a, body, timeout=1800):
    """One non-streaming request; (status, parsed body or None, seconds)."""
    host, port = re.match(r"^http://([^/:]+):(\d+)", a.endpoint.rstrip("/")).groups()
    conn = http.client.HTTPConnection(host, int(port), timeout=timeout)
    headers = {"Content-Type": "application/json"}
    key = credential(a.cred, "api_key") if a.cred else None
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.perf_counter()
    try:
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body), headers=headers)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
    except (OSError, http.client.HTTPException) as exc:
        return 0, {"error": f"{type(exc).__name__}: {exc}"}, time.perf_counter() - t0
    finally:
        conn.close()
    try:
        return status, json.loads(raw), time.perf_counter() - t0
    except ValueError:
        return status, None, time.perf_counter() - t0


def message_parts(d):
    choice = ((d or {}).get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    calls = [{"name": (c.get("function") or {}).get("name"), "arguments": (c.get("function") or {}).get("arguments")}
             for c in msg.get("tool_calls") or []]
    tokens = None
    lp = choice.get("logprobs")
    if isinstance(lp, dict) and isinstance(lp.get("content"), list):
        tokens = [t.get("token") for t in lp["content"] if isinstance(t, dict)]
    return {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "content": msg.get("content") or "", "tool_calls": calls,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": ((d or {}).get("usage") or {}).get("completion_tokens"), "tokens": tokens}


def run_all(a, items, make_body, keep):
    """Send every item, `a.concurrency` at a time; keep(item, status, data, seconds) -> record."""
    from concurrent.futures import ThreadPoolExecutor
    state = {"logprobs": True}

    def one(item):
        body = make_body(item)
        if body.pop("_logprobs", False) and state["logprobs"]:
            status, d, s = chat_once(a, dict(body, logprobs=True))
            if status == 400:  # an endpoint that refuses logprobs: text only from here on
                state["logprobs"] = False
            else:
                return keep(item, status, d, s)
        return keep(item, *chat_once(a, body))

    with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        return list(pool.map(one, items))


def cmd_agree(a):
    prompts = json.loads(Path(a.prompts).read_text())["prompts"][: a.limit or None]

    def body(p):
        b = {"model": a.model, "messages": p["messages"], "max_tokens": a.max_tokens, "temperature": 0,
             "_logprobs": True}
        if p.get("tools"):
            b["tools"] = p["tools"]
        return b

    def keep(p, status, d, s):
        return dict(id=p["id"], category=p["category"], http_status=status, seconds=round(s, 2), **message_parts(d))

    recs = run_all(a, prompts, body, keep)
    ok = sum(r["http_status"] == 200 for r in recs)
    write(a.out, {"model": a.model, "max_tokens": a.max_tokens, "records": recs,
                  "summary": {"requests": len(recs), "ok": ok,
                              "with_tokens": sum(r["tokens"] is not None for r in recs)}})


NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def final_number(text):
    """The answer a reply gives: a \\boxed{} value, else the number after an
    'answer' phrase, else the last number of the content."""
    for pat in (r"\\boxed\{([^}]*)\}", r"####\s*([^\n]+)", r"(?i)answer\s*(?:is|:)?\s*\**\s*([^\n]+)"):
        m = re.findall(pat, text)
        if m:
            nums = NUMBER.findall(m[-1])
            if nums:
                return nums[0]
    nums = NUMBER.findall(text)
    return nums[-1] if nums else None


def same_number(got, want):
    try:
        return abs(float(got.replace(",", "").replace("$", "")) - float(want)) < 1e-6
    except (AttributeError, ValueError):
        return False


def cmd_gsm8k(a):
    items = [json.loads(line) for line in Path(a.data).read_text().splitlines() if line.strip()][: a.limit or None]

    def body(q):
        return {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content":
                q["question"] + "\nSolve it step by step, then give the final answer as a number in \\boxed{}."}]}

    def keep(q, status, d, s):
        m = message_parts(d)
        got = final_number(m["content"]) if m["content"] else None
        return {"index": q["index"], "answer": q["answer"], "got": got, "correct": same_number(got, q["answer"]),
                "finish_reason": m["finish_reason"], "completion_tokens": m["completion_tokens"],
                "http_status": status, "seconds": round(s, 2), "content_tail": m["content"][-200:]}

    recs = run_all(a, items, body, keep)
    n = len(recs)
    correct = sum(r["correct"] for r in recs)
    write(a.out, {"model": a.model, "max_tokens": a.max_tokens, "records": recs,
                  "summary": {"n": n, "correct": correct, "accuracy": round(correct / n, 4) if n else None,
                              "truncated": sum(r["finish_reason"] == "length" for r in recs),
                              "errors": sum(r["http_status"] != 200 for r in recs)}})


def cmd_tools(a):
    cases = json.loads(Path(a.cases).read_text())["cases"]

    def body(c):
        return {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "tools": c["tools"],
                "tool_choice": "auto", "messages": [{"role": "user", "content": c["prompt"]}]}

    def keep(c, status, d, s):
        m = message_parts(d)
        calls = m["tool_calls"]
        args, valid = None, False
        if calls:
            try:
                args = json.loads(calls[0]["arguments"] or "")
                valid = isinstance(args, dict)
            except (TypeError, ValueError):
                pass
        right = bool(calls) and calls[0]["name"] == c["expect"]["name"]
        required = valid and all(k in args for k in c["expect"]["required"])
        return {"id": c["id"], "http_status": status, "calls": [x["name"] for x in calls], "valid_json": valid,
                "right_function": right, "required_present": required, "pass": valid and right and required,
                "finish_reason": m["finish_reason"], "seconds": round(s, 2),
                "markup_in_content": [x for x in MARKUP if x in m["content"]]}

    recs = run_all(a, cases, body, keep)
    write(a.out, {"model": a.model, "records": recs, "summary": {
        "n": len(recs), "pass": sum(r["pass"] for r in recs), "valid_json": sum(r["valid_json"] for r in recs),
        "right_function": sum(r["right_function"] for r in recs),
        "required_present": sum(r["required_present"] for r in recs)}})


def agreement(x, y):
    """Exact-match rate and first-divergence positions of two agree runs, by id."""
    ys = {r["id"]: r for r in y["records"]}
    exact, char_pos, tok_pos, n, by_cat = 0, [], [], 0, {}
    for r in x["records"]:
        s = ys.get(r["id"])
        if not s or r["http_status"] != 200 or s["http_status"] != 200:
            continue
        n += 1
        tx = r["reasoning"] + "\x00" + r["content"] + "\x00" + json.dumps(r["tool_calls"], sort_keys=True)
        ty = s["reasoning"] + "\x00" + s["content"] + "\x00" + json.dumps(s["tool_calls"], sort_keys=True)
        same = tx == ty
        exact += same
        cat = by_cat.setdefault(r["category"], [0, 0])
        cat[0] += same
        cat[1] += 1
        if not same:
            char_pos.append(next((i for i, (p, q) in enumerate(zip(tx, ty)) if p != q), min(len(tx), len(ty))))
            if r["tokens"] is not None and s["tokens"] is not None:
                tok_pos.append(next((i for i, (p, q) in enumerate(zip(r["tokens"], s["tokens"])) if p != q),
                                    min(len(r["tokens"]), len(s["tokens"]))))

    def q(v):
        if not v:
            return None
        v = sorted(v)
        return {"n": len(v), "p25": v[len(v) // 4], "median": v[len(v) // 2], "p75": v[3 * len(v) // 4], "min": v[0]}
    return {"n": n, "exact": exact, "exact_rate": round(exact / n, 4) if n else None,
            "first_divergence_chars": q(char_pos), "first_divergence_tokens": q(tok_pos),
            "by_category": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_cat.items())}}


def cmd_faster(a):
    """Of two variant results, the one with the higher C1 per-stream decode (C8 aggregate breaks a tie)."""
    def score(name):
        d = load_json(Path(a.res) / f"var-{name}.json") or {}
        lv = {x["concurrency"]: x["summary"] for x in (d.get("concurrency") or {}).get("levels") or []}
        return ((lv.get(1, {}).get("per_stream_tps") or {}).get("median") or 0,
                (lv.get(8, {}).get("aggregate_tps") or {}).get("median") or 0)
    print(max(a.names, key=score))


def cmd_experiment(a):
    """experiment.md: the variant tables, the stall test, the accuracy checks."""
    res = Path(a.res)
    cmd_variants(argparse.Namespace(res=a.res, out=a.out, names=a.names))
    lines = [Path(a.out).read_text().replace(
        "# MTP/EAGLE recipe: SGLang launched directly (raw) and through CapyCTL, residency and KV dtype variants",
        "# Qwen3.8-27B on SGLang 0.5.21: checkpoint, drafter and depth variants, direct and through CapyCTL")]
    lines += ["## Long-prompt stall", "",
              "| Variant | Longest gap while the cold prompt prefills, s | Before it, s | Long prompt tokens | Its time to first token, s |",
              "|---|---|---|---|---|"]
    for p in sorted(res.glob("stall-*-t*.json")):
        s = (load_json(p) or {}).get("summary") or {}
        lines.append(f"| {p.stem[len('stall-'):]} | {s.get('max_gap_prefill_s', 'n/a')} | {s.get('max_gap_before_s', 'n/a')}"
                     f" | {s.get('long_prompt_tokens', 'n/a')} | {s.get('long_ttft_s', 'n/a')} |")
    acc = {k: load_json(res / f"{k}.json") for k in ("agree-bf16", "agree-nvfp4", "agree-bf16-repeat", "gsm8k-bf16",
                                                       "gsm8k-nvfp4", "tools-bf16", "tools-nvfp4")}
    lines += ["", "## Accuracy: BF16 LM head against NVFP4 LM head", "",
              "Both through CapyCTL with `residency: restart_only` and the same EAGLE MTP flags, greedy (temperature 0),",
              "thinking as the model's default. The checkpoints may differ beyond the head (the BF16-head checkpoint also",
              "carries calibrated FP8 KV scales; the NVFP4-head one has none and runs FP8 KV unscaled), so a difference is",
              "the checkpoints', not the head's alone. SGLang's greedy decoding is not batch-invariant: the BF16 repeat row",
              "is the same checkpoint against itself.", ""]
    pairs = {}
    if acc["agree-bf16"] and acc["agree-nvfp4"]:
        pairs["BF16 head vs NVFP4 head"] = agreement(acc["agree-bf16"], acc["agree-nvfp4"])
    if acc["agree-bf16"] and acc["agree-bf16-repeat"]:
        pairs["BF16 head vs itself (repeat of the first prompts)"] = agreement(acc["agree-bf16-repeat"], acc["agree-bf16"])
    lines += ["| Token agreement (max_tokens 256) | Prompts | Exact match | First divergence, tokens (p25 / median / p75) | First divergence, characters (median) | By category (exact) |",
              "|---|---|---|---|---|---|"]
    for name, g in pairs.items():
        t, c = g["first_divergence_tokens"], g["first_divergence_chars"]
        tok = f"{t['p25']} / {t['median']} / {t['p75']}" if t else "n/a"
        cats = ", ".join(f"{k} {v}" for k, v in g["by_category"].items())
        lines.append(f"| {name} | {g['n']} | {g['exact']} ({(g['exact_rate'] or 0) * 100:.1f}%)"
                     f" | {tok} | {c['median'] if c else 'n/a'} | {cats} |")
    lines += ["", "| Check | BF16 head | NVFP4 head |", "|---|---|---|"]
    gs = {k: (acc[f"gsm8k-{k}"] or {}).get("summary") or {} for k in ("bf16", "nvfp4")}
    ts = {k: (acc[f"tools-{k}"] or {}).get("summary") or {} for k in ("bf16", "nvfp4")}
    lines.append("| GSM8K test subset, final number in `\\boxed{}` (max_tokens " +
                 f"{(acc['gsm8k-bf16'] or acc['gsm8k-nvfp4'] or {}).get('max_tokens', 'n/a')}) | " +
                 " | ".join(f"{g.get('correct', 'n/a')}/{g.get('n', 'n/a')} ({(g.get('accuracy') or 0) * 100:.1f}%), "
                            f"{g.get('truncated', 'n/a')} cut at max_tokens, {g.get('errors', 'n/a')} errors" for g in gs.values()) + " |")
    for key, label in (("pass", "Tool calls: all three checks"), ("valid_json", "Tool calls: arguments are a JSON object"),
                       ("right_function", "Tool calls: the expected function"),
                       ("required_present", "Tool calls: required arguments present")):
        lines.append(f"| {label} | " + " | ".join(f"{t.get(key, 'n/a')}/{t.get('n', 'n/a')}" for t in ts.values()) + " |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    Path(a.out).with_suffix(".accuracy.json").write_text(json.dumps({"agreement": pairs, "gsm8k": gs, "tools": ts}, indent=1) + "\n")
    print("\n".join(lines[-12:]))


# ---------------------------------------------------------------------------
# The sweet spot (run.sh --sweetspot): one report from one or two hosts' runs
# ---------------------------------------------------------------------------

SWEET_ORDER = ["dflash-8", "dflash-mix-22", "dflash-mix-12", "eagle-22", "eagle-12",
               "eagle-22-d213", "eagle-22-d516", "calib-eagle-22"]
SWEET_MAIN = SWEET_ORDER[:5]


def level_map(bench):
    return {lv["concurrency"]: lv["summary"] for lv in ((bench or {}).get("concurrency") or {}).get("levels") or []}


def point_map(bench):
    return {p["label"]: p["summary"] for p in ((bench or {}).get("context_sweep") or {}).get("points") or []}


def med(s, key):
    return ((s or {}).get(key) or {}).get("median")


def stall_max(paths):
    vals = [((load_json(p) or {}).get("summary") or {}) for p in paths]
    gaps = [v.get("max_gap_prefill_s") for v in vals if v.get("max_gap_prefill_s") is not None]
    return (max(gaps) if gaps else None, next((v.get("long_ttft_s") for v in vals if v.get("long_ttft_s")), None))


def figure_map(path):
    f = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                f.setdefault(parts[0], {}).setdefault(parts[1], parts[2])
    return f


def sweet_rows(host, raw):
    """One row per variant a host ran, from its raw results directory."""
    raw = Path(raw)
    figs = figure_map(raw / "figures.tsv")
    rows = []
    for v in SWEET_ORDER:
        bench = load_json(raw / f"var-{v}.json")
        f = figs.get(v, {})
        if not bench and not f:
            continue
        facts = (load_json(raw / f"facts-var-{v}.json") or {}).get("summary") or {}
        pub = (load_json(raw / f"public-var-{v}.json") or {}).get("settings") or {}
        eff = load_json(raw / f"effective-var-{v}.json")
        gap, long_ttft = stall_max(sorted(raw.glob(f"stall-{v}-t*.json")))
        rows.append({
            "variant": v, "host": host, "source": "this run",
            "request": f.get("request"), "named_request": f.get("named_request"),
            "running": facts.get("max_running_requests"), "kv_tokens": facts.get("max_total_num_tokens"),
            "kv_rendered": pub.get("max_total_tokens"),
            "levels": level_map(bench), "points": point_map(load_json(raw / f"ctx-var-{v}.json")),
            "accept": (facts.get("accept_len") or {}).get("mean"),
            "ready_gib": f.get("ready_gib"), "charge": find_path(eff, "resources.ready.allocations[0].bytes"),
            "start_s": f.get("start_s"), "park_s": f.get("park_s"), "parked_charge": f.get("parked_charge"),
            "parked_provenance": f.get("parked_provenance"), "parked_gib": f.get("parked_gib"),
            "parked_avail_gib": f.get("parked_avail_gib"), "wake_ttft_s": f.get("wake_ttft_s"),
            "wake_ok": f.get("wake_ok"), "stall_gap_s": gap, "stall_long_ttft_s": long_ttft,
            "bench_file": str(raw / f"var-{v}.json") if bench else None,
            "ctx_file": str(raw / f"ctx-var-{v}.json") if (raw / f"ctx-var-{v}.json").exists() else None})
    return rows


def ref_rows(cmp):
    """Rows from the committed earlier runs of this comparison, for reference."""
    cmp = Path(cmp)
    res, exp = cmp / "results", cmp / "results" / "experiment"
    figs1, figs2 = figure_map(res / "figures.tsv"), figure_map(exp / "figures.tsv")
    expj = (load_json(exp / "experiment.json") or {}).get("variants") or {}

    def accept1(side):
        vals = [((load_json(p) or {}).get("summary") or {}).get("accept_len", {}).get("mean")
                for p in sorted(res.glob(f"facts-conc-{side}-pass*.json"))]
        vals = [v for v in vals if v is not None]
        return statistics.mean(vals) if vals else None

    out = []
    for label, bench_path, side in (("run 1: DFlash2 catalog, 8 running", cmp / "bench/results/dflash2.json", "a"),
                                    ("run 1: MTP/EAGLE, BF16-head checkpoint, 79 GiB", cmp / "bench/results/mtp-eagle.json", "b")):
        bench = load_json(bench_path)
        if not bench:
            continue
        gap, long_ttft = stall_max(sorted(res.glob(f"stall-{side}-t*.json")))
        out.append({"variant": label, "host": "a", "source": "2026-10-14", "levels": level_map(bench),
                    "points": point_map(bench), "accept": accept1(side), "stall_gap_s": gap,
                    "stall_long_ttft_s": long_ttft, "ready_gib": figs1.get(side, {}).get("ready_gib")})
    for label, name in (("run 2: raw SGLang, BF16-head checkpoint, no CapyCTL", "raw"),
                        ("run 2: MTP/EAGLE, NVFP4 head, restart_only, 84 GiB", "nvfp4-restart")):
        bench = load_json(cmp / f"bench-experiment/results/var-{name}.json")
        if not bench:
            continue
        gap, long_ttft = stall_max(sorted(exp.glob(f"stall-{name}-t*.json")))
        v = expj.get(name) or {}
        out.append({"variant": label, "host": "a", "source": "2026-10-15", "levels": level_map(bench),
                    "points": {}, "accept": v.get("accept_len"), "kv_tokens": v.get("engine_kv"),
                    "running": v.get("engine_running"), "stall_gap_s": gap, "stall_long_ttft_s": long_ttft,
                    "ready_gib": figs2.get(name, {}).get("ready_gib")})
    return out


def host_info(raw):
    info = {}
    for name in ("versions.txt", "ckpt.txt"):
        p = Path(raw) / name
        if p.exists():
            for line in p.read_text().splitlines():
                k, _, v = line.partition("=")
                info[k] = v
    return info


def cmd_sweetspot(a):
    out = Path(a.out)
    runs = [r.split("=", 1) for r in a.run]
    rows = [row for host, raw in runs for row in sweet_rows(host, raw)]
    refs = ref_rows(a.ref) if a.ref else []
    hosts = {host: host_info(raw) for host, raw in runs}
    by = {(r["variant"]): r for r in rows}

    def cell(v, fmt="{:.1f}"):
        return num(v, fmt) if v not in (None, "", "None") else "n/a"

    def lv(r, c, key):
        return med(r["levels"].get(c), key) if key != "ttft_p50_s" else (r["levels"].get(c) or {}).get(key)

    lines = ["# Qwen3.8-27B on SGLang 0.5.21 through CapyCTL: the sweet spot between DFlash2 and MTP/EAGLE", "",
             "Every variant runs the catalog checkpoint `nvidia/Qwen3.8-27B-NVFP4@482ca0f3` (NVFP4 head), FP8 KV,",
             "a 262144-token context, typed parsers and CapyCTL's default residency (`deep`), one at a time,",
             "each deleted before the next. Short prompts: C1 to C22, 3 rounds, `max_tokens` 512; context:",
             "2k, 32k, 128k, 2 runs, `max_tokens` 128; one stall trial; one park and wake by request.", ""]
    lines += ["## Hosts", "", "| Host | Checkpoint | SGLang | torch | FlashInfer | Driver | Kernel |", "|---|---|---|---|---|---|---|"]
    for host, info in hosts.items():
        lines.append(f"| {host} | {info.get('source', 'n/a')} | {info.get('sglang', 'n/a')} | {info.get('torch', 'n/a')}"
                     f" | {info.get('flashinfer', 'n/a')} | {info.get('driver', 'n/a')} | {info.get('kernel', 'n/a')} |")
    # Host comparability: the calibration (eagle-22 at C1 and C8 on the other
    # host) against eagle-22's own C1 and C8.
    cal, ref = by.get("calib-eagle-22"), by.get("eagle-22")
    approximate = False
    if cal and ref and cal["host"] != ref["host"]:
        pairs = [("C1 per-stream decode", lv(cal, 1, "per_stream_tps"), lv(ref, 1, "per_stream_tps")),
                 ("C8 aggregate", lv(cal, 8, "aggregate_tps"), lv(ref, 8, "aggregate_tps"))]
        lines += ["", "## Host comparability", "",
                  f"eagle-22 ran on host {ref['host']}; host {cal['host']} ran the same deployment first as a calibration.", "",
                  f"| Figure | Host {cal['host']} (calibration) | Host {ref['host']} (eagle-22) | Difference |", "|---|---|---|---|"]
        worst = 0.0
        for name, x, y in pairs:
            d = (x / y - 1) * 100 if x and y else None
            worst = max(worst, abs(d)) if d is not None else worst
            lines.append(f"| {name}, tok/s | {cell(x)} | {cell(y)} | {f'{d:+.1f}%' if d is not None else 'n/a'} |")
        approximate = worst > 3.0
        lines += ["", (f"The hosts differ by up to {worst:.1f}%: comparisons between variants run on different hosts "
                       "are approximate." if approximate else
                       f"The hosts agree within {worst:.1f}%: variants from both hosts compare directly.")]
    elif cal or len(hosts) > 1:
        lines += ["", "## Host comparability", "", "No calibration pair (eagle-22 on one host and calib on the other): "
                  "comparisons across hosts are approximate."]
        approximate = len(hosts) > 1
    main = [by[v] for v in SWEET_MAIN if v in by]
    allrows = main + refs
    cs = [1, 4, 8, 16, 22]
    lines += ["", "## Short prompts", "",
              "| Variant | Host | Request | Running | Engine KV tokens | " + " | ".join(f"C{c} aggregate" for c in cs) +
              " | Per stream C1 / C22 | TTFT p50 C1 / C22 | Accept length |",
              "|---|---|---|---|---|" + "---|" * len(cs) + "---|---|---|"]
    for r in allrows:
        req = r.get("request") or "n/a"
        if r.get("named_request") not in (None, "none"):
            req += f" (named {r['named_request']} GiB)"
        lines.append(f"| {r['variant']} | {r['host']} | {req} | {r.get('running') or 'n/a'} | {r.get('kv_tokens') or 'n/a'} | " +
                     " | ".join(cell(lv(r, c, "aggregate_tps")) for c in cs) +
                     f" | {cell(lv(r, 1, 'per_stream_tps'))} / {cell(lv(r, 22, 'per_stream_tps'))}"
                     f" | {cell(lv(r, 1, 'ttft_p50_s'), '{:.2f}')} / {cell(lv(r, 22, 'ttft_p50_s'), '{:.2f}')} s"
                     f" | {cell(r.get('accept'), '{:.2f}')} |")
    lines += ["", "## Context", "", "| Variant | Host | 2k decode, TTFT | 32k decode, TTFT | 128k decode, TTFT |", "|---|---|---|---|---|"]
    for r in allrows:
        if not r.get("points"):
            continue
        cells = [f"{cell(med(r['points'].get(p), 'decode_tps'))} tok/s, {cell(med(r['points'].get(p), 'ttft_s'), '{:.1f}')} s"
                 for p in ("2k", "32k", "128k")]
        lines.append(f"| {r['variant']} | {r['host']} | " + " | ".join(cells) + " |")
    lines += ["", "## Memory, park and wake, stall", "",
              "| Variant | Host | Ready footprint (MemAvailable drop) | CapyCTL Ready charge | Start | Park | Parked charge"
              " | MemAvailable while parked (drop) | Wake, first token | Wake answer | Stall: longest gap, cold prompt TTFT |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in allrows:
        prov = f" ({r['parked_provenance']})" if r.get("parked_provenance") else ""
        lines.append(f"| {r['variant']} | {r['host']} | {cell(r.get('ready_gib'))} GiB | {gib(r.get('charge'))}"
                     f" | {cell(r.get('start_s'), '{:.0f}')} s | {cell(r.get('park_s'))} s | {gib(r.get('parked_charge'))}{prov}"
                     f" | {cell(r.get('parked_avail_gib'))} GiB ({cell(r.get('parked_gib'))}) | {cell(r.get('wake_ttft_s'), '{:.2f}')} s"
                     f" | {'correct' if r.get('wake_ok') == 'True' else ('wrong' if r.get('wake_ok') else 'n/a')}"
                     f" | {cell(r.get('stall_gap_s'))} s, {cell(r.get('stall_long_ttft_s'))} s |")
    sweeps = [by[v] for v in ("eagle-22", "eagle-22-d213", "eagle-22-d516") if v in by]
    if len(sweeps) > 1:
        lines += ["", "## Draft depth on eagle-22", "", "| Steps / top-k / draft tokens | C1 per stream | C8 aggregate | Accept length |",
                  "|---|---|---|---|"]
        for r in sweeps:
            depth = {"eagle-22": "3 / 1 / 4 (catalog)", "eagle-22-d213": "2 / 1 / 3", "eagle-22-d516": "5 / 1 / 6"}[r["variant"]]
            lines.append(f"| {depth} | {cell(lv(r, 1, 'per_stream_tps'))} | {cell(lv(r, 8, 'aggregate_tps'))} | {cell(r.get('accept'), '{:.2f}')} |")
    # Which recipe for which workload: the best of the five main variants per figure.
    def best(key_fn, low=False):
        vals = [(key_fn(r), r) for r in main]
        vals = [(x, r) for x, r in vals if x is not None]
        if not vals:
            return None
        x, r = (min if low else max)(vals, key=lambda t: t[0])
        return r, x

    def ctxd(p):
        return lambda r: med(r["points"].get(p), "decode_tps")

    picks = [("One user at a time (C1 per-stream decode)", best(lambda r: lv(r, 1, "per_stream_tps")), "{:.1f} tok/s"),
             ("A few users (C8 aggregate)", best(lambda r: lv(r, 8, "aggregate_tps")), "{:.1f} tok/s"),
             ("Many users (C22 aggregate)", best(lambda r: lv(r, 22, "aggregate_tps")), "{:.1f} tok/s"),
             ("Interactive under load (C22 TTFT p50)", best(lambda r: lv(r, 22, "ttft_p50_s"), low=True), "{:.2f} s"),
             ("Long context (128k decode)", best(ctxd("128k")), "{:.1f} tok/s"),
             ("Long prompts beside running streams (stall gap)", best(lambda r: r.get("stall_gap_s"), low=True), "{:.1f} s"),
             ("Least memory (CapyCTL Ready charge)", best(lambda r: float(r["charge"]) if r.get("charge") else None, low=True), None),
             ("Fastest wake from park (first token)", best(lambda r: float(r["wake_ttft_s"]) if r.get("wake_ttft_s") not in (None, "", "None") else None, low=True), "{:.2f} s")]
    lines += ["", "## Which recipe for which workload", "",
              "The best of the five variants on each figure, from the tables above."]
    if approximate:
        lines.append("The hosts are not within 3%: a pick that beats a variant from the other host by less than the "
                     "host difference is not established.")
    lines.append("")
    for name, b, fmt in picks:
        if not b:
            lines.append(f"- {name}: n/a")
            continue
        r, x = b
        val = gib(x) if fmt is None else fmt.format(x)
        lines.append(f"- {name}: **{r['variant']}** ({val}, host {r['host']})")
    lines += ["", "## Notes", "",
              "- The first CapyCTL start on each host built the kernels (shared cache per host), so its start time",
              "  includes the build; later starts are warm.",
              "- DFlash2 runs at most its running count (8, 12 or 22); at more streams the rest wait in CapyCTL.",
              "- Reference rows: run 1 (2026-10-14) ran 2 interleaved passes of 5 rounds at 32768 context for short",
              "  prompts; run 2 (2026-10-15) ran C1, C8 and C22 only, 3 rounds, `restart_only`.",
              "- One machine per variant, one pass; medians of 3 rounds (short prompts) and 2 runs (context)."]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    # A capyctl-bench series per main variant (short prompts and context in one file).
    series = out.parent / "series"
    series.mkdir(exist_ok=True)
    for r in main:
        bench = load_json(r["bench_file"]) if r.get("bench_file") else None
        if not bench:
            continue
        ctx = load_json(r["ctx_file"]) if r.get("ctx_file") else None
        bench["context_sweep"] = (ctx or {}).get("context_sweep")
        bench["label"] = f"{r['variant']} (host {r['host']})"
        (series / f"{r['variant']}.json").write_text(json.dumps(bench) + "\n")
    keep = ("variant", "host", "source", "request", "named_request", "running", "kv_tokens", "kv_rendered", "accept",
            "ready_gib", "charge", "start_s", "park_s", "parked_charge", "parked_provenance", "parked_gib",
            "parked_avail_gib", "wake_ttft_s", "wake_ok", "stall_gap_s", "stall_long_ttft_s")
    summary = [dict({k: r.get(k) for k in keep},
                    levels={c: {"aggregate_tps": med(s, "aggregate_tps"), "per_stream_tps": med(s, "per_stream_tps"),
                                "ttft_p50_s": s.get("ttft_p50_s")} for c, s in r["levels"].items()},
                    points={p: {"decode_tps": med(s, "decode_tps"), "ttft_s": med(s, "ttft_s")} for p, s in r["points"].items()})
               for r in rows + refs]
    out.with_suffix(".json").write_text(json.dumps({"hosts": hosts, "approximate_across_hosts": approximate,
                                                     "rows": summary}, indent=1) + "\n")
    print("\n".join(lines))



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


RUN1_BEGIN = "<!-- b18.py readme: run 1 begins -->"
RUN1_END = "<!-- b18.py readme: run 1 ends -->"


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

    def known(prefix, value):
        return "" if value is None else f"{prefix}{value}"

    def peak(side):
        d = load_json(raw / f"inspect-{side}.json") or {}
        return ((d.get("startup") or {}).get("bytes")) if (d.get("startup") or {}).get("provenance") == "measured" else None

    parks = {}
    lpath = raw / "lifecycle.tsv"
    if lpath.exists():
        for line in lpath.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[0].startswith("park "):
                parks[parts[0][5:]] = parts[1]

    def park_cell(side):
        if parks.get(side) != "pass":
            return "n/a (the park was refused)"
        return (f"{num(first(f, side, 'park_s'))} s, {gib(first(f, side, 'parked_charge'))}, "
                f"{num(first(f, side, 'parked_gib'))} GiB")

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
    text = f"""## Run 1, {settings.get('date', 'n/a')}: the two recipes

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
raised to {settings.get('managed_limit', '100GiB')}, the free reserve lowered to
{settings.get('free_reserve', '10GiB')} (the limit and the reserve together must fit the
machine's memory) and the parked limit raised to {settings.get('parked_limit', '45GiB')}:

```bash
capyctl start standalone --listen 127.0.0.1:{settings.get('port', 'PORT')} \\
  --set host.resource_policy.memory.system.managed_limit={settings.get('managed_limit', '100GiB')} \\
  --set host.resource_policy.memory.system.free_reserve={settings.get('free_reserve', '10GiB')} \\
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
| Chunked prefill, max prefill | {fact('a', 'chunked_prefill_size') or 'n/a'}, {fact('a', 'max_prefill_tokens') or 'n/a'} | {fact('b', 'chunked_prefill_size') or 'n/a'}, {fact('b', 'max_prefill_tokens') or 'n/a'}{known('; mixed chunks ', fact('b', 'enable_mixed_chunk'))} |
| State cache | sized by CapyCTL for 8 requests | {fact('b', 'max_mamba_cache_size') or '110'} slots{known(', `mamba_ssm_dtype` ', fact('b', 'mamba_ssm_dtype'))}{known(', radix strategy ', fact('b', 'mamba_radix_cache_strategy'))} |
| Memory request{', rendered `mem_fraction_static`' if fact('a', 'mem_fraction_static') or fact('b', 'mem_fraction_static') else ''} | 65 GiB{known(', ', fact('a', 'mem_fraction_static'))} | 79 GiB{known(', ', fact('b', 'mem_fraction_static'))} |
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
| CapyCTL measured startup peak | {gib(peak('a'))} | {gib(peak('b'))} |
| Park, parked charge, parked footprint | {park_cell('a')} | {park_cell('b')} |
| Wake by request, to the answer | {num(first(f, 'a', 'wake_s')) + ' s' if parks.get('a') == 'pass' else 'n/a'} | {num(first(f, 'b', 'wake_s')) + ' s' if parks.get('b') == 'pass' else 'n/a'} |
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
"""
    # Only the run 1 section is generated: the summary, the later runs and the
    # caveats around it are written by hand and kept as they are.
    readme = Path(a.readme)
    old = readme.read_text() if readme.exists() else ""
    i, j = old.find(RUN1_BEGIN), old.find(RUN1_END)
    if i < 0 or j < i:
        sys.exit(f"{readme}: no {RUN1_BEGIN} ... {RUN1_END} around the run 1 section; "
                 "refusing to overwrite the hand-written sections")
    readme.write_text(old[:i + len(RUN1_BEGIN)] + "\n" + text + "\n" + old[j:])
    print(f"wrote the run 1 section of {readme}")


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
    s = sub.add_parser("serverargs")
    s.add_argument("logs", nargs="+")
    s.add_argument("--out", required=True)
    s = sub.add_parser("public")
    s.add_argument("--name", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("effective")
    s.add_argument("--cred", required=True)
    s.add_argument("--mport", type=int, required=True)
    s.add_argument("--id", required=True)
    s.add_argument("--out", required=True)
    s = sub.add_parser("variants")
    s.add_argument("res")
    s.add_argument("out")
    s.add_argument("--names", default="raw,deep,restart,restart-autokv")
    for name, extra in (("agree", ("--prompts", "--max-tokens", "--limit")), ("gsm8k", ("--data", "--max-tokens", "--limit")),
                        ("tools", ("--cases", "--max-tokens"))):
        s = sub.add_parser(name)
        s.add_argument("--cred", help="credentials file; omit for an endpoint without a key")
        s.add_argument("--endpoint", required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--concurrency", type=int, default=8)
        for opt in extra:
            if opt in ("--max-tokens", "--limit"):
                s.add_argument(opt, type=int, default={"--max-tokens": 256, "--limit": 0}[opt])
            else:
                s.add_argument(opt, required=True)
    s = sub.add_parser("faster")
    s.add_argument("res")
    s.add_argument("names", nargs=2)
    s = sub.add_parser("experiment")
    s.add_argument("res")
    s.add_argument("out")
    s.add_argument("--names", required=True)
    s = sub.add_parser("sweetspot")
    s.add_argument("out", help="sweetspot.md to write (sweetspot.json and series/ beside it)")
    s.add_argument("--run", action="append", required=True, metavar="HOST=DIR",
                   help="a host's raw results directory (repeatable)")
    s.add_argument("--ref", help="this comparison directory, for the run 1 and run 2 reference rows")
    a = p.parse_args()
    {"toolcall": cmd_toolcall, "reasoning": cmd_reasoning, "stall": cmd_stall, "chat": cmd_chat,
     "facts": cmd_facts, "load": cmd_load, "get": cmd_get, "merge": cmd_merge,
     "readme": cmd_readme, "serverargs": cmd_serverargs, "public": cmd_public,
     "effective": cmd_effective, "variants": cmd_variants, "agree": cmd_agree, "gsm8k": cmd_gsm8k,
     "tools": cmd_tools, "faster": cmd_faster, "experiment": cmd_experiment,
     "sweetspot": cmd_sweetspot}[a.cmd](a)


if __name__ == "__main__":
    main()
