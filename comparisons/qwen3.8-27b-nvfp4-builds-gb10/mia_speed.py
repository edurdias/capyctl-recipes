#!/usr/bin/env python3
"""Decode speed of a deployment whose streamed replies carry no token count.

MiaAI-Lab's exllamav3 server streams without a usage chunk, so capyctl-bench cannot count its
tokens (its context sweep stops at calibration, and a text chunk may hold several tokens), and
CapyCTL, which collects a non-streamed reply from the engine's stream, has none to pass on either.
This sends capyctl-bench's 8 prompts through the CapyCTL endpoint, streamed, temperature 0, and
counts each reply's tokens by encoding its text (reasoning, then content) with the checkpoint's
own tokenizer (Qwen3.8's, the same for every build compared here). Encoding a reply's text again
can differ from the generated ids by a token or two around the reasoning markers; against 512
tokens that is under 1%.

Decode is (completion tokens - 1) / (stream end - first chunk), as capyctl-bench's. At 4 and 8
concurrent streams it reports total completion tokens over the round's wall time.

usage: mia_speed.py --endpoint http://127.0.0.1:18543/v1 --cred <credentials> --model <name>
                    --prompts <capyctl-bench prompts.json> --tokenizer <checkpoint>/tokenizer.json
                    --out <json> [--max-tokens 512] [--concurrency 1,4,8] [--rounds 3]
"""

import argparse
import http.client
import json
import re
import statistics as st
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def key(path):
    """The `api_key: <value>` line of the standalone credentials file; never printed."""
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^\s*api_key\s*:\s*(\S+)\s*$", line)
        if m:
            return m.group(1)
    raise SystemExit("no api_key line in the credentials file")


def request(a, body):
    """One streamed request: (seconds to the first text chunk, total seconds, reply text)."""
    host, port, base = re.match(r"^http://([^/:]+):(\d+)(/.*)$", a.endpoint.rstrip("/")).groups()
    conn = http.client.HTTPConnection(host, int(port), timeout=1800)
    t0 = time.perf_counter()
    first, reasoning, content = None, [], []
    conn.request("POST", base + "/chat/completions", body=json.dumps(dict(body, stream=True)),
                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data: "):
            continue
        if line == b"data: [DONE]":
            break
        delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            content.append(delta["content"])
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.perf_counter() - t0
    conn.close()
    return first, time.perf_counter() - t0, "".join(reasoning) + "".join(content)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--endpoint", required=True)
    p.add_argument("--cred", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--tokenizer", required=True, help="the checkpoint's tokenizer.json")
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", default="1,4,8")
    p.add_argument("--rounds", type=int, default=3)
    a = p.parse_args()
    a.key = key(a.cred)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids)
    prompts = json.loads(Path(a.prompts).read_text())["prompts"]
    body = lambda pr: {"model": a.model, "max_tokens": a.max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": pr["prompt"]}]}
    request(a, body(prompts[0]))  # warm-up, not recorded

    out = {"model": a.model, "max_tokens": a.max_tokens, "token_count": "tokenizer", "levels": []}
    for c in [int(x) for x in a.concurrency.split(",")]:
        rounds = []
        for r in range(a.rounds):
            batch = [prompts[(r * c + i) % len(prompts)] for i in range(c)]
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as pool:
                res = list(pool.map(lambda pr: request(a, body(pr)), batch))
            wall = time.perf_counter() - t0
            reqs = []
            for pr, (f, t, text) in zip(batch, res):
                n = count(text)
                reqs.append({"prompt_id": pr["id"], "ttft_s": f, "total_s": t, "completion_tokens": n,
                             "decode_tps": (n - 1) / (t - f) if f is not None and t > f else None})
            rounds.append({"wall_s": wall, "aggregate_tps": sum(q["completion_tokens"] for q in reqs) / wall,
                           "requests": reqs})
            print(f"concurrency {c} round {r + 1}/{a.rounds}: {rounds[-1]['aggregate_tps']:.1f} tok/s aggregate",
                  flush=True)
        per = [q["decode_tps"] for rd in rounds for q in rd["requests"] if q["decode_tps"]]
        out["levels"].append({"concurrency": c, "rounds": rounds,
                              "median_aggregate_tps": st.median(rd["aggregate_tps"] for rd in rounds),
                              "median_decode_tps": st.median(per) if per else None,
                              "median_ttft_s": st.median(q["ttft_s"] for rd in rounds for q in rd["requests"])})
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
