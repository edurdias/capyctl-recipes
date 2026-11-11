#!/usr/bin/env python3
"""Fan-out throughput experiment through the CapyCTL inference endpoint.

Modes:
  fanout  one task split into 8 buckets, sent as 8 concurrent streams, merged into one page
  whole   the same task as one stream (all 8 buckets in one reply)
  batch   N unrelated coding prompts at once (plain batching baseline)

Thinking off, temperature 0, streamed with usage. The credential is read from the
role's credentials file and never printed.
"""
import argparse
import http.client
import json
import statistics
import threading
import time
from pathlib import Path

TASK = "build me a html three.js website for harry potter, it should be a masterpiece."

CONTRACT = """You are writing part of ONE single-file three.js website for this request:
"{task}"

The page is assembled from 8 buckets of JavaScript, concatenated in order 1..8 into a
single <script type="module"> that ALREADY contains, before bucket 1:
  import * as THREE from 'three';
  import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 5000);
  const renderer = new THREE.WebGLRenderer({{ antialias: true }});  // already in document.body
  const controls = new OrbitControls(camera, renderer.domElement);
  const clock = new THREE.Clock();
  const updaters = [];  // functions (dt, t) => void, called every frame before rendering
  // the render loop is already running and calls every function in `updaters`

Contract shared by every bucket:
- Use `scene`, `camera`, `renderer`, `controls`, `clock` and `updaters` as they are;
  never redeclare them. Put per-frame animation into `updaters`.
- Prefix every top-level name you declare with b<N>_ (for example b3_lake).
- No import statements, no HTML, no markdown fences, no explanations: JavaScript only.
- World layout: ground at y=0; the castle centred at the origin, about 60 units wide;
  the lake south of it (positive z); the forest to the west (negative x); the
  Quidditch pitch to the east (positive x, around x=120).

The buckets:
1. Camera placement and controls tuning, renderer settings (tone mapping, shadows), sky colour, fog, lights (moonlight, ambient, warm castle glow), resize handler.
2. Hogwarts castle geometry: great hall, towers, conical spires, battlements, walls, glowing windows.
3. Terrain: hills and mountains, the Black Lake with animated water, the Forbidden Forest of stylised trees.
4. Night sky: starfield, a large moon, slowly drifting clouds.
5. Quidditch pitch: oval field, three goal hoops at each end, stands with house colours, a Golden Snitch flying an animated path.
6. Magic particles: floating candles near the castle, sparkles, fireflies over the forest.
7. The Hogwarts Express: a steam train looping on a track across a stone viaduct, with animated steam.
8. HTML overlay built with DOM APIs: title "Hogwarts", four house buttons (Gryffindor, Slytherin, Ravenclaw, Hufflepuff) that tint the lights and fog to the house colour, and a short info panel.
"""

LANE = "Write bucket #{n} only: {title}"
WHOLE = "Write all 8 buckets in order, each starting with a comment line `// Bucket #N`."

BATCH_PROMPTS = [
    "Write a Python LRU cache class with get/put in O(1), with type hints and doctests.",
    "Write a Rust function that parses an INI file into a HashMap<String, HashMap<String, String>>, with tests.",
    "Write a TypeScript debounce and throttle utility with full typings and examples.",
    "Write a Go HTTP server with a /health and a JSON /items CRUD API kept in memory, with a mutex.",
    "Write a SQL schema for a library (books, members, loans) and five useful queries with comments.",
    "Write a bash script that rotates log files older than 7 days into gzip archives, with safe quoting.",
    "Write a C function that reverses a singly linked list iteratively and recursively, with a test main.",
    "Write a Python asyncio web crawler limited to 5 concurrent requests that respects robots.txt.",
]


def bucket_titles():
    lines = CONTRACT.split("The buckets:\n", 1)[1].strip().splitlines()
    return [line.split(". ", 1)[1] for line in lines]


def credential(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)["api_key"]
    except (ValueError, KeyError):
        for line in text.splitlines():
            if line.strip().startswith("api_key"):
                return line.split(":", 1)[1].strip().strip('"')
    raise SystemExit("no api_key in the credentials file")


class Lane:
    def __init__(self, idx, messages, max_tokens):
        self.idx, self.messages, self.max_tokens = idx, messages, max_tokens
        self.sent = self.first = self.end = None
        self.tokens = 0
        self.prompt_tokens = 0
        self.text = []
        self.finish = None
        self.error = None


def run_lane(args, key, lane):
    body = {
        "model": args.model, "temperature": 0, "max_tokens": lane.max_tokens,
        "messages": lane.messages, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    host, port = args.endpoint.rsplit(":", 1)
    try:
        conn = http.client.HTTPConnection(host, int(port), timeout=3600)
        lane.sent = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        resp = conn.getresponse()
        if resp.status != 200:
            lane.error = f"HTTP {resp.status}: {resp.read()[:300]!r}"
            return
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    if lane.first is None:
                        lane.first = time.perf_counter()
                    lane.text.append(piece)
                if choice.get("finish_reason"):
                    lane.finish = choice["finish_reason"]
            usage = chunk.get("usage")
            if usage:
                lane.tokens = usage.get("completion_tokens", 0)
                lane.prompt_tokens = usage.get("prompt_tokens", 0)
        lane.end = time.perf_counter()
        conn.close()
    except (OSError, http.client.HTTPException, ValueError) as error:
        lane.error = f"{type(error).__name__}: {error}"


def run_round(args, key, lanes):
    threads = [threading.Thread(target=run_lane, args=(args, key, lane)) for lane in lanes]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    errors = [lane.error or f"lane {lane.idx + 1}: no content (finish {lane.finish})"
              for lane in lanes if lane.error or lane.first is None or lane.end is None]
    if errors:
        return {"errors": errors}
    ends = sorted(lane.end - start for lane in lanes)
    firsts = sorted(lane.first - start for lane in lanes)
    total = sum(lane.tokens for lane in lanes)
    wall = ends[-1]
    decode_window = ends[-1] - firsts[0]
    per_lane = [lane.tokens / (lane.end - lane.first) for lane in lanes if lane.end > lane.first]
    return {
        "lanes": len(lanes),
        "tokens": total,
        "prompt_tokens": sum(lane.prompt_tokens for lane in lanes),
        "wall_s": round(wall, 3),
        "e2e_tok_s": round(total / wall, 1),
        # From the first streamed token of any lane to the last lane's end.
        "decode_tok_s": round(total / decode_window, 1),
        "ttft_median_s": round(statistics.median(firsts), 3),
        "ttft_max_s": round(firsts[-1], 3),
        # How long the slowest lane ran past the median lane's end.
        "tail_s": round(ends[-1] - statistics.median(ends), 3),
        "lane_tokens": [lane.tokens for lane in lanes],
        "lane_end_s": [round(lane.end - start, 2) for lane in lanes],
        "per_lane_tok_s_median": round(statistics.median(per_lane), 1),
        "finish": [lane.finish for lane in lanes],
    }


def strip_fences(text):
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


def page(js):
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Hogwarts</title>
<style>html,body{{margin:0;height:100%;overflow:hidden;background:#000}}</style>
<script type="importmap">{{"imports":{{"three":"https://unpkg.com/three@0.160.0/build/three.module.js",
"three/addons/":"https://unpkg.com/three@0.160.0/examples/jsm/"}}}}</script>
</head><body>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 5000);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
const clock = new THREE.Clock();
const updaters = [];
renderer.setAnimationLoop(() => {{
  const dt = clock.getDelta(), t = clock.elapsedTime;
  for (const f of updaters) f(dt, t);
  controls.update();
  renderer.render(scene, camera);
}});
{js}
</script>
</body></html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fanout", "whole", "batch"])
    parser.add_argument("--endpoint", default="127.0.0.1:18743")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cred", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--streams", type=int, default=8, help="batch mode: how many prompts at once")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    key = credential(args.cred)
    args.out.mkdir(parents=True, exist_ok=True)
    system = CONTRACT.format(task=TASK)
    titles = bucket_titles()
    results = []
    for rnd in range(1, args.rounds + 1):
        if args.mode == "fanout":
            lanes = [Lane(i, [{"role": "system", "content": system},
                              {"role": "user", "content": LANE.format(n=i + 1, title=t)}], args.max_tokens)
                     for i, t in enumerate(titles)]
        elif args.mode == "whole":
            lanes = [Lane(0, [{"role": "system", "content": system},
                              {"role": "user", "content": WHOLE}], args.max_tokens)]
        else:
            prompts = BATCH_PROMPTS[: args.streams]
            lanes = [Lane(i, [{"role": "user", "content": p}], args.max_tokens) for i, p in enumerate(prompts)]
        result = run_round(args, key, lanes)
        result.update(mode=args.mode, round=rnd)
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.mode in ("fanout", "whole") and "errors" not in result:
            js = "\n\n".join(f"// ---- bucket #{lane.idx + 1} ----\n" + strip_fences("".join(lane.text))
                             if args.mode == "fanout" else strip_fences("".join(lane.text)) for lane in lanes)
            (args.out / f"{args.mode}-r{rnd}.html").write_text(page(js))
            (args.out / f"{args.mode}-r{rnd}.js").write_text(js)
    (args.out / f"{args.mode}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))


if __name__ == "__main__":
    main()
