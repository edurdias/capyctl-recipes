"""A fake OpenAI-compatible streaming server with deterministic pacing.

Used by the unit tests and to render sample reports without a GPU:

    python3 tests/fake_server.py --port 18443 --token-delay 0.004

Prompt tokens are the whitespace-separated words of the messages plus 4.
Time to first token is `ttft_base + prompt_tokens * prefill_per_token`;
each later chunk waits `token_delay * (1 + slowdown * (active - 1))`, where
`active` is the number of streams in flight (times
`1 + prompt_tokens / 1000 * decode_per_ktoken`), and carries `chunk_tokens`
tokens. Every stream carries a top-level `"capyctl_bench_sample": true`,
so capyctl-bench marks the results as sample data (meta.sample) and every
report rendered from them shows a "SAMPLE DATA" banner. GET /debug/memory prints a fake GiB figure that grows with the
prompt tokens in flight.

POST /v1/systemone answers decisions (the System One API): input tokens are the words of the
state and of the questions plus 8; each answer waits `decision_base + input_tokens *
decision_per_token`, times `1 + slowdown * (active - 1)`, and a fifth of the per-token part for a
state the server has answered before (a prefix cache). A choice picks the first option whose key
is named in the state (else one by hash); a noul is 0.9 when the state says "The answer is yes",
0.2 when it says "The answer is no" (else by hash); a score puts 0.6 on a level by hash. A request
without `instructions` gets HTTP 400, as on llama.cpp.

Small fake copies of the accuracy sources: GET /rows serves `fancyzhx/ag_news` (every fifth row
names the wrong section), `google/boolq` (every fourth row's passage gives the wrong answer) and
`LocalLLaMA/typed-decisions` in the shape of the Hugging Face datasets-server rows API;
GET /github/tree/<commit> and /github/raw/<commit>/<path> serve a three-tool API-Bank checkout
in the shape of the GitHub tree API and raw files; GET /gpqa/dataset.zip serves a six-question
GPQA Diamond zip (not encrypted: a password is ignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeConfig:
    def __init__(self, ttft_base=0.004, prefill_per_token=2e-6, token_delay=0.002,
                 slowdown=0.15, chunk_tokens=1, extras=True, reasoning_tokens=0,
                 api_key=None, base_gib=10.0, gib_per_ktoken=0.05, decode_per_ktoken=0.0, cut_after=None,
                 hidden_tokens=0, decision_base=0.004, decision_per_token=1e-6):
        self.ttft_base = ttft_base
        self.prefill_per_token = prefill_per_token
        self.token_delay = token_delay
        self.slowdown = slowdown
        self.chunk_tokens = chunk_tokens
        self.extras = extras
        self.reasoning_tokens = reasoning_tokens
        self.api_key = api_key
        self.base_gib = base_gib
        self.gib_per_ktoken = gib_per_ktoken
        self.decode_per_ktoken = decode_per_ktoken
        # End the stream after this many token chunks, with no finish_reason, usage or [DONE]
        # (a proxy closing an idle stream); 0 sends only the role chunk first.
        self.cut_after = cut_after
        # The first this many tokens stream with an empty delta, as when the engine's parser holds
        # back a format token (gpt-oss channel markers on SGLang).
        self.hidden_tokens = hidden_tokens
        self.decision_base = decision_base
        self.decision_per_token = decision_per_token


class FakeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, config: FakeConfig):
        super().__init__(addr, Handler)
        self.config = config
        self.lock = threading.Lock()
        self.active = 0
        self.active_prompt_tokens = 0
        self.requests = []
        self.rows_requests = 0
        self.source_requests = 0
        self.seen_states: set = set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


def prompt_token_count(messages) -> int:
    words = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            words += len(content.split())
    return words + 4


AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def fake_rows(dataset: str) -> tuple[list[dict], list[dict]]:
    """(features, rows) of a fake accuracy set."""
    if dataset == "fancyzhx/ag_news":
        rows = []
        for i in range(60):
            label = i % 4
            named = label if i % 5 else (label + 1) % 4
            rows.append({"text": f"Story {i}. The {AG_LABELS[named]} desk filed this report today.", "label": label})
        return [{"feature_idx": 0, "name": "text", "type": {"dtype": "string", "_type": "Value"}},
                {"feature_idx": 1, "name": "label", "type": {"names": AG_LABELS, "_type": "ClassLabel"}}], rows
    if dataset == "google/boolq":
        rows = []
        for i in range(40):
            answer = i % 2 == 0
            says = answer if i % 4 != 3 else not answer
            rows.append({"question": f"is item {i} on the list", "answer": answer,
                         "passage": f"Item {i} is described here. The answer is {'yes' if says else 'no'}."})
        return [{"feature_idx": 0, "name": "question", "type": {"dtype": "string", "_type": "Value"}},
                {"feature_idx": 1, "name": "answer", "type": {"dtype": "bool", "_type": "Value"}},
                {"feature_idx": 2, "name": "passage", "type": {"dtype": "string", "_type": "Value"}}], rows
    if dataset == "LocalLLaMA/typed-decisions":
        return [{"feature_idx": i, "name": n, "type": {"dtype": "string", "_type": "Value"}}
                for i, n in enumerate(TYPED_FIELDS)], typed_rows()
    return [], []


TYPED_FIELDS = ["id", "workflow", "split", "state", "questions", "gold", "factors", "label_agreement"]


def typed_rows() -> list[dict]:
    """Twelve cases of five questions in two workflows. The state names the gold `route` in every
    case but the third ones and says the gold `angry` answer in every case but the fourth ones."""
    rows = []
    for i in range(12):
        wf = ("customer_service", "invoice_processing")[i % 2]
        route = ("billing", "shipping", "technical")[i % 3]
        named = route if i % 3 != 2 else ("billing", "shipping", "technical")[(i + 1) % 3]
        angry = i % 2 == 0
        says = angry if i % 4 != 3 else not angry
        state = {"thread": f"Case {i}: about {named}. The answer is {'yes' if says else 'no'}.", "account": {"id": i}}
        questions = {
            "route": {"type": "choice", "instructions": "Which team should take the ticket?",
                      "criteria": {"billing": "Charges and refunds", "shipping": "Deliveries",
                                   "technical": "Bugs and outages"}},
            "angry": {"type": "noul", "instructions": "The customer is angry."},
            "urgency": {"type": "score", "instructions": "How urgent is it?",
                        "criteria": ["can wait", "this week", "today"]},
            "risk": {"type": "score", "instructions": "How risky is it?",
                     "criteria": ["none", "low", "medium", "high"]},
            "action": {"type": "choice", "instructions": "What should happen next?",
                       "criteria": {"close": "Nothing to do", "reply": "Answer the customer"}},
        }
        rp = {r: (0.6 if r == route else 0.2) for r in ("billing", "shipping", "technical")}
        gold = {
            "route": {"type": "choice", "label": route, "probabilities": rp, "confidence": 0.4},
            "angry": {"type": "noul", "label": "true" if angry else "false", "noul": 0.8 if angry else 0.3,
                      "probabilities": {"false": 0.2 if angry else 0.7, "true": 0.8 if angry else 0.3}},
            "urgency": {"type": "score", "label": str(i % 3), "score": float(i % 3),
                        "probabilities": {str(k): (1.0 if k == i % 3 else 0.0) for k in range(3)}},
            "risk": {"type": "score", "label": "1", "score": 1.2,
                     "probabilities": {"0": 0.1, "1": 0.6, "2": 0.2, "3": 0.1}},
            "action": {"type": "choice", "label": "reply", "probabilities": {"close": 0.5, "reply": 0.5}},
        }
        rows.append({"id": f"{wf}_{i:06d}", "workflow": wf, "split": "test", "state": json.dumps(state),
                     "questions": json.dumps(questions), "gold": json.dumps(gold), "factors": "{}",
                     "label_agreement": "{}"})
    return rows


APIBANK_TOOL = '''from apis.api import API


class {name}(API):
    description = "{desc}"
    input_parameters = {{"query": {{"type": "str", "description": "What to look up"}}}}
    output_parameters = {{"result": {{"type": "str", "description": "The answer"}}}}

    def call(self, query):
        return query
'''


def _dialogue(*turns) -> str:
    return "".join(json.dumps(t) + "\n" for t in turns)


# A three-tool API-Bank checkout: two dialogues, three API turns, one of them calling a tool the
# catalog does not have (dropped, as the kit drops it).
APIBANK_FILES = {
    "api-bank/README.md": "API-Bank\n",
    "api-bank/apis/__init__.py": "",
    "api-bank/apis/api.py": "class API:\n    description = ''\n",
    "api-bank/apis/add_alarm.py": APIBANK_TOOL.format(name="AddAlarm", desc="Set an alarm"),
    "api-bank/apis/get_weather.py": APIBANK_TOOL.format(name="GetWeather", desc="Weather for a city"),
    "api-bank/apis/query_stock.py": APIBANK_TOOL.format(name="QueryStock", desc="A stock's price"),
    "api-bank/lv1-lv2-samples/level-1-given-desc/AddAlarm-level-1-1.jsonl": _dialogue(
        {"role": "User", "text": "Wake me at seven."},
        {"role": "API", "api_name": "AddAlarm", "param_dict": {"query": "07:00"}, "result": {"output": "ok"}},
        {"role": "AI", "text": "Done."},
        {"role": "User", "text": "And the weather in Lisbon?"},
        {"role": "API", "api_name": "GetWeather", "param_dict": {"query": "Lisbon"}, "result": {"output": "sun"}}),
    "api-bank/lv1-lv2-samples/level-2-toolsearcher/QueryStock-level-2-1.jsonl": _dialogue(
        {"role": "User", "text": "What is ACME trading at?"},
        {"role": "API", "api_name": "ToolSearcher", "param_dict": {"keywords": "stock"}, "result": {"output": "x"}},
        {"role": "API", "api_name": "QueryStock", "param_dict": {"query": "ACME"}, "result": {"output": "12"}}),
}


def apibank_tree() -> dict:
    tree = [{"path": "api-bank", "type": "tree"}, {"path": "api-bank/apis", "type": "tree"}]
    tree += [{"path": p, "type": "blob"} for p in sorted(APIBANK_FILES)]
    return {"sha": "fake", "tree": tree, "truncated": False}


def gpqa_zip() -> bytes:
    """A six-question GPQA Diamond: question i's correct answer is "right i"."""
    text = io.StringIO()
    w = csv.DictWriter(text, ["Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
                              "Incorrect Answer 3"])
    w.writeheader()
    for i in range(6):
        w.writerow({"Question": f"Question {i}: which is right?", "Correct Answer": f"right {i}",
                    "Incorrect Answer 1": f"wrong {i}a", "Incorrect Answer 2": f"wrong {i}b",
                    "Incorrect Answer 3": f"wrong {i}c"})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("dataset/gpqa_diamond.csv", text.getvalue())
    return buf.getvalue()


GPQA_ZIP = gpqa_zip()


def _words(obj) -> int:
    if isinstance(obj, str):
        return len(obj.split())
    if isinstance(obj, dict):
        return sum(_words(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_words(v) for v in obj)
    return 0


def _unit(*parts) -> float:
    """A deterministic number in [0, 1) from the parts."""
    h = hashlib.sha256("\x00".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def check_questions(questions) -> str | None:
    if not isinstance(questions, dict) or not questions:
        return "questions must be a non-empty object"
    for qid, q in questions.items():
        if not isinstance(q, dict):
            return f"questions.{qid} must be an object"
        if not q.get("instructions"):
            return f'questions.{qid}: "instructions" must be provided'
        t, c = q.get("type"), q.get("criteria")
        if t == "choice" and not (isinstance(c, dict) and c):
            return f"questions.{qid}: a choice needs criteria, an object of options"
        if t == "score" and not (isinstance(c, list) and len(c) >= 2):
            return f"questions.{qid}: a score needs criteria, a list of at least two levels"
        if t == "noul" and c is not None and not (isinstance(c, dict) and set(c) <= {"true", "false"}):
            return f"questions.{qid}: noul criteria are an object of true and false"
        if t not in ("choice", "score", "noul"):
            return f"questions.{qid}: unknown type {t!r}"
    return None


def answer(qid: str, q: dict, state: str) -> dict:
    low = state.lower()
    if q["type"] == "choice":
        options = list(q["criteria"])
        named = [o for o in options if o.lower() in low]
        pick = named[0] if named else options[int(_unit(state, qid) * len(options))]
        rest = (1 - 0.7) / max(1, len(options) - 1)
        probs = {o: (0.7 if o == pick else rest) for o in options} if len(options) > 1 else {pick: 1.0}
        return {"type": "choice", "choice": pick, "probabilities": probs, "confidence": round(max(probs.values()), 4)}
    if q["type"] == "noul":
        if "the answer is yes" in low:
            p = 0.9
        elif "the answer is no" in low:
            p = 0.2
        else:
            p = round(_unit(state, qid), 4)
        return {"type": "noul", "noul": p}
    levels = len(q["criteria"])
    level = int(_unit(state, qid) * levels)
    probs = {str(k): (0.6 if k == level else 0.4 / (levels - 1)) for k in range(levels)}
    return {"type": "score", "score": round(sum(k * v for k, v in enumerate(probs.values())), 4),
            "legend": {str(k): c for k, c in enumerate(q["criteria"])}, "probabilities": probs,
            "confidence": 0.6}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeServer

    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        key = self.server.config.api_key
        if key and self.headers.get("Authorization") != f"Bearer {key}":
            self._send(401, b'{"error": {"message": "invalid api key"}}')
            return False
        return True

    def do_GET(self):
        if self.path == "/debug/memory":
            s = self.server
            gib = s.config.base_gib + s.active_prompt_tokens / 1000 * s.config.gib_per_ktoken
            self._send(200, f"{gib:.3f}\n".encode(), "text/plain")
            return
        if self.path.startswith("/rows?"):
            self.rows()
            return
        if self.path.startswith(("/github/", "/gpqa/")):
            self.source()
            return
        if not self._authorized():
            return
        if self.path == "/v1/models":
            self._send(200, b'{"object": "list", "data": [{"id": "sample-model", "object": "model"}]}')
            return
        self._send(404, b'{"error": {"message": "not found"}}')

    def chunk(self, obj: dict) -> None:
        data = b"data: " + json.dumps(obj).encode() + b"\n\n"
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def rows(self) -> None:
        q = {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items()}
        features, rows = fake_rows(q.get("dataset", ""))
        if not rows:
            self._send(404, b'{"error": "The dataset does not exist."}')
            return
        with self.server.lock:
            self.server.rows_requests += 1
        offset, length = int(q.get("offset", 0)), min(100, int(q.get("length", 100)))
        page = [{"row_idx": i, "row": rows[i], "truncated_cells": []}
                for i in range(offset, min(len(rows), offset + length))]
        self._send(200, json.dumps({"features": features, "rows": page, "num_rows_total": len(rows),
                                    "num_rows_per_page": 100, "partial": False}).encode())

    def source(self) -> None:
        with self.server.lock:
            self.server.source_requests += 1
        parts = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path).split("/")
        if parts[1:3] == ["github", "tree"]:
            self._send(200, json.dumps(apibank_tree()).encode())
        elif parts[1:3] == ["github", "raw"] and "/".join(parts[4:]) in APIBANK_FILES:
            self._send(200, APIBANK_FILES["/".join(parts[4:])].encode(), "text/plain")
        elif self.path == "/gpqa/dataset.zip":
            self._send(200, GPQA_ZIP, "application/zip")
        else:
            self._send(404, b"not found", "text/plain")

    def systemone(self, raw: bytes) -> None:
        try:
            req = json.loads(raw)
        except ValueError:
            req = None
        problem = "the body must be a JSON object" if not isinstance(req, dict) else check_questions(req.get("questions"))
        if problem is None and not isinstance(req.get("state"), (str, dict, list)):
            problem = '"state" must be provided'
        if problem:
            self._send(400, json.dumps({"error": {"code": 400, "message": problem,
                                                  "type": "invalid_request_error"}}).encode())
            return
        cfg, s = self.server.config, self.server
        state = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"])
        tokens = _words(state) + _words(req["questions"]) + 8
        key = hashlib.sha256(state.encode()).digest()
        with s.lock:
            s.active += 1
            s.active_prompt_tokens += tokens
            s.requests.append(req)
            cached = key in s.seen_states
            s.seen_states.add(key)
        try:
            per_token = cfg.decision_per_token * (0.2 if cached else 1.0)
            time.sleep((cfg.decision_base + tokens * per_token) * (1 + cfg.slowdown * (s.active - 1)))
            body = {"model": req.get("model", "sample-model"),
                    "answers": {qid: answer(qid, q, state) for qid, q in req["questions"].items()},
                    "usage": {"input_tokens": tokens, "output_tokens": 0},
                    "capyctl_bench_sample": True}
            self._send(200, json.dumps(body).encode())
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with s.lock:
                s.active -= 1
                s.active_prompt_tokens -= tokens

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        if not self._authorized():
            return
        if self.path == "/v1/systemone":
            self.systemone(raw)
            return
        if self.path != "/v1/chat/completions":
            self._send(404, b'{"error": {"message": "not found"}}')
            return
        req = json.loads(raw)
        cfg = self.server.config
        messages = req.get("messages") or []
        prompt_tokens = prompt_token_count(messages)
        max_tokens = int(req.get("max_tokens") or 16)
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        s = self.server
        with s.lock:
            s.active += 1
            s.active_prompt_tokens += prompt_tokens
            s.requests.append(req)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk", "created": 0,
                    "model": req.get("model", "sample-model"), "capyctl_bench_sample": True}
            time.sleep(cfg.ttft_base + prompt_tokens * cfg.prefill_per_token)
            produced = 0
            digest = hashlib.sha256()
            first = True
            if cfg.cut_after is not None:
                self.chunk(dict(base, choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
                for i in range(cfg.cut_after):
                    self.chunk(dict(base, choices=[{"index": 0, "delta": {"content": f"w{i} "}, "finish_reason": None}]))
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                return
            while produced < max_tokens:
                if not first:
                    time.sleep(cfg.token_delay * cfg.chunk_tokens * (1 + cfg.slowdown * (s.active - 1))
                               * (1 + prompt_tokens / 1000 * cfg.decode_per_ktoken))
                first = False
                n = min(cfg.chunk_tokens, max_tokens - produced)
                text = "".join(f"w{(produced + k) % 97} " for k in range(n))
                digest.update(text.encode())
                if produced < cfg.hidden_tokens:
                    delta = {}
                elif produced < cfg.reasoning_tokens:
                    delta = {"reasoning_content": text, "reasoning": text}
                else:
                    delta = {"content": text}
                if produced == 0:
                    delta["role"] = "assistant"
                produced += n
                self.chunk(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": None}]))
            self.chunk(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "length"}]))
            if include_usage:
                final = dict(base, choices=[], usage={
                    "prompt_tokens": prompt_tokens, "completion_tokens": produced,
                    "total_tokens": prompt_tokens + produced})
                if cfg.extras:
                    drafted = produced * 2
                    final["sample_engine"] = {"drafted": drafted, "accepted": int(produced * 1.2),
                                           "token_sha": digest.hexdigest()[:16]}
                self.chunk(final)
            data = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with s.lock:
                s.active -= 1
                s.active_prompt_tokens -= prompt_tokens


def start(config: FakeConfig | None = None, port: int = 0) -> FakeServer:
    srv = FakeServer(("127.0.0.1", port), config or FakeConfig())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=18443)
    ap.add_argument("--ttft-base", type=float, default=0.02)
    ap.add_argument("--prefill-per-token", type=float, default=2e-5)
    ap.add_argument("--token-delay", type=float, default=0.004)
    ap.add_argument("--slowdown", type=float, default=0.15)
    ap.add_argument("--chunk-tokens", type=int, default=1)
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--base-gib", type=float, default=10.0)
    ap.add_argument("--gib-per-ktoken", type=float, default=0.05)
    ap.add_argument("--decode-per-ktoken", type=float, default=0.0)
    ap.add_argument("--decision-base", type=float, default=0.004)
    ap.add_argument("--decision-per-token", type=float, default=1e-6)
    a = ap.parse_args()
    cfg = FakeConfig(a.ttft_base, a.prefill_per_token, a.token_delay, a.slowdown, a.chunk_tokens,
                     not a.no_extras, 0, a.api_key, a.base_gib, a.gib_per_ktoken,
                     a.decode_per_ktoken, decision_base=a.decision_base,
                     decision_per_token=a.decision_per_token)
    srv = FakeServer(("127.0.0.1", a.port), cfg)
    print(f"fake server on {srv.url}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
